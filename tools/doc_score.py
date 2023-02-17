import numpy as np
import argparse
import json
import os

from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer

COMET_REF_MODELS = ["wmt20-comet-da", "wmt21-comet-mqm", "wmt22-comet-da"]
COMET_SRC_MODELS = ["wmt20-comet-qe-da", "wmt21-comet-qe-mqm", "wmt22-cometkiwi-da"]

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")

def _is_doc_boundary(doc_ids, idx):
  after_idx = min(len(doc_ids) - 1, idx + 1)
  return (not doc_ids[after_idx] == doc_ids[idx]) or (idx == len(doc_ids) - 1)

def _build_context(doc, current_idx, context_window, start_left=True):        
  balance = context_window
  low = current_idx if start_left else max([0, current_idx - (context_window // 2)])
  balance -= (current_idx - low)
  high = min([len(doc), current_idx + balance])
  balance -= (high - current_idx)
  low = max([0, low - balance])
  pos = current_idx - low
  return doc[low:high], pos

def _check_max_tokens(src_context, mt_context, ref_context=None, max_tokens=512):
  src = " ".join(src_context).strip()
  mt = " ".join(mt_context).strip()
  if ref_context:
    ref = " ".join(ref_context).strip()
    full_input = tokenizer(" </s> ".join([src, mt, ref])).input_ids
  else:
    full_input = tokenizer(" </s> ".join([src, mt])).input_ids
  return len(full_input) < max_tokens

def _calculate_doc_comet(args, model, src_docs, hyp_docs, ref_docs=None):
  scores, doc_lengths = [], []
  if ref_docs:
    for s, h, r in zip(src_docs, hyp_docs, ref_docs):
      data_for_eval = []
      # Check if the doc has length shorter than the context length
      if len(s) <= args.context_length:
        data_for_eval.append({"src": " ".join(s).strip(), "mt": " ".join(h).strip(), "ref": " ".join(r).strip()})
      else:
        prev_context_src, prev_context_mt, prev_context_ref = [], [], []
        for i in range(len(s)):
          src_context, _ = _build_context(s, i, args.context_length)
          mt_context, _ = _build_context(h, i, args.context_length)
          ref_context, _ = _build_context(r, i, args.context_length)

          # Ensure max_tokens is respected
          reduce = 1
          while (not _check_max_tokens(src_context, mt_context, ref_context=ref_context)) and (args.context_length - reduce > 1):
            src_context, _ = _build_context(s, i, args.context_length - reduce)
            mt_context, _ = _build_context(h, i, args.context_length - reduce)
            ref_context, _ = _build_context(r, i, args.context_length - reduce)
            reduce += 1

          # Ensure same context is not evaluated twice
          if not src_context == prev_context_src and not mt_context == prev_context_mt and not ref_context == prev_context_ref:
            src, mt, ref =  " ".join(src_context).strip(), " ".join(mt_context).strip(), " ".join(ref_context).strip()
            data_for_eval.append({
              "src": src, "mt": mt, "ref": ref
            })
          prev_context_src, prev_context_mt, prev_context_ref = src_context, mt_context, ref_context

      # Compute the score
      pred = model.predict(data_for_eval, batch_size=8, gpus=1) 
      scores.append(pred.system_score)
      doc_lengths.append(len(s))

  else:
    for s, h in zip(src_docs, hyp_docs):
      data_for_eval = [] 
      # Check if the doc has length shorter than the context length
      if len(s) <= args.context_length:
        data_for_eval.append({"src": " ".join(s).strip(), "mt": " ".join(h).strip()})
      else:
        prev_context_src, prev_context_mt = [], []
        for i in range(len(s)):
          src_context, _ = _build_context(s, i, args.context_length)
          mt_context, _ = _build_context(h, i, args.context_length)
          
          # Ensure max_tokens is respected
          reduce = 1
          while (not _check_max_tokens(src_context, mt_context)) and (args.context_length - reduce > 1):
            src_context, _ = _build_context(s, i, args.context_length - reduce)
            mt_context, _ = _build_context(h, i, args.context_length - reduce)
            reduce += 1
          
          # Ensure same context is not evaluated twice
          if not src_context == prev_context_src and not mt_context == prev_context_mt:
            src, mt =  " ".join(src_context).strip(), " ".join(mt_context).strip()
            data_for_eval.append({
              "src": src, "mt": mt
            })
          prev_context_src, prev_context_mt = src_context, mt_context 

      # Compute the score
      pred = model.predict(data_for_eval, batch_size=8, gpus=1) 
      scores.append(pred.system_score) # type: ignore
      doc_lengths.append(len(s))

  return scores, doc_lengths

def _load_data(args):
  with open(args.sources_file, 'r') as src_file, open(args.hypotheses_file, 'r') as hyp_file, open(args.docids_file, 'r') as docids_file:
    sources = src_file.readlines()
    hypotheses = hyp_file.readlines()
    docids = docids_file.readlines()
    
    src_docs, hyp_docs, ref_docs = [], [], None
    current_src_doc, current_hyp_doc = [], []
    i = 0
    while i < len(docids):
      current_src_doc.append(sources[i].strip())
      current_hyp_doc.append(hypotheses[i].strip())
      if _is_doc_boundary(docids, i):
        src_docs.append(current_src_doc)
        hyp_docs.append(current_hyp_doc)
        current_src_doc, current_hyp_doc = [], []
      i += 1
      
    if args.references_file:
      # Load reference files
      with open(args.references_file, 'r') as ref_file:
        references = ref_file.readlines()
        ref_docs = []
        current_ref_doc = []
        i = 0
        while i < len(docids):
          current_ref_doc.append(references[i].strip())
          if _is_doc_boundary(docids, i):
            ref_docs.append(current_ref_doc)
            current_ref_doc = []
          i += 1
            
  return src_docs, hyp_docs, ref_docs
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sources-file', '-src', type=str, required=True, help='A path to the source file')
  parser.add_argument('--hypotheses-file', '-hyp', type=str, required=True, help='A path to the model output file')
  parser.add_argument('--references-file', '-ref', type=str, required=False, help='A path to the reference file')
  parser.add_argument('--docids-file', '-doc', type=str, required=True, help='A path to the doc-ids file')
  parser.add_argument('--model', type=str, required=True, help='The COMET model name used for automatic evaluation')
  parser.add_argument('--sliding-window', type=int, required=False, default=1, help='The stride step over document')
  parser.add_argument('--context-length', type=int, required=False, default=4, help='The number of sentences in a single context')
  args = parser.parse_args()
  
  comet_model_path = download_model(args.model)
  model = load_from_checkpoint(comet_model_path)
  
  if args.references_file:
    assert args.model in COMET_REF_MODELS, f"Reference files should not be passed for evaluating {COMET_SRC_MODELS}"
  else:
    assert args.model not in COMET_REF_MODELS, f"Reference files are required for evaluating {COMET_REF_MODELS}"
  
  src_docs, mt_docs, ref_docs = _load_data(args)
  scores, _ = _calculate_doc_comet(args, model, src_docs, mt_docs, ref_docs) 
  
  ret = {
    'model': args.model,
    'sources_file': args.sources_file, 
    'mt_file': args.hypotheses_file,
    'sliding_window': args.sliding_window,
    'context_length': args.context_length,
    'score': np.mean(scores)
  }
  
  print(json.dumps(ret, indent=2))


if __name__ == "__main__":
  main()
