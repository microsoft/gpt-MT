import os
import argparse
import json
import logging
import pandas as pd

COMET_REF_MODELS = ["wmt20-comet-da", "wmt21-comet-mqm", "wmt22-comet-da"]
COMET_SRC_MODELS = ["wmt20-comet-qe-da", "wmt21-comet-qe-mqm", "wmt22-cometkiwi-da"]
DOC_SCORING_SCRIPT = "./doc_score.py"

def count_lines(fname):
  def _make_gen(reader):
    b = reader(2 ** 16)
    while b:
      yield b
      b = reader(2 ** 16)

  with open(fname, "rb") as f:
    count = sum(buf.count(b"\n") for buf in _make_gen(f.raw.read))
  return count

def read_last_line(fname):
  with open(fname, 'rb') as f:
    try:  # catch OSError in case of a one line file 
      f.seek(-2, os.SEEK_END)
      while f.read(1) != b'\n':
        f.seek(-2, os.SEEK_CUR)
    except OSError:
      f.seek(0)
    last_line = f.readline().decode()
    return last_line
  
def is_doc_boundary(doc_ids, idx):
  after_idx = min(len(doc_ids) - 1, idx + 1)
  return (not doc_ids[after_idx] == doc_ids[idx]) or (idx == len(doc_ids) - 1)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--testset', type=str, required=True, help='A path to the test set directory containing references and sources for each language pair. Must contain {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{tgt_lang} and {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{src_lang}')
  parser.add_argument('--docids', type=str, required=False, help='A path to the directory containing doc-ids corresponding to testset for each language pair. Must contain {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.docids')
  parser.add_argument('--hypotheses', type=str, nargs='+', required=True, help='A path to the model output files. must contain {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{tgt_lang}')
  parser.add_argument('--directions', type=str, required=True, nargs='+', help='Language directions to evaluate on e.g. "en-de de-en"')
  parser.add_argument('--comet-models', type=str, required=False, nargs='+', help='A list of COMET models to use for evaluation')
  parser.add_argument('--gpus', type=int, required=False, default=1, help='Number of GPUs to use with COMET')
  parser.add_argument('--metrics', type=str, required=True, nargs='+', help='A list of metrics to use for evaluation, options ["bleu", "comet", "doc-comet", "chrf", "doc-bleu", "doc-chrf"]')
  parser.add_argument('--save-name', type=str, required=False, default='scores', help='name of the output files/folders')
  parser.add_argument('--sliding-window', type=int, required=False, default=1, help='The stride step over document')
  parser.add_argument('--context-length', type=int, required=False, default=4, help='The number of sentences in a single context')
  args = parser.parse_args()
  
  logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )
  
  for hypotheses in args.hypotheses:
    scores = {}
    for direction in args.directions:
      src = direction.split('-')[0]
      tgt = direction.split('-')[1]
      logging.info(f"Evaluating {direction}")

      os.makedirs(f"{hypotheses}/{src}{tgt}/{args.save_name}", exist_ok=True)

      hyp_file = f"{hypotheses}/{src}{tgt}/test.{direction}.{tgt}"
      src_file = f"{args.testset}/{src}{tgt}/test.{direction}.{src}"
      ref_file = f"{args.testset}/{src}{tgt}/test.{direction}.{tgt}"

      hyp_line_count = count_lines(hyp_file)
      src_line_count = count_lines(src_file)
      ref_line_count = count_lines(ref_file)

      assert (ref_line_count == hyp_line_count) and (ref_line_count == src_line_count), f"ref_file = {ref_line_count}, hyp_file = {hyp_line_count}, src_file = {src_line_count} - src/ref/hyp lines count should be matched"

      scores[direction] = {
        "references": ref_file,
        "hypotheses": hyp_file,
        "sources": src_file
      }
      for m in ["chrf", "bleu"]:
        tokenizer = "ja-mecab" if tgt == "ja" else "zh" if tgt == "zh" else "13a"
        if m in args.metrics:
          command = f"sacrebleu -m {m} -tok {tokenizer} {ref_file} < {hyp_file} > {hypotheses}/{src}{tgt}/{args.save_name}/{m}.scores"
          logging.info(command)
          os.system(command)
          with open(f"{hypotheses}/{src}{tgt}/{args.save_name}/{m}.scores", 'r') as score_file:
            score = json.load(score_file)
          # Logging detailed evaluation
          logging.info(f"{direction} {m} scores: {json.dumps(score, indent=2)}")
          scores[direction][m] = score
          
        if f"doc-{m}" in args.metrics:
          assert args.docids, f'document ids directory must be probided to calculate doc-{m}'
          docids_file = f"{args.docids}/{src}{tgt}/test.{direction}.docids"
          scores[direction]["docids"] = docids_file
          docids_line_count = count_lines(docids_file)
          assert docids_line_count == src_line_count, "Doc Ids file line count is not matching"
          with open(src_file, 'r') as f_src, open(ref_file, 'r') as f_ref, open(hyp_file, 'r') as f_hyp, open(docids_file, 'r') as f_docids:
            lines_src = [x.strip() for x in f_src.readlines()]
            lines_ref = [x.strip() for x in f_ref.readlines()]
            lines_hyp = [x.strip() for x in f_hyp.readlines()]
            docid_lines = [x.strip() for x in f_docids.readlines()]            
            assert len(lines_src) == len(docid_lines), "Doc id file lines are not matching"
          docs_src, docs_ref, docs_hyp = [], [], []
          current_doc = []
          i = 0
          while i < len(lines_src):
            current_doc.append({
              'source': lines_src[i],
              'reference': lines_ref[i],
              'hypothesis': lines_hyp[i]
            })
            if is_doc_boundary(docid_lines, i):
              docs_src.append([current_doc[j]['source'] for j in range(len(current_doc))])
              docs_ref.append([current_doc[j]['reference'] for j in range(len(current_doc))])
              docs_hyp.append([current_doc[j]['hypothesis'] for j in range(len(current_doc))])
              current_doc = []
            i += 1
          assert len(docs_src) == len(docs_ref) and len(docs_src) == len(docs_hyp), "docs reconstruction failed"
          tmp_dir = f"{hypotheses}/{src}{tgt}/{args.save_name}/tmp"
          os.makedirs(tmp_dir, exist_ok=True)
          with open(f"{tmp_dir}/test.{direction}.docsnt.src.{src}", 'w') as src_tmp_out,  open(f"{tmp_dir}/test.{direction}.docsnt.ref.{tgt}", 'w') as ref_tmp_out,  open(f"{tmp_dir}/test.{direction}.docsnt.hyp.{tgt}", 'w') as hyp_tmp_out:
            for s_doc, r_doc, h_doc in zip(docs_src, docs_ref, docs_hyp):
              s = ' '.join([x.strip() for x in s_doc]).strip()
              r = ' '.join([x.strip() for x in r_doc]).strip()
              h = ' '.join([x.strip() for x in h_doc]).strip()
              src_tmp_out.write(s + '\n')
              ref_tmp_out.write(r + '\n')
              hyp_tmp_out.write(h + '\n')
          tmp_ref_path = f"{tmp_dir}/test.{direction}.docsnt.ref.{tgt}"
          tmp_hyp_path = f"{tmp_dir}/test.{direction}.docsnt.hyp.{tgt}"
          command = f"sacrebleu -m {m} -tok {tokenizer} {tmp_ref_path} < {tmp_hyp_path} > {hypotheses}/{src}{tgt}/{args.save_name}/doc-{m}.scores"
          logging.info(command)
          os.system(command)
          with open(f"{hypotheses}/{src}{tgt}/{args.save_name}/doc-{m}.scores", 'r') as score_file:
            score = json.load(score_file)
          logging.info(f"{direction} doc-{m} scores: {json.dumps(score, indent=2)}")
          scores[direction][f'doc-{m}'] = score
        
      if "comet" in args.metrics:
        scores[direction]['comet'] = {}
        for model in args.comet_models:
          if model not in COMET_REF_MODELS + COMET_SRC_MODELS:
            logging.info(f"Skipping evaluation using {model} since it is not available")
            continue
          if model in COMET_REF_MODELS:
            command = f"comet-score -s {src_file} -t {hyp_file} -r {ref_file} --gpus {args.gpus} --model {model} > {hypotheses}/{src}{tgt}/{args.save_name}/{model}.scores"
            logging.info(command)
            os.system(command)
            score_line = read_last_line(f"{hypotheses}/{src}{tgt}/{args.save_name}/{model}.scores")
            score = float(score_line.split()[-1])
            scores[direction]['comet'][model] = score
          elif model in COMET_SRC_MODELS:
            command = f"comet-score -s {src_file} -t {hyp_file} --gpus {args.gpus} --model {model} > {hypotheses}/{src}{tgt}/{args.save_name}/{model}.scores"
            logging.info(command)
            os.system(command)
            score_line = read_last_line(f"{hypotheses}/{src}{tgt}/{args.save_name}/{model}.scores")
            score = float(score_line.split()[-1])
            scores[direction]['comet'][model] = score
        logging.info(f"{direction} comet scores: {json.dumps(scores[direction]['comet'], indent=2)}")

      if "doc-comet" in args.metrics:
        assert args.docids, 'document ids directory must be probided to calculate doc-comet'
        docids_file = f"{args.docids}/{src}{tgt}/test.{direction}.docids"
        scores[direction]["docids"] = docids_file
        docids_line_count = count_lines(docids_file)
        assert docids_line_count == src_line_count, "Doc Ids file line count is not matching"
        scores[direction]['doc-comet'] = {}

        for model in ["wmt22-cometkiwi-da"]:
          if model not in COMET_REF_MODELS + COMET_SRC_MODELS:
            logging.info(f"Skipping evaluation using {model} since it is not available")
            continue
          if model in COMET_REF_MODELS:
            command = f"python {DOC_SCORING_SCRIPT} -src {src_file} -hyp {hyp_file} -ref {ref_file} --model {model} --sliding-window {args.sliding_window} --context-length {args.context_length} -doc {docids_file} > {hypotheses}/{src}{tgt}/{args.save_name}/{model}.doclevel.scores"
            logging.info(command)
            os.system(command)
            with open(f"{hypotheses}/{src}{tgt}/{args.save_name}/{model}.doclevel.scores", 'r') as score_file:
              score_json = json.load(score_file)
            score = score_json['score']
            scores[direction]['doc-comet'][model] = score
          elif model in COMET_SRC_MODELS:
            command =f"python {DOC_SCORING_SCRIPT} -src {src_file} -hyp {hyp_file} --model {model} --sliding-window {args.sliding_window} --context-length {args.context_length} -doc {docids_file} > {hypotheses}/{src}{tgt}/{args.save_name}/{model}.doclevel.scores"
            logging.info(command)
            os.system(command)
            with open(f"{hypotheses}/{src}{tgt}/{args.save_name}/{model}.doclevel.scores", 'r') as score_file:
              score_json = json.load(score_file)
            score = score_json['score']
            scores[direction]['doc-comet'][model] = score

    with open(f"{hypotheses}/{args.save_name}.json", 'w') as score_file:
      score_file.write(json.dumps(scores, indent=2))

    scores_csv = {
      'langs': args.directions,
    }
    for metric in args.metrics:
      if metric == 'comet':
        for model in args.comet_models:
          scores_csv[f"{model}"] = []
          for lang in args.directions:
            scores_csv[f"{model}"].append(scores[lang][metric][model])
      elif metric == 'doc-comet':
        for model in ["wmt22-cometkiwi-da"]:
          scores_csv[f"doc-{model}"] = []
          for lang in args.directions:
            scores_csv[f"doc-{model}"].append(scores[lang][metric][model])
      else:
        scores_csv[metric] = []
        for lang in args.directions:
          scores_csv[f"{metric}"].append(scores[lang][metric]["score"])
  
    df = pd.DataFrame(scores_csv)
    logging.info(f"Scores:\n{df}")

    with open(f"{hypotheses}/{args.save_name}.txt", 'w') as score_file:
      print(df, file=score_file)


if __name__ == "__main__":
  main()
  