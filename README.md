<p align="center">
  <img width="600" height="150" src="assets/MT-GPT.png">
</p>
<hr />

## How Good Are GPT Models at Machine Translation? 
## A Comprehensive Evaluation

Paper: https://arxiv.org/abs/2302.09210

## Introduction
In this work, we present a comprehensive evaluation of GPT models for machine translation, covering various aspects such as quality of different GPT models in comparison with state-of-the-art research and commercial systems, effect of prompting strategies, robustness towards domain shifts and document-level translation, all accompanied with an extensive analysis of the differential aspects of translations produced by GPT. We experiment with 18 different translation directions involving high and low resource languages, as well as non English-centric translations, and evaluate the performance of three GPT models: ChatGPT, GPT3.5 (text-davinci-003), and text-davinci-002. We also show that hybrid approaches, which combine GPT models with other translation systems, can further enhance the translation quality.

## Quick Installation
```bash
$ git clone https://github.com/microsoft/gpt-MT.git
$ cd tools
$ conda create -n gpt-mt-eval python=3.10
$ conda activate gpt-mt-eval
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ git clone https://github.com/Unbabel/COMET.git
$ cd COMET
$ git checkout fc2f2b3 
$ poetry install
```

## Data Shots and System Outputs
We have released all selected shots in our experiments including the sentence-level shots (RR, QR and QS) and the document-level shots (DR and DF). These shots have been organized under [data-shots](./data-shots/).

Moreover, To make reproducing all results an easy task, all system outputs have been released under [system-output](./evaluation/system-outputs/) in addition the WMT official test sets along with document-separated and domain-separated files.

## Reproducing Results
To reproduce the reported results in the paper, you need to run the evaluation script [evaluate.py](./tools/evaluate.py).
### CLI Usage:
```bash
$ python evaluate.py -h
usage: evaluate.py [-h] --testset TESTSET [--docids DOCIDS] --hypotheses HYPOTHESES [HYPOTHESES ...] --directions DIRECTIONS [DIRECTIONS ...]
                   [--comet-models COMET_MODELS [COMET_MODELS ...]] [--gpus GPUS] --metrics METRICS [METRICS ...] [--save-name SAVE_NAME]
                   [--sliding-window SLIDING_WINDOW] [--context-length CONTEXT_LENGTH]

options:
  -h, --help            show this help message and exit
  --testset TESTSET     A path to the test set directory containing references and sources for each language pair. Must contain
                        {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{tgt_lang} and {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{src_lang}
  --docids DOCIDS       A path to the directory containing doc-ids corresponding to testset for each language pair. Must contain
                        {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.docids
  --hypotheses HYPOTHESES [HYPOTHESES ...]
                        A path to the model output files. must contain {src_lang}{tgt_lang}/test.{src_lang}-{tgt_lang}.{tgt_lang}
  --directions DIRECTIONS [DIRECTIONS ...]
                        Language directions to evaluate on e.g. "en-de de-en"
  --comet-models COMET_MODELS [COMET_MODELS ...]
                        A list of COMET models to use for evaluation
  --gpus GPUS           Number of GPUs to use with COMET
  --metrics METRICS [METRICS ...]
                        A list of metrics to use for evaluation, options ["bleu", "comet", "doc-comet", "chrf", "doc-bleu", "doc-chrf"]
  --save-name SAVE_NAME
                        name of the output files/folders
  --sliding-window SLIDING_WINDOW
                        The stride step over document
  --context-length CONTEXT_LENGTH
                        The number of sentences in a single context
```
For example:
- To reproduce `GPT 5-Shot QR` results in `Table3`:
```bash
$ cd ./tools
$ python evaluate.py \
    --testset ../evaluation/testset/wmt-testset \
    --directions de-en en-de cs-en en-cs ja-en en-ja zh-en en-zh ru-en en-ru uk-en en-uk is-en en-is ha-en en-ha fr-de de-fr \
    --metrics comet chrf bleu \
    --comet-models wmt22-comet-da wmt22-cometkiwi-da \
    --hypotheses ../evaluation/system-outputs/text-davinci-003/QR/5-shot
``` 
- To reproduce `GPT Doc ZS w=16` results in `Table5`: 
```bash
$ cd ./tools
$ python evaluate.py \
    --testset ../evaluation/testset/wmt-testset \
    --docids ../evaluation/testset/wmt-testset-docids \
    --directions de-en en-de \
    --metrics comet doc-comet chrf bleu doc-bleu \
    --comet-models wmt22-comet-da wmt22-cometkiwi-da \
    --hypotheses ../evaluation/system-outputs/text-davinci-003-doc-level/Doc-W16/zeroshot
``` 

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{gpt-mt-2023,
      title={How Good Are GPT Models at Machine Translation? A Comprehensive Evaluation}, 
      author={Amr Hendy and Mohamed Abdelrehim and Amr Sharaf and Vikas Raunak and Mohamed Gabr and Hitokazu Matsushita and Young Jin Kim and Mohamed Afify and Hany Hassan Awadalla},
      journal={arXiv preprint arXiv:2302.09210},
      year={2023}
}
```    

