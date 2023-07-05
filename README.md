# Systematic-Generalization-via-Meaningful-Learning
This repository is for the paper [Revisit Systematic Generalization via Meaningful Learning](https://aclanthology.org/2022.blackboxnlp-1.6). *In Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP*, pages 62–79, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

[[arXiv](https://arxiv.org/abs/2003.06658)] [[Poster](https://www.shininglab.ai/assets/posters/Revisit%20Systematic%20Generalization%20via%20Meaningful%20Learning.pdf)]

## Directory
+ **main/config.py** - Configurations
+ **main/res** - Resources including model check points, datasets, experiment records, and results
+ **main/src** - Source code including model structures and utility functions
```
Systematic-Generalization-via-Meaningful-Learning
├── README.md
├── main
│   ├── config.py
│   ├── res
│   │   ├── check_points
│   │   ├── data
│   │   │   ├── scan
│   │   │   ├── geography
│   │   │   ├── advising
│   │   │   ├── geo_vars.txt
│   │   │   ├── adv_vars.txt
│   │   │   ├── iwslt14
│   │   │   ├── iwslt15
│   │   │   ├── prepare-iwslt14.sh
│   │   │   └── prepare-iwslt15.sh
│   │   ├── log
│   │   └── result
│   ├── src
│   │   ├── models
│   │   └── utils
│   └── train.py
└── requirements.txt
```

## Dependencies
+ python >= 3.10.6
+ tqdm >= 4.64.1
+ numpy >= 1.23.4
+ torch >= 1.13.0

## Data
All datasets can be downloaded [here](https://drive.google.com/drive/folders/19vFBn5C-nTdjxMeuMgw-BvsPNsLF6DpV?usp=sharing) and should be placed under **main/res/data** according to specific tasks. Please refer to [text2sql-data](https://github.com/jkkummerfeld/text2sql-data/tree/master/data) for details.
* main/res/data/scan
* main/res/data/geography
* main/res/data/advising

### Notes
+ main/res/data/iwslt14 - both vocabulary augmentation set and the entire dataset for IWSLT14
+ main/res/data/iwslt15 - both vocabulary augmentation set and the entire dataset for IWSLT15
+ main/res/data/prepare-iwslt14.sh - [fairseq](https://github.com/facebookresearch/fairseq) preprocess script for IWSLT14
+ main/res/data/prepare-iwslt15.sh - [fairseq](https://github.com/facebookresearch/fairseq) preprocess script for IWSLT15
+ main/res/data/geo_vars.txt - the entity augmentation set for Grography
+ main/res/data/adv_vars.txt - the entity augmentation set for Advising

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```
$ cd Systematic-Generalization-via-Meaningful-Learning
$ cd main
$ pip install pip --upgrade
$ pip install -r requirements.txt
```

## Run
Before training, please double check **config.py** to ensure training configurations.
```
$ vim config.py
$ python train.py
```

## Outputs
If everything goes well, there should be a similar progressing shown as below.
```
Initialize...
*Configuration*
model name: bi_lstm_rnn_att
trainable parameters:5,027,337
...
Training...
Loss:1.7061: 100%|██████████| 132/132 [00:14<00:00,  9.37it/s]
Train Epoch 0 Total Step 132 Loss:1.9179
...
```

## NMT
We use [fairseq](https://github.com/facebookresearch/fairseq) for NMT tasks in Section 4.1. Please find the example pipeline shown below.

### Models
+ LSTM - lstm_luong_wmt_en_de
+ Transformer - transformer_iwslt_de_en
+ Dynamic Conv. - lightconv_iwslt_de_en

### BPE
```
examples/translation/subword-nmt/apply_bpe.py -c iwslt14.tokenized.de-en/code <iwslt14.tokenized.de-en/iwslt14.vocab.en> iwslt14.tokenized.de-en/iwslt14.vocab.en.bpe
```

### Preprocessing
```
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### Training
LSTM
```
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    -s en -t de \
    --arch lstm_luong_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.2 --weight-decay 0.0 \
    --encoder-dropout-out 0.2 --decoder-dropout-out 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 32768 \
    --fp16 --no-epoch-checkpoints >train.log 2>&1 &
```
Transformer
```
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    -s en -t de \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 32768 \
    --fp16 --no-epoch-checkpoints >train.log 2>&1 &
```
Dynamic Conv.
```
fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    -s en -t de \
    --arch lightconv_iwslt_de_en \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 32768 \
    --fp16 --no-epoch-checkpoints >train.log 2>&1 &
```

### Evaluation
BLEU
```
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    -s en -t de \
    --batch-size 128 --beam 5 --lenpen 0.6 \
    --scoring bleu --remove-bpe --cpu >bleu.log 2>&1 &
```
ScareBLEU
```
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    -s en -t de \
    --batch-size 128 --beam 5 --lenpen 0.6 \
    --scoring sacrebleu --remove-bpe --cpu >sacrebleu.log 2>&1 &
```

## Authors
* **Ning Shi** - mrshininnnnn@gmail.com

## BibTex
```
@inproceedings{shi-etal-2022-revisit,
    title = "Revisit Systematic Generalization via Meaningful Learning",
    author = "Shi, Ning  and
      Wang, Boxin  and
      Wang, Wei  and
      Liu, Xiangyu  and
      Lin, Zhouhan",
    booktitle = "Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.blackboxnlp-1.6",
    pages = "62--79",
    abstract = "Humans can systematically generalize to novel compositions of existing concepts. Recent studies argue that neural networks appear inherently ineffective in such cognitive capacity, leading to a pessimistic view and a lack of attention to optimistic results. We revisit this controversial topic from the perspective of meaningful learning, an exceptional capability of humans to learn novel concepts by connecting them with known ones. We reassess the compositional skills of sequence-to-sequence models conditioned on the semantic links between new and old concepts. Our observations suggest that models can successfully one-shot generalize to novel concepts and compositions through semantic linking, either inductively or deductively. We demonstrate that prior knowledge plays a key role as well. In addition to synthetic tests, we further conduct proof-of-concept experiments in machine translation and semantic parsing, showing the benefits of meaningful learning in applications. We hope our positive findings will encourage excavating modern neural networks{'} potential in systematic generalization through more advanced learning schemes.",
}
```
