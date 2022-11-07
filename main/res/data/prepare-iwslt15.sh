#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com.cnpmjs.org/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com.cnpmjs.org/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src=fr
tgt=en
lang=fr-en
prep=iwslt15.tokenized.fr-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT15.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done

echo "creating train, valid, test..."
for l in $src $tgt; do
    cat $tmp/train.tags.fr-en.$l > $tmp/train.$l

    cat $tmp/IWSLT15.TED.dev2010.fr-en.$l \
        $tmp/IWSLT15.TED.tst2010.fr-en.$l \
        $tmp/IWSLT15.TED.tst2011.fr-en.$l \
        $tmp/IWSLT15.TED.tst2012.fr-en.$l \
        $tmp/IWSLT15.TED.tst2013.fr-en.$l \
        > $tmp/valid.$l

    cat $tmp/IWSLT15.TED.tst2014.fr-en.$l \
        $tmp/IWSLT15.TED.tst2015.fr-en.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.en-fr
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
