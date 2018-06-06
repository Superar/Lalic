OPEN_NMT_PATH=$HOME/bin/OpenNMT-py
DATA_PATH=../Corpus_FAPESP_v2
TRAIN_PATH=$DATA_PATH/corpus_treinamento/pt-en
TEST_PATH=$DATA_PATH/corpus_teste/pt-en
SRC_LAN=en
TGT_LAN=pt

mkdir -p $DATA_PATH/nmt_model $TRAIN_PATH/preprocessed

# Download do tokenizador moses
echo "Instalando moses"
if [ ! -f tokenizer.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl;
fi

for l in $SRC_LAN $TGT_LAN
do
    if [ ! -f nonbreaking_prefix.$l ]; then
        wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.$l;
    fi
done
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl

if [ ! -f detokenizer.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl;
fi

if [ ! -f mteval-v14.pl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/mteval-v14.pl;
fi

if [ ! -f multi-bleu.perl ]; then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl;
fi

# Tokeniza textos
echo "Tokenizando textos"
for l in $SRC_LAN $TGT_LAN
do
    for f in $TRAIN_PATH/*.$l
    do
        if [ ! -f $f.atok ]; then
            sed -i "s/\`\`/\"/g" $f
            sed -i "s/\`/'/g" $f
            sed -i "s/''/\"/g" $f
            sed -i "s/--/-/g" $f
            perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok;
        fi
    done

    for f in $TEST_PATH/*.$l
    do
        if [ ! -f $f.atok ]; then
            sed -i "s/\`\`/\"/g" $f
            sed -i "s/\`/'/g" $f
            sed -i "s/''/\"/g" $f
            sed -i "s/--/-/g" $f
            perl tokenizer.perl -a -no-escape -l $l -q < $f > $f.atok;
        fi
    done
done

# Pre-processamento
echo "Pre-processando"
python3 $OPEN_NMT_PATH/preprocess.py \
    -train_src $TRAIN_PATH/fapesp-v2.pt-en.train.en.atok \
    -train_tgt $TRAIN_PATH/fapesp-v2.pt-en.train.pt.atok \
    -valid_src $TRAIN_PATH/fapesp-v2.pt-en.dev.en.atok \
    -valid_tgt $TRAIN_PATH/fapesp-v2.pt-en.dev.pt.atok \
    -save_data $TRAIN_PATH/preprocessed/fapesp-v2.atok.low \
    -lower

# Treinamento
echo "Treinando"
python3 $OPEN_NMT_PATH/train.py \
    -data $TRAIN_PATH/preprocessed/fapesp-v2.atok.low \
    -save_model $DATA_PATH/nmt_model/nmt_model \
    -gpuid 0 \
    -word_vec_size 300 \
    -pre_word_vecs_enc vectors-en-torch.enc.pt \
    -pre_word_vecs_dec vectors-en-torch.dec.pt

# Teste
echo "Testando"
python3 $OPEN_NMT_PATH/translate.py \
    -gpu 0 \
    -model $DATA_PATH/nmt_model/nmt_model_*_e13.pt \
    -src $TEST_PATH/fapesp-v2.pt-en.test-a.en.atok \
    -tgt $TEST_PATH/fapesp-v2.pt-en.test-a.pt.atok \
    -replace_unk \
    -output $TEST_PATH/fapesp-v2.pt-en.test-a.output

python3 $OPEN_NMT_PATH/translate.py \
    -gpu 0 \
    -model $DATA_PATH/nmt_model/nmt_model_*_e13.pt \
    -src $TEST_PATH/fapesp-v2.pt-en.test-b.en.atok \
    -tgt $TEST_PATH/fapesp-v2.pt-en.test-b.pt.atok \
    -replace_unk \
    -output $TEST_PATH/fapesp-v2.pt-en.test-b.output

# Destokenizar
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-a.detok ]; then
    perl detokenizer.perl -l pt < $TEST_PATH/fapesp-v2.pt-en.test-a.output > $TEST_PATH/fapesp-v2.pt-en.test-a.detok;
fi
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-b.detok ]; then
    perl detokenizer.perl -l pt < $TEST_PATH/fapesp-v2.pt-en.test-b.output > $TEST_PATH/fapesp-v2.pt-en.test-b.detok;
fi

# Calculo BLEU
echo "Calculando BLEU"
# test-a
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-a.en.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-a.en src en pt $TEST_PATH/fapesp-v2.pt-en.test-a.en.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-a.pt.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-a.pt ref en pt $TEST_PATH/fapesp-v2.pt-en.test-a.pt.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-a.detok.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-a.detok test en pt $TEST_PATH/fapesp-v2.pt-en.test-a.detok.sgm;
fi
perl mteval-v14.pl -r $TEST_PATH/fapesp-v2.pt-en.test-a.pt.sgm \
                   -s $TEST_PATH/fapesp-v2.pt-en.test-a.en.sgm \
                   -t $TEST_PATH/fapesp-v2.pt-en.test-a.detok.sgm > $TEST_PATH/test-a.bleu

# test-b
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-b.en.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-b.en src en pt $TEST_PATH/fapesp-v2.pt-en.test-b.en.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-b.pt.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-b.pt ref en pt $TEST_PATH/fapesp-v2.pt-en.test-b.pt.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.pt-en.test-b.detok.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.pt-en.test-b.detok test en pt $TEST_PATH/fapesp-v2.pt-en.test-b.detok.sgm;
fi
perl mteval-v14.pl -r $TEST_PATH/fapesp-v2.pt-en.test-b.pt.sgm \
                   -s $TEST_PATH/fapesp-v2.pt-en.test-b.en.sgm \
                   -t $TEST_PATH/fapesp-v2.pt-en.test-b.detok.sgm > $TEST_PATH/test-b.bleu
