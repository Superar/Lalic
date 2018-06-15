OPEN_NMT_PATH=$HOME/bin/OpenNMT-py
DATA_PATH=../Corpus_FAPESP_v2
SRC_LAN=en
TGT_LAN=pt
TRAIN_PATH=$DATA_PATH/corpus_treinamento/$SRC_LAN-$TGT_LAN
TEST_PATH=$DATA_PATH/corpus_teste/$SRC_LAN-$TGT_LAN

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
    -train_src $TRAIN_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.train.$SRC_LAN.atok \
    -train_tgt $TRAIN_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.train.$TGT_LAN.atok \
    -valid_src $TRAIN_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.dev.$SRC_LAN.atok \
    -valid_tgt $TRAIN_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.dev.$TGT_LAN.atok \
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
    -src $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$SRC_LAN.atok \
    -tgt $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$TGT_LAN.atok \
    -replace_unk \
    -output $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.output

python3 $OPEN_NMT_PATH/translate.py \
    -gpu 0 \
    -model $DATA_PATH/nmt_model/nmt_model_*_e13.pt \
    -src $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$SRC_LAN.atok \
    -tgt $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$TGT_LAN.atok \
    -replace_unk \
    -output $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.output

# Calculo BLEU
echo "Calculando BLEU"
# test-a
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$SRC_LAN.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$SRC_LAN.atok src $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$SRC_LAN.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$TGT_LAN.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$TGT_LAN.atok ref $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$TGT_LAN.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.output.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.output test $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.output.sgm;
fi
perl mteval-v14.pl -r $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$TGT_LAN.sgm \
                   -s $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.$SRC_LAN.sgm \
                   -t $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-a.output.sgm > $TEST_PATH/test-a.bleu

# test-b
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$SRC_LAN.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$SRC_LAN.atok src $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$SRC_LAN.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$TGT_LAN.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$TGT_LAN.atok ref $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$TGT_LAN.sgm;
fi
if [ ! -f $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.output.sgm ]; then
perl formata-mteval-v14.pl $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.output test $SRC_LAN $TGT_LAN $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.output.sgm;
fi
perl mteval-v14.pl -r $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$TGT_LAN.sgm \
                   -s $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.$SRC_LAN.sgm \
                   -t $TEST_PATH/fapesp-v2.$SRC_LAN-$TGT_LAN.test-b.output.sgm > $TEST_PATH/test-b.bleu
