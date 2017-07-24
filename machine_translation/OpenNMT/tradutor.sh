#Install python torch
# pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
# pip3 install torchvision

# Download do tokenizador moses
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
# sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

DATA_PATH=../Corpus_FAPESP_v2
TRAIN_PATH=$DATA_PATH/corpus_treinamento/pt-en
TEST_PATH=$DATA_PATH/corpus_teste/pt-en
SRC_LAN=en
TGT_LAN=pt

mkdir -p $DATA_PATH/nmt_model $TRAIN_PATH/preprocessed

# Substitui `` e '' por ""
sed -i "s/\`\`/\"/g" $TRAIN_PATH/*.en
sed -i "s/''/\"/g" $TRAIN_PATH/*.en

# Tokeniza textos
for l in $SRC_LAN $TGT_LAN
do
    for f in $TRAIN_PATH/*.$l
    do
        perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok
    done

    for f in $TEST_PATH/*.$l
    do
      perl tokenizer.perl -a -no-escape -l $l -q < $f > $f.atok
    done
done

# Pre-processamento
python3 preprocess.py \
    -train_src $TRAIN_PATH/fapesp-v2.pt-en.train.en.atok \
    -train_tgt $TRAIN_PATH/fapesp-v2.pt-en.train.pt.atok \
    -valid_src $TRAIN_PATH/fapesp-v2.pt-en.dev.en.atok \
    -valid_tgt $TRAIN_PATH/fapesp-v2.pt-en.dev.pt.atok \
    -save_data $TRAIN_PATH/preprocessed/fapesp-v2.atok.low \
    -lower

# Treinamento
python3 train.py \
    -data $TRAIN_PATH/preprocessed/fapesp-v2.atok.low.train.pt \
    -save_model $DATA_PATH/nmt_model/fapesp-v2_model \
    -gpus 0

# Teste
python3 translate.py \
    -gpu 0 \
    -model $DATA_PATH/nmt_model/1/fapesp-v2_model_*_e13.pt \
    -src $TEST_PATH/fapesp-v2.pt-en.test-a.en.atok \
    -tgt $TEST_PATH/fapesp-v2.pt-en.test-a.pt.atok \
    -replace_unk \
    -verbose \
    -output $TEST_PATH/fapesp-v2.pt-en.test-a.output

python3 translate.py \
    -gpu 0 \
    -model $DATA_PATH/nmt_model/fapesp-v2_model_*_e13.pt \
    -src $TEST_PATH/fapesp-v2.pt-en.test-b.en.atok \
    -tgt $TEST_PATH/fapesp-v2.pt-en.test-b.pt.atok \
    -replace_unk \
    -verbose \
    -output $TEST_PATH/fapesp-v2.pt-en.test-b.output

# Calculo BLEU
perl multi-bleu.perl $TEST_PATH/fapesp-v2.pt-en.test-a.pt.atok < $TEST_PATH/fapesp-v2.pt-en.test-a.output > $TEST_PATH/teste-a.bleu
perl multi-bleu.perl $TEST_PATH/fapesp-v2.pt-en.test-b.pt.atok < $TEST_PATH/fapesp-v2.pt-en.test-b.output > $TEST_PATH/teste-b.bleu
