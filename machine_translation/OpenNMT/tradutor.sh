# Download do tokenizador moses
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
# sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

# Faz download dos dados wmt
# mkdir -p data/multi30k
# wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz &&  tar -xf training.tar.gz -C data/multi30k && rm training.tar.gz
# wget http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz && tar -xf validation.tar.gz -C data/multi30k && rm validation.tar.gz
# wget https://staff.fnwi.uva.nl/d.elliott/wmt16/mmt16_task1_test.tgz && tar -xf mmt16_task1_test.tgz -C data/multi30k && rm mmt16_task1_test.tgz

# DATA_PATH=data/multi30k
# SRC_LAN=en
# TGT_LAN=de
DATA_PATH=../../Corpus_FAPESP_pt-en_bitexts
SRC_LAN=en
TGT_LAN=pt

# Retira ultima linha dos arquivos que nao sao de teste
# Porque eles terminam com duas linhas em branco por algum motivo
# for l in $SRC_LAN $TGT_LAN
# do
#     for f in $DATA_PATH/*.$l
#     do
#         if [[ "$f" != *"test"* ]]
#         then
#             sed -i "$ d" $f
#         fi
#     done
# done

# Tokeniza textos
for l in $SRC_LAN $TGT_LAN
do
    for f in $DATA_PATH/*.$l
    do
        perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok
    done
done

python3 preprocess.py \
    -train_src $DATA_PATH/fapesp-bitexts.pt-en.en.atok \
    -train_tgt $DATA_PATH/fapesp-bitexts.pt-en.pt.atok \
    -valid_src $DATA_PATH/fapesp-bitexts.pt-en.en.atok \
    -valid_tgt $DATA_PATH/fapesp-bitexts.pt-en.pt.atok \
    -save_data $DATA_PATH/fapesp-bitexts.atok.low \
    -lower

python3 train.py \
    -data $DATA_PATH/fapesp-bitexts.atok.low.train.pt \
    -save_model $DATA_PATH/tradutor_model \
    -gpus 0

python3 translate.py \
    -gpu 0 \
    -model $DATA_PATH/tradutor_model_e13_*.pt \
    -src $DATA_PATH/fapesp-bitexts.pt-en.en.atok \
    -tgt $DATA_PATH/fapesp-bitexts.pt-en.pt.atok \
    -replace_unk \
    -verbose \
    -output $DATA_PATH/tradutor.test.pred.atok
