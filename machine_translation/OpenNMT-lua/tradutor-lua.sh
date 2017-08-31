### CONFIGURACOES
# Todos os diretorios estao relativos ao PROJ_DIR_PATH
PROJ_DIR_PATH=fapesp-en-pt # Diretorio do projeto
DATA_PATH=../../Corpus_FAPESP_v2 # Diretorio da base de dados
TRAIN_PATH=corpus_treinamento/pt-en # Diretorio dos dados para treinamento
TEST_PATH=corpus_teste/pt-en # Diretorio dos dados para teste
TOOLS_PATH=moses_tools # Diretorio com ferramentas e utilitarios
OPENNMT_PATH=OpenNMT
SRC_LAN=en # Lingua fonte
TGT_LAN=pt # Lingua alvo


### CRIACAO DO PROJETO
# mkdir -p $PROJ_DIR_PATH
cd $PROJ_DIR_PATH
# mkdir -p $TRAIN_PATH
# mkdir -p $TEST_PATH
mkdir -p $TOOLS_PATH


### DOWNLOADS
# Download e instalacao do torch
# git clone https://github.com/torch/distro.git ~/torch --recursive
# cd ~/torch; bash install-deps;
# ./install.sh

# Download e instalacao de utilidades do Moses
# cd $TOOLS_PATH
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
# wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.pt
# sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
# cd ..

# Download e instalacao do toolkit OpenNMT
# git clone https://github.com/OpenNMT/OpenNMT.git
# luarocks install tds


### PREPROCESSAMENTO

# Substitui `` e '' por "
echo "Substituindo \`\` e ''"
sed -i "s/\`\`/\"/g" $DATA_PATH/$TRAIN_PATH/*.en
sed -i "s/''/\"/g" $DATA_PATH/$TRAIN_PATH/*.en

# Tokeniza textos
echo "Tokenizando textos"
for l in $SRC_LAN $TGT_LAN
do
  for f in $DATA_PATH/$TRAIN_PATH/*.$l
  do
    # perl $TOOLS_PATH/tokenizer.perl -a -no-escape -l $l -q  < $DATA_PATH/$f > $f.atok
  done

  # for f in $TEST_PATH/*.$l
  # do
  #   perl $TOOLS_PATH/tokenizer.perl -a -no-escape -l $l -q < $DATA_PATH/$f > $f.atok
  # done
done
