MUSEPATH="$HOME/bin/MUSE"

python3 $MUSEPATH/unsupervised.py --src_lang en \
                                  --tgt_lang pt \
                                  --src_emb fapesp-pt-en.bitexts.en.model \
                                  --tgt_emb fapesp-pt-en.bitexts.pt.model \
                                  --emb_dim 100 \
                                  --dis_most_frequent 25000 # Frequencia minima das palavras

# cg90dhz7kg