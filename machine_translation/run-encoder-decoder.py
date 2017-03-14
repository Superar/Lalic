# coding=utf-8
import tensorflow as tf
import re
import codecs
import numpy as np


def recreate_dataset(path):
    with codecs.open(path, encoding='utf-8') as file_:
        #print(file_.readline())
        #data = file_.readline().split()
        dictionary = {}
        reverse_dictionary = {}
        rev = False

        for line in file_.readlines():
            line = line.rstrip()
            print(line)
            if rev:
                split_line = re.split('@@', line, flags=re.UNICODE)
                reverse_dictionary[split_line[0]] = split_line[1]
            elif line == 'REVERSE_DICTIONARY':
                rev = True
            elif line == 'DATA':
                data = file_.readline().split()
            else:
                split_line = re.split('@@', line, flags=re.UNICODE)
                dictionary[split_line[0]] = split_line[1]

    return data, dictionary, reverse_dictionary


def process_text(text, dict_):
    text_list = []
    for t in re.split('\W+', text, flags=re.UNICODE):
        if t not in dict_.keys():
            text_list.append(dict_['UKN'])
        else:
            text_list.append((dict_[t]))
    return text_list


seq_length = 128
batch_size = 128

data_pt, dict_pt, rev_dict_pt = recreate_dataset('tradutor/vocab_pt')

sess = tf.Session()
# new_saver = tf.train.import_meta_graph('tradutor/modelo-encoder-decoder.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))

print('Modelo carregado')

inp = 'O chá impede a agregação dos cristais de oxalato de cálcio, o componente químico mais comum das pedras.'
processed_data = process_text(inp, dict_pt)
processed_data.extend([0] * (seq_length - len(processed_data)))
processed_data = np.array(processed_data).T
print(processed_data)

# feed_dict = {enc_inp[t]: processed_data[t] for t in range(seq_length)}
#
# dec_outputs_batch = sess.run(dec_outputs, feed_dict)
# print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])
