import tensorflow as tf
import pandas as pd
from seq2seq_voice.seq2seq import *
from seq2seq_voice.data_loda import *
import random

# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 128
# Number of Layers
rnn_num_layers = 1
# Embedding Size
encoder_embedding_size = 100
decoder_embedding_size = 100
# Learning Rate
lr = 0.001
# 每50轮打一次结果
display_step = 50

#'D:/Python_work/worm/voice_cut.csv'
cut_file = '../file/voice_cut_n_v.csv'
file = pd.read_csv(cut_file, ',', encoding='utf-8-sig')
source_vocab_to_int, target_vocab_to_int, source_list,target_list = word_vocab(file)


def sentence_to_seq(sentence, source_vocab_to_int):
    """
    将句子转化为数字编码
    """
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence]

    return word_idx


loaded_graph = tf.Graph()
id_list, poi_list = [], []
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints_poi_n_v_clean/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints_poi_n_v_clean'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    for i in range(4000):
        translate_sentence_text = file['voice'][i].split(',')

        translate_sentence = sentence_to_seq(translate_sentence_text, source_vocab_to_int)

        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size})[0]
        #
        # print('='*100)
        # print('【title】')
        # print(file['title'][int_random])
        #
        # print('【Input】')
        # print('  title Words: {}'.format([source_list[i] for i in translate_sentence]))
        #
        # print('【target】')
        # print(' target : %s' % file['poi'][int_random])
        #
        # # print('\n【Prediction】')
        # # print('  Word Ids:      {}'.format([i for i in translate_logits]))
        # # print('  poi Words: {}'.format([target_list[i] for i in translate_logits]))
        #
        # print("\n【Full Sentence】")
        # print(" ".join([target_list[i] for i in translate_logits]))
        #         print('=' * 100)
        id_list.append(file['id'][i])
        poi_list.append(" ".join([target_list[i] for i in translate_logits]))
        if i%100 == 0:
            print(i)

pd.DataFrame({'id':id_list,'predict_poi':poi_list}).to_csv('voice_to_poi.csv',',',encoding='utf-8',index=False)

