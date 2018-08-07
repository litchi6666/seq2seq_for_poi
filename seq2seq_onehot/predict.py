import tensorflow as tf
from pre import data_utils, load_data
import pandas as pd
# Number of Epochs
epochs = 30
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
lr = 0.003
# 每50轮打一次结果
display_step = 50

source_list, source_vocab_to_int = data_utils.word_vocab(column=0)
target_list, target_vocab_to_int = data_utils.word_vocab(column=1)

# source_text_to_int, target_text_to_int = load_data.get_sts2idx()


def sentence_to_seq(sentence, source_vocab_to_int):
    """
    将句子转化为数字编码
    """
    unk_idx = source_vocab_to_int["<UNK>"]
    word_idx = [source_vocab_to_int.get(word, unk_idx) for word in sentence.split(',')]

    return word_idx


tf.get_variable
loaded_graph = tf.Graph()

id_list, poi_list = [], []
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints_poi_clean/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints_poi_clean'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')

    ff = pd.read_csv('../file/videos_train.csv',encoding='utf-8-sig')

    for i in range(1000):
        translate_sentence_text = ff['title'][i]
        translate_sentence = sentence_to_seq(translate_sentence_text, source_vocab_to_int)
        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size})[0]

        # print('【Input】')
        # print('  Word Ids:      {}'.format([i for i in translate_sentence]))
        # print('  title Words: {}'.format([source_list[i] for i in translate_sentence]))
        #
        # print('\n【Prediction】')
        # print('  Word Ids:      {}'.format([i for i in translate_logits]))
        # print('  poi Words: {}'.format([target_list[i] for i in translate_logits]))
        #
        # print("\n【Full Sentence】")
        # print(" ".join([target_list[i] for i in translate_logits]))
        id_list.append(ff['id'][i])
        poi_list.append(" ".join([target_list[i] for i in translate_logits]))
        if i%100 == 0:

            print(i)

pd.DataFrame({'id':id_list,'poi':poi_list}).to_csv('title_to_poi__clean.csv',index=False,encoding='utf-8')