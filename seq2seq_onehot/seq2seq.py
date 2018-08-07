import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def model_inputs():
    """
    构造输入
    返回：inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len，类型为tensor
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len


def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):
    """
    构造Encoder端

    @param rnn_inputs: rnn的输入
    @param rnn_size: rnn的隐层结点数
    @param rnn_num_layers: rnn的堆叠层数
    @param source_sequence_len: 英文句子序列的长度
    @param source_vocab_size: 英文词典的大小
    @param encoder_embedding_size: Encoder层中对单词进行词向量嵌入后的维度
    """
    # 对输入的单词进行词向量嵌入
    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    # LSTM单元
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    # 堆叠rnn_num_layers层LSTM
    lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
                                                        dtype=tf.float32)

    return encoder_outputs, encoder_states


def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):
    """
    对Decoder端的输入进行处理

    @param target_data: 法语数据的tensor
    @param target_vocab_to_int: 法语数据的词典到索引的映射
    @param batch_size: batch size
    """
    # 去掉batch中每个序列句子的最后一个单词
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # 在batch中每个序列句子的前面添加”<GO>"
    decoder_inputs = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int["<GO>"]),
                                ending], 1)

    return decoder_inputs


def decoder_layer_train(encoder_states, decoder_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer):
    """
    Decoder端的训练

    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    """

    # 生成helper对象
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                        sequence_length=target_sequence_len,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                       training_helper,
                                                       encoder_states,
                                                       output_layer)

    training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_len)

    return training_decoder_outputs


def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size):
    """
    Decoder端的预测/推断

    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_cell: Decoder端
    @param decoder_embed: Decoder端词向量嵌入后的输入
    @param start_id: 句子起始单词的token id， 即"<GO>"的编码
    @param end_id: 句子结束的token id，即"<EOS>"的编码
    @param max_target_sequence_len: 法语文本的最大长度
    @param output_layer: 输出层
    @batch_size: batch size
    """

    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
                                                                start_tokens,
                                                                end_id)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                        inference_helper,
                                                        encoder_states,
                                                        output_layer)

    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                        impute_finished=True,
                                                                        maximum_iterations=max_target_sequence_len)

    return inference_decoder_outputs


def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
                  max_target_sequence_len, rnn_size, rnn_num_layers,
                  target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size):
    """
    构造Decoder端

    @param encoder_states: Encoder端编码得到的Context Vector
    @param decoder_inputs: Decoder端的输入
    @param target_sequence_len: 法语文本的长度
    @param max_target_sequence_len: 法语文本的最大长度
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    @param target_vocab_size: 法语词典的大小
    @param decoder_embedding_size: Decoder端词向量嵌入的大小
    @param batch_size: batch size
    """

    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm

    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])

    # output_layer logits
    output_layer = tf.layers.Dense(target_vocab_size)

    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(encoder_states,
                                              decoder_cell,
                                              decoder_embed,
                                              target_sequence_len,
                                              max_target_sequence_len,
                                              output_layer)

    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(encoder_states,
                                               decoder_cell,
                                               decoder_embeddings,
                                               target_vocab_to_int["<GO>"],
                                               target_vocab_to_int["<EOS>"],
                                               max_target_sequence_len,
                                               output_layer,
                                               batch_size)

    return training_logits, inference_logits


def seq2seq_model(input_data, target_data, batch_size,
                  source_sequence_len, target_sequence_len, max_target_sentence_len,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embeding_size,
                  rnn_size, rnn_num_layers, target_vocab_to_int):
    """
    构造Seq2Seq模型

    @param input_data: tensor of input data
    @param target_data: tensor of target data
    @param batch_size: batch size
    @param source_sequence_len: 英文语料的长度
    @param target_sequence_len: 法语语料的长度
    @param max_target_sentence_len: 法语的最大句子长度
    @param source_vocab_size: 英文词典的大小
    @param target_vocab_size: 法语词典的大小
    @param encoder_embedding_size: Encoder端词嵌入向量大小
    @param decoder_embedding_size: Decoder端词嵌入向量大小
    @param rnn_size: rnn隐层结点数
    @param rnn_num_layers: rnn堆叠层数
    @param target_vocab_to_int: 法语单词到token id的映射
    """
    _, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)

    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states,
                                                                        decoder_inputs,
                                                                        target_sequence_len,
                                                                        max_target_sentence_len,
                                                                        rnn_size,
                                                                        rnn_num_layers,
                                                                        target_vocab_to_int,
                                                                        target_vocab_size,
                                                                        decoder_embeding_size,
                                                                        batch_size)
    return training_decoder_outputs, inference_decoder_outputs