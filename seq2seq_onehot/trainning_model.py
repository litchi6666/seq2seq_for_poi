from seq2seq_onehot.seq2seq import *
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from pre import data_utils, load_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

_, source_vocab_to_int = data_utils.word_vocab(column=0)
_, target_vocab_to_int = data_utils.word_vocab(column=1)

source_text_to_int, target_text_to_int = load_data.get_sts2idx()


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

train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()

    max_target_sequence_len = 25
    train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                   targets,
                                                   batch_size,
                                                   source_sequence_len,
                                                   target_sequence_len,
                                                   max_target_sequence_len,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoder_embedding_size,
                                                   decoder_embedding_size,
                                                   rnn_size,
                                                   rnn_num_layers,
                                                   target_vocab_to_int)

    training_logits = tf.identity(train_logits.rnn_output, name="logits")
    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)


def get_batches(sources, targets, batch_size):
    """
    获取batch
    """
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Need the lengths for the _lengths parameters
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield sources_batch, targets_batch, source_lengths, targets_lengths


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(source_text_to_int, target_text_to_int, batch_size)):

            _, loss = sess.run(
                [train_op, cost],
                {inputs: source_batch,
                 targets: target_batch,
                 learning_rate: lr,
                 source_sequence_len: sources_lengths,
                 target_sequence_len: targets_lengths})

            if batch_i % display_step == 0 and batch_i > 0:
                batch_train_logits = sess.run(
                    inference_logits,
                    {inputs: source_batch,
                     source_sequence_len: sources_lengths,
                     target_sequence_len: targets_lengths})

                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_text_to_int) // batch_size, loss))

        if epoch_i > 0 and epoch_i % 2 == 0:
            saver = tf.train.Saver()
            saver.save(sess, "checkpoints_poi_clean_%s/dev" % epoch_i,)
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, "checkpoints_poi_clean/dev")
    print('Model Trained and Saved')