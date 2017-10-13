import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.learn import ModeKeys
from processData import processData, output_max_len, vocab_size, index2word
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
embed_dim = 50
num_units = 256
batch_size = 20

trainData, testData = processData('input.dat', 'output.dat')

in_data, la_data, out_data = trainData
in_data_t, la_data_t, out_data_t = testData

# `(features, labels, mode, params) -> (predictions, loss, train_op)`
def seq2seq(features, labels, mode, params):
    in_data = features['input']
    out_data = features['output']
    # (batch_size, seq_length, embed_dim)
    in_embed = layers.embed_sequence(in_data, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')
    out_embed = layers.embed_sequence(out_data, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)

    cell1 = tf.contrib.rnn.GRUCell(num_units=num_units)
    # encoder_out (batch_size, input_max_len, num_units)
    # encoder_final_state (batch_size, num_units)
    encoder_out, encoder_final_state = tf.nn.dynamic_rnn(cell1, in_embed, dtype=tf.float32)

    output_lengths = tf.convert_to_tensor([output_max_len]*batch_size)
    train_helper = tf.contrib.seq2seq.TrainingHelper(out_embed, output_lengths)

    with tf.variable_scope('embed', reuse=True):
        # embeddings (29, 50)
        embeddings = tf.get_variable('embeddings')
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embeddings,
                    start_tokens=np.array([GO_TOKEN]*batch_size),
                    end_token=END_TOKEN)


    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # attention (batch_size, max_time)
            attention = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=num_units,
                        memory=encoder_out,
                        memory_sequence_length=None)
            cell2 = tf.contrib.rnn.GRUCell(num_units=num_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                        cell2,
                        attention,
                        attention_layer_size=num_units/2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                        attn_cell,
                        vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))
            # outputs (final_outputs, final_state, final_sequence_lengths)
            outputs = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder,
                        output_time_major=False,
                        impute_finished=True,
                        maximum_iterations=output_max_len)
            return outputs[0]

    # train_out -> BasicDecoderOutput(rnn_output=(20, ?, 29), sample_id=(20, ?))
    train_out = decode(train_helper, 'decode')
    pred_out = decode(pred_helper, 'decode', reuse=True)

    tf.identity(train_out.sample_id[0], name='train_pred_jaja')
    tf.identity(train_out.sample_id[0], name='test_pred_jaja')

    # weights (batch_size, sequence_length)
    weights = tf.ones([batch_size, output_max_len])
    loss = tf.contrib.seq2seq.sequence_loss(
            train_out.rnn_output,
            out_data,
            weights=weights)
    tf.identity(loss, name='loss_jaja')

    train_op = layers.optimize_loss(
                loss,
                tf.train.get_global_step(),
                optimizer='Adam',
                learning_rate=1e-3,
                summaries=['loss', 'learning_rate'])

    est_spec = tf.estimator.EstimatorSpec(
                mode=ModeKeys.TRAIN, # .EVAL .PREDICT
                predictions=pred_out.sample_id,
                loss=loss,
                train_op=train_op)

    return est_spec

def input_fn():
    input_data = tf.placeholder(tf.int32, shape=[batch_size, None], name='input')
    output_data = tf.placeholder(tf.int32, shape=[batch_size, None], name='output')
    tf.identity(input_data[0], 'input_jaja_0')
    tf.identity(output_data[0], 'output_jaja_0')
    return {'input': input_data, 'output': output_data}, None

def sampler():
    num = len(in_data) // batch_size
    while True:
        for i in range(num):
            yield {'input': in_data[i*batch_size: (i+1)*batch_size], 'output': out_data[i*batch_size: (i+1)*batch_size]}

sample = sampler()

def feed_fn():
    res = sample.__next__()
    return {'input:0': res['input'], 'output:0': res['output']}

def get_formatter(keys):
    def to_str(seq):
        res = [index2word[i] for i in seq]
        return ' '.join(res)

    def format(values):
        ress = []
        for k in keys:
            ress.append('%s = %s' % (k, to_str(values[k])))
        return '\n'.join(ress)
    return format

def print_hooks():
    print_inputs = tf.train.LoggingTensorHook(['input_jaja_0', 'output_jaja_0'], every_n_iter=100,
                    formatter=get_formatter(['input_jaja_0', 'output_jaja_0']))
    print_predictions = tf.train.LoggingTensorHook(['train_pred_jaja', 'test_pred_jaja'], every_n_iter=100,
                    formatter=get_formatter(['train_pred_jaja', 'test_pred_jaja']))
    print_loss = tf.train.LoggingTensorHook(['loss_jaja'], every_n_iter=10)
    return (print_inputs, print_predictions, print_loss)

if __name__ == '__main__':
    est = tf.estimator.Estimator(
            model_fn=seq2seq,
            model_dir='./model_dir',
            params=None)
    est.train(
            input_fn=input_fn,
            hooks=[tf.train.FeedFnHook(feed_fn), *print_hooks()],
            steps=2000)


