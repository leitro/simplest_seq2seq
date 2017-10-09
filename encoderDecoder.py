import tensorflow as tf
import tensorflow.contrib.legacy_seq2seq as seq2seq
import numpy as np
import time
import os
from processData import vocab_size, index2word, input_max_len, output_max_len, END_TOKEN, processData

batch_size = 20
embed_dim = 10
num_units = 256
layer_size = 4
epochs = 100
learning_rate = 1e-3
global_num = 0

# <Model>
enc_in = tf.placeholder(tf.int32, shape=[batch_size, input_max_len])
labels = tf.placeholder(tf.int32, shape=[batch_size, output_max_len])
dec_in = tf.placeholder(tf.int32, shape=[batch_size, output_max_len])

enc_in2 = tf.unstack(enc_in, axis=1)
labels2 = tf.unstack(labels, axis=1)
dec_in2 = tf.unstack(dec_in, axis=1)


with tf.variable_scope('decoder'):
    cell = tf.contrib.rnn.GRUCell(num_units)
    decode_outputs, decode_states = seq2seq.embedding_rnn_seq2seq(enc_in2, dec_in2, cell, vocab_size, vocab_size, embed_dim, output_projection=None, feed_previous=False)

with tf.variable_scope('decoder', reuse=True):
    cell = tf.contrib.rnn.GRUCell(num_units)
    decode_outputs_t, decode_states_t = seq2seq.embedding_rnn_seq2seq(enc_in2, dec_in2, cell, vocab_size, vocab_size, embed_dim, output_projection=None, feed_previous=True)

loss_weights = [tf.ones(l.shape, dtype=tf.float32) for l in labels2]
loss = seq2seq.sequence_loss(decode_outputs, labels2, loss_weights, vocab_size)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# </Model>

def train(training_data, testing_data):
    in_data_train, la_data_train, out_data_train = training_data
    in_data_test, la_data_test, out_data_test = testing_data

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_num = len(in_data_train)
        num_per_epoch = total_num // batch_size
        for epoch in range(epochs):
            global global_num
            global_num = 0
            total_loss = 0
            start = time.time()
            for i in range(num_per_epoch):
                input_data =  np.array(in_data_train[i*batch_size: (i+1)*batch_size])
                output_data = np.array(out_data_train[i*batch_size: (i+1)*batch_size])
                labels_data = np.array(la_data_train[i*batch_size: (i+1)*batch_size])
                loss_value, _ = sess.run([loss, train_op], feed_dict={enc_in: input_data, dec_in: output_data, labels: labels_data})
                total_loss += loss_value
            print('epoch %d, loss=%.2f, time=%.2f' %(epoch, total_loss, time.time()-start))
            total_num_t = len(in_data_test)
            num_per_epoch_t = total_num_t // batch_size
            total_loss_t = 0
            start_t = time.time()
            for i in range(num_per_epoch_t):
                input_data_t =  np.array(in_data_test[i*batch_size: (i+1)*batch_size])
                output_data_t = np.array(out_data_test[i*batch_size: (i+1)*batch_size])
                labels_data_t = np.array(la_data_test[i*batch_size: (i+1)*batch_size])
                loss_value_t, pred_t = sess.run([loss, decode_outputs_t], feed_dict={enc_in: input_data_t, dec_in: output_data_t, labels: labels_data_t})
                writePredict(epoch, pred_t)
                total_loss_t += loss_value_t
            print('##TEST## loss=%.2f, time=%.2f' %(total_loss_t, time.time()-start_t))


def writePredict(epoch, pred): # [(batch_size, vocab_size)] * max_output_len
    global global_num
    pred = [np.argmax(item, axis=1) for item in pred]
    pred = np.vstack(pred) # (max_output_len, batch_size)
    pred = np.transpose(pred, (1, 0)) # (batch_size, max_output_len)
    if not os.path.exists('pred_logs'):
        os.makedirs('pred_logs')

    with open('pred_logs/test_predict_seq.'+str(epoch)+'.log', 'a') as f:
        for seq in pred:
            f.write(str(global_num)+' ')
            for i in seq:
                if i == END_TOKEN:
                    f.write('\n')
                    break
                else:
                    word = index2word[i]
                    f.write(word+' ')
            global_num += 1


if __name__ == '__main__':
    train_data, test_data = processData('input.dat', 'output.dat')
    train(train_data, test_data)
