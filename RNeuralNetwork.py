import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
import sys
import os


class RecurrentNeuralNetwork:
    def __init__(self, layer_size, out_size, n_chunks, chunk_size, batch_size=256, epochs=10, learning_rate=0.001,
                 color_channels=3, model_path='/tmp/model.ckpt'):
        self.out_size = out_size
        self.batch_size = batch_size
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.learning_rate = learning_rate
        self.color_channels = color_channels
        self.epochs = epochs
        self.model_path = model_path
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.loss_history = []
        self.layer = {
            'weights': tf.Variable(tf.random_normal([layer_size, out_size])),
            'biases': tf.Variable(tf.random_normal([out_size]))
        }
        self.lstm_cell = rnn_cell.BasicLSTMCell(layer_size)

    def model(self, data):
        data = tf.transpose(data, (1, 0, 2, 3))
        data = tf.reshape(data, (-1, self.chunk_size * self.color_channels,))
        data = tf.split(data, self.n_chunks, 0)

        outputs, _ = rnn.static_rnn(self.lstm_cell, data, dtype=tf.float32)

        return tf.add(
            tf.matmul(outputs[-1], self.layer['weights']),
            self.layer['biases']
        )

    def fit(self, data, labels):
        prediction = self.model(self.x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                if epoch != 1 and os.path.isfile(self.model_path):
                    saver.restore(sess, self.model_path)

                epoch_loss = 0

                i = 0
                while i < len(data):
                    batch_x = np.array(data[i:i + self.batch_size]).reshape((-1, self.n_chunks, self.chunk_size, self.color_channels))
                    batch_y = np.array(labels[i:i + self.batch_size])

                    _, batch_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += batch_cost

                    i += self.batch_size

                saver.save(sess, self.model_path)
                self.loss_history.append(epoch_loss)
                sys.stdout.write('\rFinished epoch {0} out of {1} with loss: {2}'.format(
                                 epoch + 1, self.epochs, int(epoch_loss)))

            print('\nTraining finished, model saved in: ', self.model_path)

    def score(self, data, labels):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            prediction = self.model(self.x)

            correct = tf.equal(
                tf.argmax(prediction, 1),
                tf.argmax(self.y, 1)
            )
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval(
                {self.x: data.reshape((-1, self.n_chunks, self.chunk_size, self.color_channels)), self.y: labels}
            ))

    def predict(self, data):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            model = self.model(self.x)
            prediction = model.eval({self.x: data})

            output = np.zeros(self.out_size)
            output[tf.argmax(prediction, 1)] = 1
            return output
