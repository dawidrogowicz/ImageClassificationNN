import tensorflow as tf
import numpy as np
import sys
import os


class NeuralNetwork:
    def __init__(self, layer_sizes, batch_size=256, epochs=15,
                 model_path='/tmp/model.ckpt'):
        self.h_layers = []
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')
        self.loss_history = []

        i = 1
        while i < len(layer_sizes) - 1:
            layer = {
                'weights': tf.Variable(tf.random_normal([layer_sizes[i - 1], layer_sizes[i]])),
                'biases': tf.Variable(tf.random_normal([layer_sizes[i]]))
            }
            self.h_layers.append(layer)
            i += 1

        self.out_layer = {
            'weights': tf.Variable(tf.random_normal([layer_sizes[-2], layer_sizes[-1]])),
            'biases': tf.Variable(tf.random_normal([layer_sizes[-1]])),
        }

    def model(self, data):
        output = data

        for layer in self.h_layers:
            output = tf.nn.relu(
                tf.add(
                    tf.matmul(output, layer['weights']),
                    layer['biases']
                )
            )

        return tf.add(
            tf.matmul(output, self.out_layer['weights']),
            self.out_layer['biases']
        )

    def train(self, data, labels):
        prediction = self.model(self.x)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(self.epochs):
                if epoch != 1 and os.path.isfile(self.model_path):
                    saver.restore(sess, self.model_path)

                epoch_loss = 0

                i = 0
                while i < len(data):
                    batch_x = np.array(data[i:i + self.batch_size])
                    batch_y = np.array(labels[i:i + self.batch_size])

                    _, batch_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                    epoch_loss += batch_cost

                    i += self.batch_size

                saver.save(sess, self.model_path)
                self.loss_history.append(epoch_loss)
                sys.stdout.write('\rFinished epoch {0} out of {1} with loss: {2}'.format(
                                 epoch + 1, self.epochs, int(epoch_loss)))

            print('\nTraining finished, model saved in: ', self.model_path)

    def test(self, data, labels):
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
            print('Accuracy: ', accuracy.eval({self.x: data, self.y: labels}))

    def predict(self, data):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)

            model = self.model(self.x)
            prediction = model.eval({self.x: data})

            output = np.zeros(self.layer_sizes[-1])
            output[tf.argmax(prediction, 1)] = 1
            return output
