import numpy as np
from NeuralNetwork import NeuralNetwork
from RNeuralNetwork import RecurrentNeuralNetwork
import matplotlib.pyplot as plt
import pickle


def number_to_one_hot(number):
    output = np.zeros(10)
    output[number] = 1
    return output


def prepare_files(file_path_format):
    data_x = []
    data_y = []

    for i in range(1, 6):
        file_path = file_path_format.format(i)
        with open(file_path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')

        x = dict[b'data']
        y = np.array(list(map(number_to_one_hot, dict[b'labels'])))

        if i < 2:
            data_x = x
            data_y = y
        else:
            data_x = np.concatenate((data_x, x))
            data_y = np.concatenate((data_y, y))

    data_x = np.reshape(data_x, (-1, 3, 1024))
    data_x = np.transpose(data_x, (0, 2, 1))
    data_x = np.reshape(data_x, (-1, 3072))

    return data_x, data_y


train_x, train_y = prepare_files('cifar-10-batches-py/data_batch_{}')
test_x, test_y = prepare_files('cifar-10-batches-py/test_batch')
print('\nData obtained')

data_shape = np.shape(test_x)

layer_sizes = [data_shape[1], 5000, 3000, 1000, 10]

nn = NeuralNetwork(layer_sizes, model_path='./model.ckpt')
rnn = RecurrentNeuralNetwork(1024, 10, 32, 32,  model_path='./model_rnn.ckpt')

clf = nn

print('Training model')
clf.fit(train_x, train_y)

plt.plot(clf.loss_history)
plt.draw()

print('Testing model')
clf.score(test_x, test_y)

plt.show()
