import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from RNeuralNetwork import RecurrentNeuralNetwork
import matplotlib.pyplot as plt


def convert_image(img_path, grey_scale=False):
    img = Image.open(img_path)
    width, height = img.size

    if width < height:
        img = img.rotate(90, expand=True, resample=Image.BILINEAR)

    img = img.resize((45, 30), Image.ANTIALIAS)

    if grey_scale:
        img = img.convert('L')
        pixel_data = img.getdata()
    else:
        img = img.convert('RGB')
        pixel_data = np.array(img.getdata()).flatten()

    return pixel_data


def prepare_files(dir_in):
    counter = 1
    total_dirs = len(os.listdir(dir_in)) - 1

    x = []
    y = []

    for category_dir in os.listdir(dir_in):
        if category_dir == 'BACKGROUND_Google':
            print('\rSkipping directory: ', category_dir)
            continue

        sys.stdout.write('\rPreparing category {0} out of {1} |{2}{3}|'
                         .format(counter, total_dirs, '=' * counter, ' ' * (total_dirs - counter)))

        for file_name in os.listdir('{0}/{1}'.format(dir_in, category_dir)):
            file_path = '{0}/{1}/{2}'.format(dir_in, category_dir, file_name)

            if not file_name.endswith('.jpg'):
                print('\rSkipping file: ', file_path, ' (unsupported file extension)')
                continue

            pixel_data = convert_image(file_path, grey_scale=True)
            x.append(list(pixel_data))
            y.append(category_dir)
        counter += 1

    return np.array(x), np.array(y)


original_data_path = '101_ObjectCategories'

data_x, labels = prepare_files(original_data_path)
print('\nData obtained')

label_indexes = {}
no_categories = len(np.unique(labels))

for i, label in enumerate(np.unique(labels)):
    output_array = np.zeros(no_categories)
    output_array[i] = 1
    label_indexes[label] = tuple(output_array)

data_y = [label_indexes[label] for label in labels]
print('Created labels from dictionary')

train_X, test_X, train_y, test_y = train_test_split(data_x, data_y, test_size=.1)

no_pixels = np.shape(data_x)[1]

layer_sizes = [no_pixels, 500, 500, 500, no_categories]

nn = NeuralNetwork(layer_sizes, model_path='./model.ckpt')
rnn = RecurrentNeuralNetwork(128, no_categories, 45, 30,  model_path='./model_rnn.ckpt', epochs=10)

clf = rnn

print('Training model')
clf.train(train_X, train_y)

plt.plot(clf.loss_history)
plt.draw()

print('Testing model')
clf.test(test_X, test_y)

plt.show()
