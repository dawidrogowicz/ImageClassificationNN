import os
import sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import pickle


def convert_image(img_path, grey_scale=False):
    img = Image.open(img_path)
    width, height = img.size

    if width < height:
        img = img.rotate(90, expand=True, resample=Image.BILINEAR)

    if grey_scale:
        img = img.convert('L')

    img = img.resize((150, 100), Image.ANTIALIAS)
    return img


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

            img = convert_image(file_path, grey_scale=True)
            x.append(list(img.getdata()))
            y.append(category_dir)
        counter += 1

    return np.array(x), np.array(y)


original_data_path = '101_ObjectCategories'

if os.path.isfile('preprocessed_data.pickle'):
    with open('preprocessed_data.pickle', 'rb') as f:
        data_x, labels = pickle.load(f)
else:
    data_x, labels = prepare_files(original_data_path)
    with open('preprocessed_data.pickle', 'wb') as f:
        pickle.dump([data_x, labels], f)

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
layer_sizes = [no_pixels, 5000, 5000, 5000, no_categories]

clf = NeuralNetwork(layer_sizes, model_path='./model.ckpt', epochs=20)

print('Training model')
clf.train(train_X, train_y)

plt.plot(clf.loss_history)
plt.draw()

print('Testing model')
clf.test(test_X, test_y)

plt.show()
