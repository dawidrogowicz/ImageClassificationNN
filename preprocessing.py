import os
import sys
from PIL import Image


def convert_image(img_path, img_out_path, grey_scale=False):
    img = Image.open(img_path)
    width, height = img.size

    if width < height:
        img = img.rotate(90, expand=True, resample=Image.BILINEAR)

    if grey_scale:
        img = img.convert('L')

    img = img.resize((300, 200), Image.ANTIALIAS)
    img.save(img_out_path)


def prepare_files(dir_in, dir_out):
    counter = 1
    total_dirs = len(os.listdir(dir_in))

    os.mkdir(dir_out)

    for category_dir in os.listdir(dir_in):
        if category_dir == 'BACKGROUND_Google':
            print('\rSkipping directory: ', category_dir)
            counter += 1
            continue

        sys.stdout.write('\rPreparing category {0} out of {1} |{2}{3}|'
                         .format(counter, total_dirs, '=' * counter, ' ' * (total_dirs - counter)))

        os.mkdir('{0}/{1}'.format(dir_out, category_dir))

        for file_name in os.listdir('{0}/{1}'.format(dir_in, category_dir)):
            file_path = '{0}/{1}/{2}'.format(dir_in, category_dir, file_name)
            file_out_path = '{0}/{1}/{2}'.format(dir_out, category_dir, file_name)

            if not file_name.endswith('.jpg'):
                print('\rSkipping file: ', file_path, ' (unsupported file extension)')
                continue

            convert_image(file_path, file_out_path, grey_scale=True)

        counter += 1


original_data_path = '101_ObjectCategories'
copy_data_path = 'data_categories'

if not os.path.isdir(copy_data_path):
    prepare_files(original_data_path, copy_data_path)

