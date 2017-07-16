import numpy as np
import cv2
# import scipy.io as sio
import pickle
import os
from tqdm import tqdm

image_path = "Data/train/"     # images

# target_path = "images/miml data.mat"   # target labels
# image_path = "images"     # images
#
# y = sio.loadmat(target_path)
# y = y['targets']
# y = y.transpose()
# y = np.array([[elem if elem == 1 else 0 for elem in row]for row in y])
#
# x = []
#
# print('Processing images...')
#
# for i in range(1, 2001):
#     img = image_path + "/" + str(i) + ".jpg"
#     img = cv2.imread(img)
#     img = cv2.resize(img, (100, 100))
#     img = img.transpose((2, 0, 1))
#     x.append(img)
#
# data_dict = dict()
# data_dict['x'] = np.array(x)
# data_dict['y'] = y
#
# file = open('images.pkl', 'wb')
# pickle.dump(data_dict, file)
# file.close()
#
# print('Saved data...')

images_list = os.listdir(image_path)

x, y = [], []

print('Processing images...')

for image in tqdm(images_list):
    img_path = 'Data/train/' + image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img = img.transpose((2, 0, 1))
    x.append(img)

    if 'cat' in image:
        y.append([1, 0])
    else:
        y.append([0, 1])

x = np.array(x)
y = np.array(y)

data_dict = dict()
data_dict['x'] = np.array(x)
data_dict['y'] = y

file = open('images.pkl', 'wb')
pickle.dump(data_dict, file)
file.close()

print('Saved data...')
