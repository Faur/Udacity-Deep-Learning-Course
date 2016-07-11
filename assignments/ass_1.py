import os
import time
import random; random.seed(int(time.time()))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tools import *


print()
print('Aquiring dataset ...')

url = 'http://commondatastorage.googleapis.com/books1000/'
train_filename = maybe_download('notMNIST_large.tar.gz', url, 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', url, 8458043)


print()
print('Extracting dataset ...')

num_classes = 10
train_folders = maybe_extract(train_filename, num_classes)
test_folders = maybe_extract(test_filename, num_classes)


print()
print('Problem 1: Visulizing data ...')
png_img = img_plot(train_folders[0], 9, 9)
plt.draw()


print()
print('Pickeling dataset')
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


print('Done!')
plt.show()
