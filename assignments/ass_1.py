import os
import time
import random; random.seed(int(time.time()))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tools import *

FAST = True

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
if FAST:
	print('Skipping visualization')
else:
	png_img = img_plot(train_folders[0], 9, 9)
	plt.draw()


print()
print('Pickeling dataset')
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)


print()
print('Problem 2: Visualize the dataset again')
if FAST:
	print('Skipping visualization')
else:
	png_img = data_plot(train_datasets[0], 9, 9)
	plt.draw()


print()
print('Problem 3: Verify that the data is balaced accross classes')
if FAST:
	print('Skipping count')
else:
	print('Training data:')
	for pickle_file in train_datasets:
		with open(pickle_file, 'rb') as f:
			data = pickle.load(f)
		print('  {}'.format(len(data)))

	print('Test data:')
	for pickle_file in test_datasets:
		with open(pickle_file, 'rb') as f:
			data = pickle.load(f)
		print('  {}'.format(len(data)))




print('Done!')
plt.show()
