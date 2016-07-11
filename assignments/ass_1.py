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



print()
print('Merge and prune dataset')
train_size = 200000
val_size = 10000
test_size = 10000
# train_size = 500000
# val_size = 18000
# test_size = 18000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
	train_datasets, train_size, val_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:  \t{}\t {}'.format(train_dataset.shape, train_labels.shape))
print('Validation:\t{}\t\t {}'.format(valid_dataset.shape, valid_labels.shape))
print('Test:      \t{}\t\t {}'.format(test_dataset.shape, test_labels.shape))


train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)


print()
print('Saving dataset ...')
pickle_file = 'notMNIST.pickle'

try:
	f = open(pickle_file, 'wb')
	save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		'test_dataset': test_dataset,
		'test_labels': test_labels,
		}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print('Unable to save data to {} : {}'.format(pickle_file, e))
	raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size: {}'.format(statinfo.st_size))




print()
print('Done!')
plt.show()
