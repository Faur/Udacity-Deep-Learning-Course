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
all_A = os.listdir(train_folders[0])
	# Get all files in the 'A' folder

png_idx = random.sample(range(1, len(all_A)), 9*9)


def png_plot():
	plt.figure(1)
	for i in range(9*9):
		plt.subplot(9,9,i+1)
		img = mpimg.imread(os.path.join(train_folders[0], all_A[png_idx[i]]))
		plt.imshow(img, cmap='gray', interpolation='nearest') #, aspect='auto'
		plt.axis('off')

	plt.suptitle('PNG plot', fontsize=20)
	plt.tight_layout(w_pad=0, h_pad=0)
	plt.subplots_adjust(top=0.91)
	plt.draw()

png_plot()




print('Done!')
plt.show()
