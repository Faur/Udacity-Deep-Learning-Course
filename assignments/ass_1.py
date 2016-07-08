
# from __future__ import print_function
	# Bring the print function from python 3 into python 2.6+
import os
import sys
import tarfile
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
# from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

from tools import *

# Configure the matplotlib backend as plotting inline in IPython
# %matplotlib inline


print('Aquiring dataset ...')
url = 'http://commondatastorage.googleapis.com/books1000/'

train_filename = maybe_download('notMNIST_large.tar.gz', url, 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', url, 8458043)









