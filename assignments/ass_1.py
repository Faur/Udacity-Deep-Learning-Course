
# from __future__ import print_function
	# Bring the print function from python 3 into python 2.6+
import os
import sys
import tarfile
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
# from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Configure the matplotlib backend as plotting inline in IPython
# %matplotlib inline


print('Aquiring dataset ...')
last_percentage_reported = None

def download_progress_hook(count, blockSize, totalSize):
	"""A hook to report the progress of a download.
	Reports every 1% change ind download progress"""
	global last_percentage_reported
	percent = int(count * blockSize * 100/totalSize)
	if last_percentage_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write('%s%% ' % percent)
			sys.stdout.flush()
		last_percentage_reported = percent

def maybe_download(filename, url, expected_bytes, force=False):
	"""Check if file is present, if not download it. Also check file size"""
	if force or not os.path.exists(filename):
		print('Downloading dataset:', filename, '...')
		urlretrieve(url + filename, filename, reporthook=download_progress_hook)
		print('\nDownload complete')
	statinfo = os.stat(filename)

	if statinfo.st_size == expected_bytes:
		print('Download succesfully verified')
	else:
		os.remove(filename)
		raise Exception('Failed to verify {:s} from {:s}\n\t\t'
			'Actual size: {:d}. Expected size: {:d}'.format(
			filename, url, statinfo.st_size, expected_bytes))
	return filename


url = 'http://commondatastorage.googleapis.com/books1000/'

# train_filename = maybe_download('notMNIST_large.tar.gz', url, 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', url, 8458043)









