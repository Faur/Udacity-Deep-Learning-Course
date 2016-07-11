import os
import sys
import tarfile

from six.moves.urllib.request import urlretrieve

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=0)

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
		print('{:s} succesfully verified'.format(filename))
	else:
		os.remove(filename)
		raise Exception('Failed to verify {:s} from {:s}\n\t\t'
			'Actual size: {:d}. Expected size: {:d}'.format(
			filename, url, statinfo.st_size, expected_bytes))
	return filename

def maybe_extract(filename, num_classes, force=False):
	"""Assumes that all the data for each class is in a separate folder"""
	root = os.path.splitext(os.path.splitext(filename)[0])[0] #remove.tar.gz

	if os.path.isdir(root) and not force:
		print('{} already present - skipping extraction of {}.'.format(
			root, filename))
	else:
		print('Extracting data from {}. This may take a while.'.format(
			root))
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
		print('Extraction complete.')

	data_folders = [
		os.path.join(root, d) for d in sorted(os.listdir(root))
		if os.path.isdir(os.path.join(root, d))]

	if len(data_folders) != num_classes:
		raise Exception(
			'Expected {} folders, one per class. Found {} instead.'.format(
				num_classes, len(data_folders)))

	return data_folders


def img_plot(folder, row=12, col=12, title='Image Plot'):
	"""Assumes there are only picutres that can be displayed with plt.imshow in folder"""
	all_A = os.listdir(folder)
		# Get all files in the 'A' folder

	png_idx = random.sample(range(0, len(all_A)), row*col)
	all_img = plt.figure(1)
	for i in range(row*col):
		plt.subplot(row,col,i+1)
		img = mpimg.imread(os.path.join(folder, all_A[png_idx[i]]))
		plt.imshow(img, cmap='gray_r', interpolation='nearest')#, aspect='auto')
		plt.axis('off')

	plt.suptitle(title, fontsize=20)
	# plt.tight_layout(pad=0, w_pad=0, h_pad=0)
	plt.subplots_adjust(top=0.91, hspace=0.1, wspace=0.1)
	return all_img

