import os
import sys
import tarfile

from six.moves.urllib.request import urlretrieve


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

