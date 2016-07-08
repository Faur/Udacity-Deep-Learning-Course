import os
import sys
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
