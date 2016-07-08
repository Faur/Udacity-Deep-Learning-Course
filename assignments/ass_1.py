from tools import *


print('Aquiring dataset ...')
url = 'http://commondatastorage.googleapis.com/books1000/'

train_filename = maybe_download('notMNIST_large.tar.gz', url, 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', url, 8458043)









