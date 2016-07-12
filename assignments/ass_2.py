
from six.moves import cPickle as pickle
from tools import *

pickle_file = 'notMNIST.pickle'

print(' * Loading data ...')
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save # Free up memory
print('Training set\t', train_dataset.shape, train_labels.shape)
print('Validation set\t', valid_dataset.shape, valid_labels.shape)
print('Test set\t', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print()
print('Training set\t', train_dataset.shape, train_labels.shape)
print('Validation set\t', valid_dataset.shape, valid_labels.shape)
print('Test set\t', test_dataset.shape, test_labels.shape)

