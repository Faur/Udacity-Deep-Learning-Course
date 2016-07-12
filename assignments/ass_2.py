print('  Inspiration from: http://deeplearning.net/tutorial/logreg.html')

pickle_file = 'notMNIST.pickle'

print(' * Loading data ...')
with open(pickle_file, 'rb' as f):
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save # Free up memory
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Validation set', valid_dataset.shape, valid_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)



