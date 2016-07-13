
print(' ... Importin Libaries')


import numpy as np
import timeit

from six.moves import cPickle as pickle

from FFNN_tools import *
from tools import * 


print(' ... Seting up parameters')
pickle_file = 'notMNIST.pickle'

learning_rate=0.01
L1_reg 		= 0.0001
L2_reg 		= 0.0001
n_epochs 	= 1000
batch_size 	= 100
n_hidden 	= 500

image_size = 28
num_labels = 10

n_epochs = 10000
batch_size = 100
learning_rate = 0.13

patience = 5000 # Look at this many examples regardless
patience_increase = 2 
improvement_threshold = 0.995	# A relative improvement of this much is considered

rng = np.random.RandomState(int(timeit.default_timer()))

print(' ... Importing data')
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	valid_dataset = save['valid_dataset']
	valid_labels = save['valid_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save # Free up memory

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

train_dataset, train_labels = shared_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = shared_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = shared_dataset(test_dataset, test_labels)

n_train_batches = train_dataset.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_dataset.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_dataset.get_value(borrow=True).shape[0] // batch_size

validation_frequency = min(n_train_batches, patience//2)
							# Go through this many minibatches before checking the network


index 	= T.lscalar('index')
x 		= T.matrix('x')
y 		= T.ivector('y')


print(' ... building model')
classifier = MLP(
	rng = rng,
	input = x,
	n_in = image_size * image_size,
	n_hidden = n_hidden,
	n_out = num_labels
)


print(' ... building functions')

cost = (
	classifier.negative_log_likelihood(y)
	+ L1_reg * classifier.L1
	+ L2_reg * classifier.L2
)

test_model = theano.function(
	inputs  = [index],
	outputs = classifier.errors(y),
	givens = {
		x: test_dataset[index * batch_size : (index + 1) * batch_size],
		y: test_labels[index * batch_size : (index + 1) * batch_size]
	})


valid_model = theano.function(
	inputs  = [index],
	outputs = classifier.errors(y),
	givens = {
		x: valid_dataset[index * batch_size : (index + 1) * batch_size],
		y: valid_labels[index * batch_size : (index + 1) * batch_size]
	})

# Compute gradient of cot wrt the parameters 
gparams = [T.grad(cost, param) for param in classifier.params]

# given two lists of the same length, A = [a1, a2, a3, a4] and
# B = [b1, b2, b3, b4], zip generates a list C of same size, where each
# element is a pair formed from the two lists :
#    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
updates = [
	(param, param - learning_rate * gparam)
	for param, gparam in zip(classifier.params, gparams)
]

train_model = theano.function(
	inputs=[index],
	outputs=cost,
	updates=updates,
	givens={
		x: train_dataset[index * batch_size : (index + 1) * batch_size],
		y: train_labels[index * batch_size : (index + 1) * batch_size],
	}
)


print(' ... training model')

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
	epoch += 1

	for minibatch_index in range(n_train_batches):

		minibatch_avg_cost = train_model(minibatch_index)

		iter = (epoch-1) * n_train_batches + minibatch_index

		if (iter + 1) % validation_frequency == 0:
			validation_losses = [valid_model(i) for i in range(n_valid_batches)]
			this_validation_loss = np.mean(validation_losses)

			print('Epoch {}, minibatch {}/{}, validation error {:2} %%'.format(
				epoch, minibatch_index + 1, n_train_batches, this_validation_loss*100)
			)

			if this_validation_loss < best_validation_loss:
				if(this_validation_loss < best_validation_loss * improvement_threshold):
					patience = max(patience, iter * patience_increase)

				best_validation_loss = this_validation_loss
				best_iter = iter

				test_losses = [test_model(i) for i in range(n_test_batches)]
				test_score = np.mean(test_losses)

				print('    test error of best model {:3} %%'.format(
					test_score*100)
				)


print('Done!')



