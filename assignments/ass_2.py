import timeit
import os

from six.moves import cPickle as pickle

from tools import *
from log_reg import * 



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



print()
print('Here the code diverges from the Udacity DL course.')
print('    Inspiration from: http://deeplearning.net/tutorial/logreg.html')

import numpy as np
import theano
import theano.tensor as T
print('... Building the model')

##### PARAMETERS AND SETTINGS #####

# output_path = os.path.join('..', 'output')
model_save = os.path.join('..', 'output', 'best_model.pkl')

# Data parameters
image_size = 28
num_labels = 10

# Training parameters
n_epochs = 10000
batch_size = 100
learning_rate = 0.13

train_dataset, train_labels = shared_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = shared_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = shared_dataset(test_dataset, test_labels)
n_train_batches = train_dataset.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_dataset.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_dataset.get_value(borrow=True).shape[0] // batch_size

# Early stopping parameters
patience = 5000 # Look at this many examples regardless
patience_increase = 2 
improvement_threshold = 1.005	# A relative improvement of this much is considered
validation_frequency = min(n_train_batches, patience//2)
							# Go through this many minibatches before checking the network


##### AUTO RUN #####
index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

classifier = LogisticRegression(input=x, n_in=image_size*image_size, n_out=num_labels)

# Symbolic expression for the cost we are trying to minimize
cost = classifier.negative_log_likelihood(y)

# theano function computing the mistakes the model makes
test_model = theano.function(
	inputs=[index],
	outputs=classifier.errors(y),
	givens={
		x: test_dataset[index*batch_size: (index+1)*batch_size, :],
		y: test_labels[index*batch_size: (index+1)*batch_size],
	}
)

validate_model = theano.function(
	inputs=[index], 
	outputs=classifier.errors(y),
	givens={
		x: valid_dataset[index*batch_size: (index+1)*batch_size],
		y: valid_labels[index*batch_size: (index+1)*batch_size],		
	}
)

# Compute the gradient of cost with respect to W and b
g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)


# Specifiy how the paramters of the model should be updated
updates = [	(classifier.W, classifier.W - learning_rate*g_W),
			(classifier.b, classifier.b - learning_rate*g_b)]


# Compile a theano function for training the model
train_model = theano.function(
	inputs=[index],
	outputs=cost,
	updates=updates,
	givens={
		x: train_dataset[index*batch_size: (index+1)*batch_size],
		y: train_labels[index*batch_size: (index+1)*batch_size]
	}
)


###############
# TRAIN MODEL #
###############
print('... training the model')
best_validation_loss = np.inf
test_score = 0
start_time = timeit.default_timer()


done_looping = False
epoch = 0

while (epoch < n_epochs) and (not done_looping):
	epoch += 1

	for minibatch_index in range(n_train_batches):

		minibatch_avg_cost = train_model(minibatch_index)
		# Iteration number
		iter = (epoch-1) * n_train_batches + minibatch_index

		if (iter+1) % validation_frequency == 0:
			# Compute zero-one loss on validation set
			validation_losses = [validate_model(i) for i in range(n_valid_batches)]
			this_validation_loss = np.mean(validation_losses)

			print('epoch {}, minibatch {}/{}, validation error {} %%'.format(
					epoch, minibatch_index+1, n_train_batches, this_validation_loss*100)
			)

			# Check if new best score is found
			if this_validation_loss < best_validation_loss:
				# if improvement is good enough improve patiences
				if this_validation_loss < best_validation_loss * improvement_threshold:
					patience = max(patience, iter * patience_increase)

				best_validation_loss = this_validation_loss

				# test on the test set
				test_losses = [test_model(i) for i in range(n_test_batches)]
				test_score = np.mean(test_losses)

				print('    epoch {}, minibatch {}/{}, test error of '
						'best model {} %%'.format(
						epoch, minibatch_index+1, n_test_batches, test_score*100)
				)

				with open(model_save, 'wb') as f:
					pickle.dump(classifier, f)
		# if patience <= iter:
		# 	done_looping = True
		# 	break

end_time = timeit.default_timer()
print()
print(
	(
		'Optimization complete with best validation score of %f %%, '
		'with test performance %f %%'
	)
		% (best_validation_loss * 100., test_score * 100.)
	)
print('The code run for %d epochs, with %f epochs/sec' % (
	epoch, 1. * epoch / (end_time - start_time)))
print(('The code for file ' +
	os.path.split(__file__)[1] +
	' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)



# print()
# print('Problem 1: Turn the logistic regression example with SGD into a 1-hidden layer '
# 	'neural network with rectified linear units and 1024 hidden nodes. This model should '
# 	'improve your validation / test accuracy.')