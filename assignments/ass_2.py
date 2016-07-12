
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

image_size = 28
num_labels = 10

batch_size = 100

learning_rate = 0.13


##### AUTO RUN #####

train_dataset, train_labels = shared_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = shared_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = shared_dataset(test_dataset, test_labels)

n_train_batches = train_dataset.get_value(borrow=True).shape[0] // batch_size
n_valid_batches = valid_dataset.get_value(borrow=True).shape[0] // batch_size
n_test_batches = test_dataset.get_value(borrow=True).shape[0] // batch_size

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

# print()
# print('Problem 1: Turn the logistic regression example with SGD into a 1-hidden layer '
# 	'neural network with rectified linear units and 1024 hidden nodes. This model should '
# 	'improve your validation / test accuracy.')