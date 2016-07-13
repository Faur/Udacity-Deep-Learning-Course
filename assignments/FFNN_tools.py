# 
# Inspiration from:
#     http://deeplearning.net/tutorial/mlp.html

import numpy as np
import theano
import theano.tensor as T



class DenseLayer(object):

	def __init__(self, rng, input, n_in, n_out, activation=T.tanh, W=None, b=None):
		self.input = input

		if W is None:
			W_values = np.asarray(
				rng.uniform(
					low	= -np.sqrt(6. / (n_in + n_out)),
					high=  np.sqrt(6. / (n_in + n_out)),
					size= (n_in, n_out)
				),
				dtype=theano.config.floatX
			)
			if activation == T.nnet.sigmoid:
				W_values *= 4

			# Create W as a shared variable, for GPU optimization
			W = theano.shared(
				value=W_values,
				name='W',
				borrow=True
			)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(
				value=b_values,
				name='b',
				borrow=True
			)

		self.W = W
		self.b = b
		self.params = [self.W, self.b]

		lin_output = T.dot(input, self.W) + self.b
		if activation is None:
			self.output = lin_output
		else:
			self.output = activation(lin_output)
		

class SoftMax(object):

	def __init__(self, input, n_in, n_out):
		self.input = input

		self.W = theano.shared(
			value 	= np.zeros((n_in, n_out), dtype=theano.config.floatX),
			name	= 'W',
			borrow	= True
		)

		self.b = theano.shared(
			value 	= np.zeros((n_out, ), dtype=theano.config.floatX),
			name	= 'b',
			borrow	= True
		)

		self.params = [self.W, self.b]

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		Note: we use the mean instead of the sum so that
		      the learning rate is less dependent on the batch size
		"""
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch

		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] 

		# T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class 

		# LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,

		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(
			T.log(self.p_y_given_x) [T.arange(y.shape[0]), y]
		)

	def errors(self, y):

		# Check if y has the same dimensions as y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred:',
				('y', y.type, 'y_pred', self.y_pred.type)
			)

		# Check that y has the correct datatype
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()




class MLP(object):

	def __init__(self, rng, input, n_in, n_hidden, n_out, activation=T.tanh):
		self.input = input

		self.hiddenLayer = DenseLayer(
			rng=rng,
			input=input,
			n_in=n_in,
			n_out=n_hidden,
			activation=activation
		)

		self.softmaxLayer = SoftMax(
			input	= self.hiddenLayer.output,
			n_in 	= n_hidden,
			n_out 	= n_out
		)

		self.L1 = (
			abs(self.hiddenLayer.W).sum()
			+ abs(self.softmaxLayer.W).sum()
		)

		self.L2 = (
			(self.hiddenLayer.W ** 2).sum()
			+ (self.softmaxLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = (
			self.softmaxLayer.negative_log_likelihood
		)

		self.errors = self.softmaxLayer.errors

		self.params = self.hiddenLayer.params + self.softmaxLayer.params




