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

			if activation == T.nnet.relu:
				b_values += 1

		self.W = W
		self.b = b
		self.params = [self.W, self.b]

		lin_output = T.dot(input, self.W) + self.b
		if activation is None:
			self.output = lin_output
		else:
			self.output = activation(lin_output)


def dropout(rng, input, drop_rate=0.2):
	# TODO: implement a way of disabeling dropout for testing purposes

	srng = T.shared_randomstreams.RandomStreams(rng.randint(1000000))

	# Use p = 1 - drop_rate because 1's indicate keep
	mask = srng.binomial(n=1, p=1-drop_rate, size=input.shape)
	# The cast is important because: int * float32 = float64, which is bad for the GPU
	return input * T.cast(mask, theano.config.floatX) / (1 - drop_rate)
	


class SoftMaxLayer(object):

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


class MLP_multi(object):

	def __init__(self, rng, input, n_in, n_hidden_1, n_hidden_2, n_hidden_3, n_out,
			activation=T.tanh):
		
		self.input = input

		input_drop = dropout(rng, input, drop_rate=0.4)

		self.hiddenLayer_1 = DenseLayer(
			rng 	= rng,
			# input 	= input,
			input 	= input_drop,
			n_in 	= n_in,
			n_out 	= n_hidden_1,
			activation=activation
		)
		hidden_1_drop = dropout(rng, self.hiddenLayer_1.output, drop_rate=0.15)

		self.hiddenLayer_2 = DenseLayer(
			rng 	= rng,
			# input 	= input,
			input 	= hidden_1_drop,
			n_in 	= n_hidden_1,
			n_out 	= n_hidden_2,
			activation=activation
		)
		hidden_2_drop = dropout(rng, self.hiddenLayer_2.output, drop_rate=0.15)

		self.hiddenLayer_3 = DenseLayer(
			rng 	= rng,
			# input 	= input,
			input 	= hidden_2_drop,
			n_in 	= n_hidden_2,
			n_out 	= n_hidden_3,
			activation=activation
		)
		hidden_3_drop = dropout(rng, self.hiddenLayer_3.output, drop_rate=0.15)


		self.softmaxLayer = SoftMaxLayer(
			# input	= self.hiddenLayer.output,
			input	= hidden_3_drop,
			n_in 	= n_hidden_3,
			n_out 	= n_out
		)

		self.L1 = (
			abs(self.hiddenLayer_1.W).sum()
			+ abs(self.hiddenLayer_2.W).sum()
			+ abs(self.hiddenLayer_3.W).sum()
			+ abs(self.softmaxLayer.W).sum()
		)

		self.L2 = (
			(self.hiddenLayer_1.W ** 2).sum()
			+ (self.hiddenLayer_2.W ** 2).sum()
			+ (self.hiddenLayer_3.W ** 2).sum()
			+ (self.softmaxLayer.W ** 2).sum()
		)

		self.negative_log_likelihood = (
			self.softmaxLayer.negative_log_likelihood
		)

		self.errors = self.softmaxLayer.errors

		self.params = (self.hiddenLayer_1.params + self.hiddenLayer_2.params 
				+ self.hiddenLayer_3.params + self.softmaxLayer.params)




