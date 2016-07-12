import numpy as np
import theano
import theano.tensor as T

class LogisticRegression(object):

	def __init__(self, input, n_in, n_out):
		'''Initalize a logistic regression classifier'''

		# Keep track of model inputs
		self.input = input

		##### Initialize parameters #####
		self.W = theano.shared(
					value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
					name='W',
					borrow=True
					)

		self.b = theano.shared(
					value=np.zeros((n_out,), dtype=theano.config.floatX),
					name='b',
					borrow=True
					)

		# parameters of the model
		self.params = [self.W, self.b]


		##### Initialize functions #####
		# Symbolic expression computing the class membership probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# Symbolic expression computing the predicted class
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)


	def negative_log_likelihood(self, y):
		"""Returns the mean of the negative log-likelihood of the prediction
		of this model given a target. The mean is used in order to allow the 
		use of minibatches."""

		return -T.mean(
					T.log(
						T.sum(
							self.p_y_given_x * y
						)
					)
				)
		# y.shape[0] is the symbolic expression for the number of examples
					# T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])




