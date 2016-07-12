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





