##########################################################################

# Tutorial custom layers with keras:
# https://keras.io/layers/writing-your-own-keras-layers/

# K.relu: 
# https://keras.io/backend/

# original SReLU:
# https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/advanced_activations/srelu.py

from keras import backend as K
from keras.layers import Layer

class MySReLU(Layer):

	def __init__(self, **kwargs):
		super(MySReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		param_shape = tuple(list(input_shape[1:])) # input_shape is: (batch, height, width, channels)
		
		self.tl = self.add_weight(shape=param_shape, name='tl', initializer='zeros', trainable=True)
		self.al = self.add_weight(shape=param_shape, name='al', initializer='uniform', trainable=True)
		self.delta = self.add_weight(shape=param_shape, name='delta', initializer='uniform', trainable=True)
		self.ar = self.add_weight(shape=param_shape, name='ar', initializer='ones', trainable=True)
		
		super(MySReLU, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		# ensure the the right part is always to the right of the left
		tr = self.tl + K.abs(self.delta)
		tl = self.tl
		al = self.al
		ar = self.ar
		
		### Keras Implementation
		
		# The K.relu(x, alpha=0.0, max_value=None, threshold=0.0) works as:
		# 	- with default values, it returns element-wise max(x, 0),
		#		- otherwise:		
		#				- if x >= max_value, then f(x) = max_value
		#				- if threshold <= x < max_value, then f(x) = x
		#				- if x < threshold, then f(x) = alpha * (x - threshold)
		
		# return tl + K.relu(x - tl, al, tr - tl, 0.0) + K.relu(x - tr, 0.0, None, 0.0) * ar
		
		# if x < tl
		#	tl + al*((x - tl) - 0) + 0*((x-tr) - 0)*ar =
		#	tl + (x - tl)*al
		
		# if x >= tr
		#	tl + (tr - tl) + (x - t)*ar =
		# 	tr +  (x - tr)*ar
		
		# otherwise
		#	tl + (x - tl) + 0*ar =
		#	x	

		### My Implementation
		
		eps=0.000001
		if_x_gtr_tr = K.relu(x-tr)/(x-tr+eps); # is 1 if x > tr and is 0 if x <= tr
		if_x_lss_tl = K.relu(tl-x)/(tl-x+eps); # is 1 if x < tl and is 0 if x >= tl
		if_x_btw_tlr = (1-if_x_gtr_tr)*(1-if_x_lss_tl);

		return if_x_gtr_tr*(ar*(x-tr) + tr) + if_x_lss_tl*(al*(x-tl) + tl) + if_x_btw_tlr*x

	def compute_output_shape(self, input_shape):
		return input_shape

##########################################################################
