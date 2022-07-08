from keras import layers
from keras.layers import Input, Activation, ZeroPadding2D, BatchNormalization, Conv2D
from keras.layers import MaxPooling2D, Dropout, Add, Conv2DTranspose, concatenate
from keras.layers import SeparableConv2D
from keras.models import Model

def ConvBlock(X, n_filters, kernel_size=(3, 3)):
	X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(n_filters, kernel_size, strides = (1, 1) )(X)
	X = BatchNormalization(axis = 3 )(X)
	X = Activation('relu')(X)
	return X


def DepthwiseSeparableConvBlock(X, n_filters, kernel_size=(3, 3)):
	X = ZeroPadding2D((1, 1))(X)
	X = SeparableConv2D(n_filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None)(X)
	X = BatchNormalization(axis = 3 )(X)
	X = Activation('relu')(X)
	return X


def conv_transpose_block(X, n_filters, kernel_size=(3, 3)):
	X = Conv2DTranspose(n_filters, kernel_size, strides=(2, 2), padding='same',output_padding= (1,1), activation=None)(X)
	X = Activation('relu')(X)
	return X


def build_model(input_shape, preset_model, num_classes):
	has_skip = False
	if preset_model == "MobileUNet":
		has_skip = False
	elif preset_model == "MobileUNet-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported MobileUNet model '%s'. This function only supports MobileUNet and MobileUNet-Skip" % (preset_model))


	# Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
	X_input = Input(input_shape)

	# Zero-Padding: pads the border of X_input with zeroes

	X = ConvBlock(X_input, 64)

	X = DepthwiseSeparableConvBlock(X, 64)

	# MAXPOOL
	X = MaxPooling2D((2, 2) )(X)

	skip_1 = X

	X = DepthwiseSeparableConvBlock(X, 128)
	X = DepthwiseSeparableConvBlock(X, 128)
	X = MaxPooling2D((2, 2) )(X)
	skip_2 = X

	X = DepthwiseSeparableConvBlock(X, 256)
	X = DepthwiseSeparableConvBlock(X, 256)
	X = DepthwiseSeparableConvBlock(X, 256)
	X = MaxPooling2D((2, 2) )(X)
	skip_3 = X

	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = MaxPooling2D((2, 2) )(X)
	skip_4 = X

	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = MaxPooling2D((2, 2) )(X)

	#####################
	# Upsampling path #
	#####################
	X = conv_transpose_block(X, 512)

	X = DepthwiseSeparableConvBlock(X, 512)

	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	if has_skip:
		X = concatenate([X, skip_4],axis = -1)

	X = conv_transpose_block(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 512)
	X = DepthwiseSeparableConvBlock(X, 256)
	if has_skip:
		X = concatenate([X, skip_3],axis = -1)

	X = conv_transpose_block(X, 256)
	X = DepthwiseSeparableConvBlock(X, 256)
	X = DepthwiseSeparableConvBlock(X, 256)
	X = DepthwiseSeparableConvBlock(X, 128)
	if has_skip:
		X = concatenate([X, skip_2],axis = -1)

	X = conv_transpose_block(X, 128)
	X = DepthwiseSeparableConvBlock(X, 128)
	X = DepthwiseSeparableConvBlock(X, 64)
	if has_skip:
		X = concatenate([X, skip_1],axis = -1)

	X = conv_transpose_block(X, 64)
	X = DepthwiseSeparableConvBlock(X, 64)
	X = DepthwiseSeparableConvBlock(X, 64)

	#####################
	#      Softmax      #
	#####################
	#X = ZeroPadding2D((1, 1))(X)
	X = Conv2D(num_classes, kernel_size = [1, 1], strides = (1, 1) )(X)
       
	X = Activation('softmax')(X)
	
        
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
	model = Model(inputs = X_input, outputs = X, name='MobileUNet')
	return model
