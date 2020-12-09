import scipy.io as sio
import numpy as np
import keras.backend as K
from keras.layers import Input, GlobalAveragePooling2D, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation
from keras.models import Model

import constants as c


# Block of layers: Conv --> BatchNorm --> ReLU --> Pool
def conv_bn_pool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	pool='',pool_size=(2, 2),pool_strides=None,
	conv_layer_prefix='conv'):
	x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor) 
	# đệm vào input để thuận tiện cho việc trượt các filter (có bước nhảy)
	x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x) 
	# sử dụng cửa sổ trượt (filter) để trích xuất các đặc trưng của âm thanh
	x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = Activation('relu', name='relu{}'.format(layer_idx))(x) 
	# ReLU layer áp dụng các kích hoạt (activation function) max(0,x) đưa các giá trị âm về 0, mục đích là đảm bảo các giá trị không tuyến tính qua cấc layer (khử tuyến tính)
	# không làm thay đổi kích thước của input

	# Pooling layer: giảm chiều không gian của đầu vào, giảm độ phức tạp cần tính toán, thường hay dùng max pooling hơn
	# giữ lại chi tiết quan trọng, max pooling: giữ lại pixel có giá trị lớn nhất, average pooling: tính trung bình các giá trị
	if pool == 'max':
		x = MaxPooling2D(pool_size=pool_size,strides=pool_strides,name='mpool{}'.format(layer_idx))(x)
	elif pool == 'avg':
		x = AveragePooling2D(pool_size=pool_size,strides=pool_strides,name='apool{}'.format(layer_idx))(x)
	return x

# FC layer: Fully connected layer: lớp kết nối đầy đủ như mạng NN thông thường
# là phẳng input (flattent) thành 1 vector thay vì là 1 mảng nhiều chiều, rồi sortmax để phân loại đối tượng dựa vào vector đặc trưng đã tính toán ở các layer trước (tính ra xác suất)
# BatchNorm để giảm thời gian training model, giảm tầm quan trọng của số lượng input vào, tăng tốc độ

# Block of layers: Conv --> BatchNorm --> ReLU --> Dynamic average pool (fc6 -> apool6 only)
def conv_bn_dynamic_apool(inp_tensor,layer_idx,conv_filters,conv_kernel_size,conv_strides,conv_pad,
	conv_layer_prefix='conv'):
	x = ZeroPadding2D(padding=conv_pad,name='pad{}'.format(layer_idx))(inp_tensor)
	x = Conv2D(filters=conv_filters,kernel_size=conv_kernel_size, strides=conv_strides, padding='valid', name='{}{}'.format(conv_layer_prefix,layer_idx))(x)
	x = BatchNormalization(epsilon=1e-5,momentum=1,name='bn{}'.format(layer_idx))(x)
	x = Activation('relu', name='relu{}'.format(layer_idx))(x)
	x = GlobalAveragePooling2D(name='gapool{}'.format(layer_idx))(x)
	x = Reshape((1,1,conv_filters),name='reshape{}'.format(layer_idx))(x)
	return x


# VGGVox verification model
def vggvox_model():
	inp = Input(c.INPUT_SHAPE,name='input')
	x = conv_bn_pool(inp,layer_idx=1,conv_filters=96,conv_kernel_size=(7,7),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=2,conv_filters=256,conv_kernel_size=(5,5),conv_strides=(2,2),conv_pad=(1,1),
		pool='max',pool_size=(3,3),pool_strides=(2,2))
	x = conv_bn_pool(x,layer_idx=3,conv_filters=384,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=4,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1))
	x = conv_bn_pool(x,layer_idx=5,conv_filters=256,conv_kernel_size=(3,3),conv_strides=(1,1),conv_pad=(1,1),
		pool='max',pool_size=(5,3),pool_strides=(3,2))		
	x = conv_bn_dynamic_apool(x,layer_idx=6,conv_filters=4096,conv_kernel_size=(9,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')
	x = conv_bn_pool(x,layer_idx=7,conv_filters=1024,conv_kernel_size=(1,1),conv_strides=(1,1),conv_pad=(0,0),
		conv_layer_prefix='fc')
	x = Lambda(lambda y: K.l2_normalize(y, axis=3), name='norm')(x)
	x = Conv2D(filters=1024,kernel_size=(1,1), strides=(1,1), padding='valid', name='fc8')(x)
	m = Model(inp, x, name='VGGVox')
	return m


def test():
	model = vggvox_model()
	num_layers = len(model.layers)

	x = np.random.randn(1,512,30,1)
	outputs = []

	for i in range(num_layers):
		get_ith_layer_output = K.function([model.layers[0].input, K.learning_phase()],
		                              [model.layers[i].output])	
		layer_output = get_ith_layer_output([x, 0])[0] 	# output in test mode = 0
		outputs.append(layer_output)

	for i in range(11):
		print("Shape of layer {} output:{}".format(i, outputs[i].shape))


if __name__ == '__main__':
	test()
