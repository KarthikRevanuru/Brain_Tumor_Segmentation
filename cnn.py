import tensorflow as tf
import numpy as np

x = tf.placeholder('float')
y = tf.placeholder('float')

def conv3d(x, W):
	#conv3d(input,filter,strides,padding,data_format='NCDHW',name=None) i'm taking channels,depth,height,width
	return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME',data_format='NCDHW')

def relu(x):
	#relu(features,name=None)
	return tf.nn.relu(x)

def maxpool3d(x):
	#max_pool3d(input,ksize,strides,padding,data_format='NCDHW',name=None) i'm taking channels,depth,height,width 
	#How much should i move my channel
	return tf.nn.max_pool3d(x, ksize=[1,1,3,3,3], strides=[1,1,2,2,2],data_format='NCDHW',padding='SAME')
	

	
def build_network(x):
	#conv1+relu, conv2+relu, conv3+relu, conv4, conv5
	weights = {'W_conv1':tf.Variable(tf.random_normal([4,128,3,3,3])),
	'W_conv2':tf.Variable(tf.random_normal([128,256,3,3,3])),
	'W_conv3':tf.Variable(tf.random_normal([256,512,3,3,3])),
	'W_conv4':tf.Variable(tf.random_normal([512,256,3,3,3])),
	'W_conv5':tf.Variable(tf.random_normal([256,128,3,3,3])),
	'W_conv6':tf.Variable(tf.random_normal([128,4,3,3,3]))}
               
	biases = {'b_conv1':tf.Variable(tf.random_normal([128])),
	'b_conv2':tf.Variable(tf.random_normal([256])),
	'b_conv3':tf.Variable(tf.random_normal([512])),
	'b_conv4':tf.Variable(tf.random_normal([256])),
	'b_conv5':tf.Variable(tf.random_normal([128])),
	'b_conv6':tf.Variable(tf.random_normal([4]))}
	
	x = tf.reshape(x, shape=[-1, 4,240,240,155])#Think about batch	
	
	conv1 = conv3d(x, weights['W_conv1']) + biases['b_conv1']
	relu1 = relu(conv1)
	
	conv2 = conv3d(relu1, weights['W_conv2']) + biases['b_conv2']
	relu2 = relu(conv2)
	
	conv3 = conv3d(relu2, weights['W_conv3']) + biases['b_conv3']
	relu3 = relu(conv3)
	
	conv4 = conv3d(relu3, weights['W_conv4']) + biases['b_conv4']
	conv5 = conv3d(conv4, weights['W_conv5']) + biases['b_conv5']
	conv6 = conv3d(conv5, weights['W_conv6']) + biases['b_conv6']
	
	#fc = tf.nn.dropout(conv6,0.8) #Think about dropout
	
	
	return conv6
	
def train_neural_network(x):
	prediction = build_network(x) #Prediction size
	#cost = tf.reduce_mean(cost_mat(pred[0], y[0]) + cost_mat(pred[1], y[1]) + cost_mat(pred[2], y[2]) + ...) #cost_mat -> custom cost fn to find error in matrixes perhaps sum of pixel wise difference
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #Think about cost function and how to return build network
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
	hm_epochs = 100
	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		sess.run(tf.global_variables_initializer())
        	successful_runs = 0
        	total_runs = 0
        	for epoch in range(hm_epochs):
            		epoch_loss = 0
            		for data in train_data:
            			total_runs += 1
            			try:
            				X = data[0]
            				Y = data[1]
            				_, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
            				epoch_loss += c
            				successful_runs += 1
            				#print successful_runs,total_runs
            			except Exception as e:
            				# I am passing for the sake of notebook space, but we are getting 1 shaping issue from one 
            				# input tensor. Not sure why, will have to look into it. Guessing it's
            				# one of the depths that doesn't come to 20.
            				#pass
            				print(str(e))
            		print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            		
            		
def test_my_network(X):
	y=sess.run(feed_dict={x: X})
	
#cost function and how to train all images, visualize images at the end, test for 10-15 images, add tensor board, save weights.		
