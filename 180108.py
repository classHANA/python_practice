import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

################ Ensemble CNN MNIST #######################

################ CNN MNIST with CLASS ######################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:
	
	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self._build_net()

	def _build_net(self):
		with tf.variable_scope(self.name):
			self.training = tf.placeholder(tf.bool)
			self.X = tf.placeholder(tf.float32, [None,784])
			X_img = tf.reshape(self.X, [-1,28,28,1])
			self.Y = tf.placeholder(tf.float32, [None,10])
			# Image : 28 * 28 * 1
			# Conv1 : 28 * 28 * 32
			# Pool1 : 14 * 14 * 32
			conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3],
				padding="SAME", activation=tf.nn.relu)
			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],
				padding="SAME", strides=2)
			dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7,
									training=self.training)

			# Pool1 : 14 * 14 * 32
			# conv2 : 14 * 14 * 64
			# pool2 : 7 * 7 * 64
			conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3],
				padding="SAME", activation=tf.nn.relu)
			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
				padding="SAME", strides=2)
			dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7,
									training=self.training)

			# conv3 : 7 * 7 * 128
			# pool3 : 4 * 4 * 128
			conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3],
				padding="SAME", activation=tf.nn.relu)
			pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2],
				padding="SAME", strides=2)
			dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7,
									training=self.training)

			flat = tf.reshape(dropout3, [-1,4 * 4 * 128])
			dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
			dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5,
									training=self.training)

			self.logits = tf.layers.dense(inputs=dropout4, units=10)

		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.logits, labels=self.Y))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

		correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.Y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def predict(self, x_test, training=False):
		return self.sess.run(self.logits, feed_dict={
			self.X:x_test, self.training:training})

	def get_accuracy(self, x_test, y_test, training=False):
		return self.sess.run(self.accuracy, feed_dict={
			self.X:x_test, self.Y:y_test, self.training:training})

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.cost, self.optimizer], feed_dict={
			self.X:x_data, self.Y:y_data, self.training:training})

#######################
sess = tf.Session()


#######################
# sess = tf.Session()

# m1 = Model(sess, "m1")
# sess.run(tf.global_variables_initializer())

# print ("Learnig Start!")
# for epoch in range(training_epochs):
# 	avg_cost = 0 
# 	total_batch = int(mnist.train.num_examples/batch_size)

# 	for i in range(total_batch):
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# 		c, _ = m1.train(batch_xs,batch_ys)
# 		avg_cost += c/total_batch

# 	print ("Epoch","%04d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))

# print ("Learnig Finished!")
# print ("Accuracy:", m1.get_accuracy(mnist.test.images, mnist.test.labels))

################ CNN MNIST #################################
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100

# X = tf.placeholder(tf.float32, [None,784])
# X_img = tf.reshape(X, [-1,28,28,1])
# Y = tf.placeholder(tf.float32, [None,10])

# # L1 Image : (-1,28,28,1)
# # Conv : (-1,28,28,32)
# # Pool : (-1,14,14,32)
# W1 = tf.Variable(tf.random_normal([3,3,1,32]))
# L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# # L2 Image : (-1,14,14,32)
# # Conv : (-1,14,14,64)
# # Pool : (-1,7,7,64)
# W2 = tf.Variable(tf.random_normal([3,3,32,64]))
# L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# L2_flat = tf.reshape(L2,[-1,7*7*64])
# print (L2_flat)

# # Fully Connected(FC) Network
# W3 = tf.get_variable("W3", shape=[7*7*64,10],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([10]))
# logits = tf.matmul(L2_flat,W3)+b


# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# 					logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# print ("Learnig Start!")
# for epoch in range(training_epochs):
# 	avg_cost = 0
# 	total_batch = int(mnist.train.num_examples/batch_size)
# 	for i in range(total_batch):
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# 		feed_dict = {X:batch_xs, Y:batch_ys}
# 		c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
# 		avg_cost += c/total_batch
# 	print ("Epoch:",'%04d' % (epoch+1), 'cost=','{:.9f}'.format(avg_cost))

# print ("Learnig Finished!")

# correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print ("Accuracy:",sess.run(accuracy, feed_dict={
# 	X:mnist.test.images, Y: mnist.test.labels}))

# r = random.randint(0, mnist.test.num_examples-1)
# print ("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
# print ("Prediction:", sess.run(tf.argmax(logits,1), feed_dict={
# 	X:mnist.test.images[r:r+1]}))
#########################################################################################

# sess = tf.InteractiveSession()
# image = np.array([[[[1],[2],[3]],
#                    [[4],[5],[6]], 
#                    [[7],[8],[9]]]], dtype=np.float32)
# print(image.shape)
# plt.imshow(image.reshape(3,3), cmap='Greys')
# plt.show()

# print("imag:\n", image)
# print("image.shape", image.shape)
# weight = tf.constant([[[[1.]],[[1.]]],
#                       [[[1.]],[[1.]]]])
# print("weight.shape", weight.shape)
# conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
# conv2d_img = conv2d.eval()

# print("conv2d_img.shape", conv2d_img.shape)
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     print(one_img.reshape(2,2))
#     plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray'),plt.show()

# # print("imag:\n", image)
# print("image.shape", image.shape)

# weight = tf.constant([[[[1.]],[[1.]]],
#                       [[[1.]],[[1.]]]])
# print("weight.shape", weight.shape)
# conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
# conv2d_img = conv2d.eval()
# print("conv2d_img.shape", conv2d_img.shape)
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     print(one_img.reshape(3,3))
#     plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray'),plt.show()


# # print("imag:\n", image)
# print("image.shape", image.shape)

# weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
#                       [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
# print("weight.shape", weight.shape)
# conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
# conv2d_img = conv2d.eval()
# print("conv2d_img.shape", conv2d_img.shape)
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     print(one_img.reshape(3,3))
#     plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray'),plt.show()
    
# image = np.array([[[[4],[3]],
#                     [[2],[1]]]], dtype=np.float32)
# pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
#                     strides=[1, 1, 1, 1], padding='VALID')
# print(pool.shape)
# print(pool.eval())

# image = np.array([[[[4],[3]],
#                     [[2],[1]]]], dtype=np.float32)
# pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
#                     strides=[1, 1, 1, 1], padding='SAME')
# print(pool.shape)
# print(pool.eval())

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# img = mnist.train.images[0].reshape(28,28)
# plt.imshow(img, cmap='gray'),plt.show()

# sess = tf.InteractiveSession()

# img = img.reshape(-1,28,28,1)
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
# print(conv2d)
# sess.run(tf.global_variables_initializer())
# conv2d_img = conv2d.eval()
# conv2d_img = np.swapaxes(conv2d_img, 0, 3)
# for i, one_img in enumerate(conv2d_img):
#     plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')

# pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[
#                         1, 2, 2, 1], padding='SAME')
# print(pool)
# sess.run(tf.global_variables_initializer())
# pool_img = pool.eval()
# pool_img = np.swapaxes(pool_img, 0, 3)
# for i, one_img in enumerate(pool_img):
#     plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray'),plt.show()
## MNIST with Deep Neural Network
## activation function : Relu function
## layer : 5, 중간 layer node 수: 512
## 초기화 방법 : xavier initializer
## Optimizer : Adam Optimizer
## Dropout rate: 0.7
## learning_rate = 0.001
## Epochs = 15
## batch_size = 100
## Result Accuracy : 약 97~98

# import random
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100

# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)

# W1 = tf.get_variable("W1", shape=[784,512],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
# L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

# W2 = tf.get_variable("W2", shape=[512,512],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([512]))
# L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
# L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

# W3 = tf.get_variable("W3", shape=[512,512],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([512]))
# L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
# L3 = tf.nn.dropout(L3,keep_prob=keep_prob)

# W4 = tf.get_variable("W4", shape=[512,512],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([512]))
# L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
# L4 = tf.nn.dropout(L4,keep_prob=keep_prob)

# W5 = tf.get_variable("W5", shape=[512,10],
# 		initializer=tf.contrib.layers.xavier_initializer())
# b5 = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L4,W5)+b5

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# 	logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for epoch in range(training_epochs):
# 	avg_cost = 0
# 	total_batch = int(mnist.train.num_examples / batch_size)

# 	for i in range(total_batch):
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

# 		feed_dict={X:batch_xs, Y:batch_ys, keep_prob:0.7}
# 		c,_ = sess.run([cost, optimizer], feed_dict=feed_dict)
# 		avg_cost += c/total_batch

# 	print ('Epoch:','%04d' % (epoch+1),'cost=','{:.9f}'.format(avg_cost))

# print ('Learning Finished!')

# correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print ('Accuracy:', sess.run(accuracy, feed_dict={
# 	X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

# r = random.randint(0,mnist.test.num_examples-1)
# print ('Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
# print ('Prediction:', sess.run(
# 	tf.argmax(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1],keep_prob:1}))



########## relu, xavier_initializer, AdamOptimizer
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100

# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)

# W1 = tf.get_variable("W1", shape=[784,256],
#                 initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.Variable(tf.random_normal([256]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# # dropout
# L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

# W2 = tf.get_variable("W2", shape=[256,256],
#                 initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([256]))
# L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
# # dropout
# L2 = tf.nn.dropout(L2,keep_prob=keep_prob)

# W3 = tf.get_variable("W3", shape=[256,10],
#                 initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([10]))
# hypothesis = tf.matmul(L2, W3) + b3

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for epoch in range(training_epochs):
#     avg_cost = 0
#     total_batch = int(mnist.train.num_examples / batch_size)

#     for i in range(total_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         # dropout parameter setting
#         feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
#         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
#         avg_cost += c / total_batch

#     print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # dropout parameter setting
# print('Accuracy:', sess.run(accuracy, feed_dict={
#       X: mnist.test.images, Y: mnist.test.labels, keep_prob:1}))