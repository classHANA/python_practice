import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
########## RNN - Long Sentence With Batch ###################
sentence = ("영국이 낳은 세계 최고 극작가로 불리고 있는 셰익스피어는 잉글랜드 중부의 영국의 전형적인 소읍 스트랫퍼드 어폰 에이번에서 출생하였다. 셰익스피어는 아름다운 숲과 계곡으로 둘러싸인 인구 2000명 정도의 작은 마을 스트랫퍼드에서 존 부부의 첫 번째아들로, 8남매 중 셋째로 태어났고, 이곳에서 학교를 다녔다")

char_set = list(set(sentence))
char_dic = {w: i for i,w in enumerate(char_set)}

data_dim = len(char_dic)
hidden_size = len(char_dic)
num_classes = len(char_dic)
sequence_length = 40
learning_rate = 0.1

# X, Y 여러개 - batch_size 만큼 여러개
dataX = []
dataY = []

for i in range(0,len(sentence)-sequence_length):

	x_str = sentence[i:i+sequence_length]
	y_str = sentence[i+1:i+1+sequence_length]

	x = [char_dic[c] for c in x_str]
	y = [char_dic[c] for c in y_str]

	dataX.append(x)
	dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X,num_classes)

def lstm_cell():
	cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
	return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell()] * 2, state_is_tuple=True)
outputs, _state = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

## FC Layer
X_for_fc = tf.reshape(outputs, [-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
	logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	_, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X:dataX, Y:dataY})

	for j, result in enumerate(results):
		index = np.argmax(result, axis=1)
		print (i,j,''.join([char_set[t] for t in index]))

results = sess.run(outputs, feed_dict={X:dataX})
for j, result in enumerate(results):
	index = np.argmax(result, axis=1)
	if j is 0:
		print (''.join([char_set[t] for t in index]), end='')
	else:
		print (char_set[index[-1]], end='')

################ RNN - if you want you ######################
# sample = "if you want you"
# idx2char = list(set(sample))
# # char -> idx
# char2idx = {c:i for i,c in enumerate(idx2char)}

# dic_size = len(char2idx) # input dim
# hidden_size = len(char2idx)
# num_classes = len(char2idx)
# batch_size = 1
# sequence_length = len(sample)-1
# learning_rate = 0.1

# sample_idx = [char2idx[c] for c in sample] #[1,1,2,3,4,2,5,3]
# x_data = [sample_idx[:-1]]
# y_data = [sample_idx[1:]]

# X = tf.placeholder(tf.int32, [None, sequence_length])
# Y = tf.placeholder(tf.int32, [None, sequence_length])

# x_one_hot = tf.one_hot(X,num_classes) # [1]->[010000000]

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# initial_state = cell.zero_state(batch_size,tf.float32)
# outputs, _state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# # FC Layer
# x_for_fc = tf.reshape(outputs, [-1,hidden_size])
# outputs = tf.contrib.layers.fully_connected(x_for_fc, num_classes, activation_fn=None)

# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# weights = tf.ones([batch_size, sequence_length])
# sequence_loss = tf.contrib.seq2seq.sequence_loss(
# 				logits=outputs, targets=Y, weights=weights)
# loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# prediction = tf.argmax(outputs, axis = 2)

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
	
# 	for i in range(100):
# 		l, _ = sess.run([loss,train], feed_dict={X:x_data, Y:y_data})
# 		result = sess.run(prediction, feed_dict={X:x_data})
# 		result_str = [idx2char[c] for c in np.squeeze(result)] # 보기 좋게 char로 변환
# 		print (i, "loss:",l,"Prediction:",''.join(result_str))


################ RNN - hihello ############################
# #idx2char[2]->e
# idx2char = ['h','i','e','l','o']
# # hihell - > ihello
# x_data = [[0,1,0,2,3,3]] #hihell
# x_one_hot = [[[1,0,0,0,0], #h 0
# 			  [0,1,0,0,0], #i 1
# 			  [1,0,0,0,0], #h 0
# 			  [0,0,1,0,0], #e 2
# 			  [0,0,0,1,0], #l 3
# 			  [0,0,0,1,0]]] #l 3
# y_data = [[1,0,2,3,3,4]] #ihello

# num_classes = 5
# input_dim = 5
# hidden_size = 5
# batch_size = 1
# sequence_length = 6
# learning_rate = 0.1

# # Shape!
# X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
# Y = tf.placeholder(tf.int32, [None, sequence_length])

# cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# initial_state = cell.zero_state(batch_size, tf.float32)
# outputs, _states = tf.nn.dynamic_rnn(cell,X,initial_state=initial_state,
# 									 dtype=tf.float32)

# # FC Layer
# x_for_fc = tf.reshape(outputs, [-1, hidden_size])
# outputs = tf.contrib.layers.fully_connected(
# 			inputs=x_for_fc, num_outputs=num_classes, activation_fn=None)

# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# weights = tf.ones([batch_size, sequence_length])

# # sequence loss : sequence를 위한 cost function! 
# sequence_loss = tf.contrib.seq2seq.sequence_loss(
# 						logits=outputs, targets=Y, weights=weights)
# loss = tf.reduce_mean(sequence_loss)

# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# prediction = tf.argmax(outputs, axis=2)

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	for i in range(50):
# 		l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
# 		result = sess.run(prediction, feed_dict={X:x_one_hot})
# 		print (i, "loss:", l, "prediction:",result, "true Y:",y_data)

# 		result_str = [idx2char[c] for c in np.squeeze(result)] # 배열만들어주기
# 		print ("\t Prediction str:","".join(result_str)) # 배열 -> 문자열


################ Ensemble CNN MNIST #######################

# 0. Hypter Parameter Setting & Data Load
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# learning_rate = 0.001
# training_epochs = 20
# batch_size = 100

# # 1. Convolutional Layer 3개 짜리 Model Class 생성
# class Model:
	
# 	def __init__(self, sess, name):
# 		self.sess = sess
# 		self.name = name
# 		self._build_net()

# 	def _build_net(self):
# 		with tf.variable_scope(self.name):

# 			#T : 학습 단계, F : 실행 단계 - 드랍아웃 적용시 필요
# 			self.training = tf.placeholder(tf.bool)

# 			self.X = tf.placeholder(tf.float32, [None, 784])
# 			X_img = tf.reshape(self.X, [-1,28,28,1])

# 			self.Y = tf.placeholder(tf.float32, [None, 10])

# 			# Conv Layer 1
# 			# Image = [28,28,1]
# 			# conv1 = [28,28,32]
# 			# pool1 = [14,14,32]
# 			conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3,3],
# 							padding="SAME", activation=tf.nn.relu)
# 			pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],
# 							padding="SAME", strides=2)
# 			dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

# 			# Conv Layer 2
# 			# Image = [14,14,32]
# 			# conv2 = [14,14,64]
# 			# pool2 = [7,7,64]
# 			conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3,3],
# 							padding="SAME", activation=tf.nn.relu)
# 			pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],
# 							padding="SAME", strides=2)
# 			dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

# 			# Conv Layer 3
# 			# Image = [7,7,64]
# 			# conv3 = [7,7,128]
# 			# pool3 = [4,4,128]
# 			conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3,3],
# 							padding="SAME", activation=tf.nn.relu)
# 			pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2],
# 							padding="SAME", strides=2)
# 			dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

# 			# FC(Fully Connected) Layer / Dense Layer
# 			flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
# 			dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
# 			dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

# 			# Logistic Layer
# 			self.logits = tf.layers.dense(inputs=dropout4, units=10)

# 		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# 			logits=self.logits, labels=self.Y))
# 		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# 		correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.Y,1))
# 		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 	def predict(self,x_test, training=False):
# 		return self.sess.run(self.logits, feed_dict={
# 			self.X : x_test, self.training: training})

# 	def get_accuracy(self, x_test, y_test, training=False):
# 		return self.sess.run(self.accuracy, feed_dict={
# 			self.X:x_test, self.Y:y_test, self.training:training})

# 	def train(self, x_data, y_data, training=True):
# 		return self.sess.run([self.cost, self.optimizer], feed_dict={
# 			self.X : x_data, self.Y : y_data, self.training: training})

# sess = tf.Session()
# m1 = Model(sess, "m1")

# sess.run(tf.global_variables_initializer())

# print ("Learning Start!")
# for epoch in range(training_epochs):
# 	avg_cost = 0
# 	total_batch = int(mnist.train.num_examples / batch_size)

# 	for i in range(total_batch):
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
# 		c, _ = m1.train(batch_xs, batch_ys)
# 		avg_cost += c/total_batch

# 	print ("Epoch:",'%04d' % (epoch+1), 'cost=','{:.9f}'.format(avg_cost))

# print ("Learning Finished!")

# print ("Accuracy:", m1.get_accuracy(mnist.test.images, mnist.test.labels))

# 2. Ensemble 적용
# sess = tf.Session()

# models = []
# num_models = 2

# for m in range(num_models):
# 	models.append(Model(sess, "model"+str(m)))

# sess.run(tf.global_variables_initializer())

# print ("Learning Start!")

# for epoch in range(training_epochs):
# 	avg_cost_list = np.zeros(len(models))
# 	total_batch = int(mnist.train.num_examples/batch_size)

# 	for i in range(total_batch):
# 		batch_xs, batch_ys = mnist.train.next_batch(batch_size)

# 		# models = [model0, model1]
# 		for m_idx, m in enumerate(models):
# 			c, _ = m.train(batch_xs, batch_ys)
# 			avg_cost_list[m_idx] += c/total_batch

# 	print ('Epoch:','%04d' % (epoch+1),'cost=',avg_cost_list)

# print ("Learning Finished!")
# test_size = len(mnist.test.labels)
# predictions = np.zeros([test_size,10])
# for m_idx, m in enumerate(models):
# 	print (m_idx, 'Accuracy:', m.get_accuracy(mnist.test.images, mnist.test.labels))
# 	p = m.predict(mnist.test.images)
# 	predictions += p

# ensemble_correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(
# 								mnist.test.labels,1))
# ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
# print ("Ensemble Accuracy:", sess.run(ensemble_accuracy))