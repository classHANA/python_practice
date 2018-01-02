# http://cs231n.stanford.edu/syllabus.html
# https://class.coursera.org/ml-003/lecture

import tensorflow as tf
# import numpy as np

# xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:,0:-1]
# y_data = xy[:,[-1]]

filename_queue = tf.train.string_input_producer(
	['data-01-test-score.csv'], shuffle=False, name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value,record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]], batch_size=10)

# print (x_data.shape, x_data, len(x_data))
# print (y_data.shape, y_data, len(y_data))
# import matplotlib.pyplot as plt

# x_data = [
# 			[73.,80.,75.],
# 			[93.,88.,93.,],
# 			[89.,91.,90.],
# 			[96.,98.,100.],
# 			[73.,66.,70.]
# 		 ]
# # x1_data = [73., 93., 89., 96., 73.]
# # x2_data = [80., 88., 91., 98., 66.]
# # x3_data = [75., 93., 90., 100., 70]
# y_data = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder(tf.float32, shape=[None,3])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2000):
	x_batch, y_batch = sess.run([train_x_batch,train_y_batch])
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
		feed_dict={X:x_batch, Y:y_batch})
	if step % 10 == 0:
		print (step,cost_val,hy_val)

print ("Your score will be",sess.run(hypothesis, feed_dict={X:[[100,70,101]]}))
print ("Your score will be",sess.run(hypothesis, feed_dict={X:[[60,100,80],[70,80,90]]}))

# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# W = tf.Variable(tf.random_normal([1]), name='weight')

# hypothesis = X * W

# cost = tf.reduce_mean(tf.square(hypothesis-Y))

# # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2000):
# 	cost_val, W_val, _ = sess.run([cost, W, update],
# 		feed_dict={X:[1,2,3], Y:[51,69,12]})
# 	if step % 1 == 0:
# 		print (step, cost_val, W_val)

# W_val = []
# cost_val = []

# for i in range(-30,50):
	# feed_W = i * 0.1
	# curr_cost, curr_W = sess.run([cost,W], feed_dict={W:feed_W})
	# W_val.append(curr_W)
	# cost_val.append(curr_cost)

# plt.plot(W_val,cost_val)
# plt.show()




# x_train = tf.placeholder(tf.float32, shape=[None])
# y_train = tf.placeholder(tf.float32, shape=[None])

# # Variables
# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = W * x_train + b

# # cost
# cost = tf.reduce_mean(tf.square(hypothesis-y_train))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2001):
# 	cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
# 		feed_dict={x_train:[1.0,2.0,3.0],y_train:[1.0,2.0,3.0]})
# 	if step % 20 == 0:
# 		print (step, cost_val, W_val, b_val)

# placeholder 
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a+b

# sess = tf.Session()
# print (sess.run(adder_node, feed_dict={a:3,b:4}))

# constant, session 개념
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# node3 = tf.add(node1,node2)

# sess = tf.Session()
# print (sess.run(node3))
# print (sess.run([node1,node2]))

# hello = tf.constant("Hello tensorflow!")
# sess = tf.Session()
# print (sess.run(hello))