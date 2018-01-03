import tensorflow as tf
import numpy as np

### Softmax Classification 실습 ##
### 데이터 : 공유폴더내/iris.csv
xy = np.loadtxt('iris.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

### Softmax Classification ####
# xy = np.loadtxt('data-04-zoo.csv', delimiter=",", dtype=np.float32)
# x_data = xy[:,0:-1]
# y_data = xy[:,[-1]]

# print (x_data, y_data)

X = tf.placeholder(tf.float32, [None,4])
Y = tf.placeholder(tf.int32, [None,1])

# One Hot Encoding
Y_one_hot = tf.one_hot(Y,4)
Y_one_hot = tf.reshape(Y_one_hot,[-1,4])

W = tf.Variable(tf.random_normal([4,4]), name='weight')
b = tf.Variable(tf.random_normal([4]), name='bias')

logits = tf.matmul(X,W)+b
hypothesis = tf.nn.softmax(logits)

# Cross Entropy
cost_i = tf.nn.softmax_cross_entropy_with_logits(
		logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# accuracy 계산
prediction = tf.argmax(hypothesis,1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for step in range(2000):
		sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
		if step % 100:
			loss, acc = sess.run([cost, accuracy], 
							feed_dict={X:x_data, Y:y_data})
			print ("Step=",step,"Loss=",loss,"Acc=",acc,"\n")

	pred = sess.run(prediction, 
							feed_dict={X:x_data})

	# 실제 값과 비교
	for p, y in zip(pred, y_data.flatten()):
		print (p,y)

########## 복습 #################
# Regression : x -> 분류, x -> y
# 공부한 시간 -> 내 점수 예측
# x_data = [1,2,3,4,5,6]
# y_data = [3,6,9,12,15,18]

# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x_data * W + b
# cost = tf.reduce_mean(tf.square(hypothesis-y_data))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2000):
# 	sess.run(train)
# 	if step % 20 == 0:
# 		print (step, sess.run(cost), sess.run(W), sess.run(b))

####### Logistic Classification ##############
# xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:,0:-1]
# y_data = xy[:,[-1]]

# # x_data = [[1,2],[2,3],[4,1],[4,3],[5,3],[6,2]]
# # y_data = [[0],[0],[0],[1],[1],[1]]

# X = tf.placeholder(tf.float32, shape=[None,8])
# Y = tf.placeholder(tf.float32, shape=[None,1])

# W = tf.Variable(tf.random_normal([8,1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
# # hypothesis = tf.div(1,(1+tf.exp(tf.matmul(X,W)+b)))
# cost = -tf.reduce_mean(Y * tf.log(hypothesis)+ (1-Y) * tf.log(1-hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())

# 	for i in range(10000):
# 		cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
# 		if i % 200 == 0:
# 			print (i, cost_val)

# 	h,c,a = sess.run([hypothesis, predicted, accuracy],
# 				feed_dict={X:x_data, Y:y_data})
# 	print ("\nHypothesis=",h,"\nCorrect=",c,"\nAccuracy=",a)


# 주식정보 가져오는거
# pip install googlefinance.client
# from googlefinance.client import get_price_data

# param = {
# 	'q':"GOOGL", # Stock Symbol
# 	'i':"86400", # Interval size(second)
# 	'x':"NASD", # Stock exchange symbol
# 	'p':"1Y"  # Period
# }

# result = get_price_data(param)
# print (result)