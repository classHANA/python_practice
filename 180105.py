import tensorflow as tf
import numpy as np

########## MNIST - NN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

########## Neural Network
# x_data = [[0, 0],
#           [0, 1],
#           [1, 0],
#           [1, 1]]
# y_data = [[0],
#           [1],
#           [1],
#           [0]]

# X = tf.placeholder(tf.float32, [None,2])
# Y = tf.placeholder(tf.float32, [None,1])

# ########## Wide Neural Network
# W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
# b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

# W2 = tf.Variable(tf.random_normal([10,10]), name='weight2')
# b2 = tf.Variable(tf.random_normal([10]), name='bias2')
# layer2 = tf.sigmoid(tf.matmul(layer1,W2)+b2)

# W3 = tf.Variable(tf.random_normal([10,10]), name='weight3')
# b3 = tf.Variable(tf.random_normal([10]), name='bias3')
# layer3 = tf.sigmoid(tf.matmul(layer2,W3)+b3)

# W4 = tf.Variable(tf.random_normal([10,1]), name='weight4')
# b4 = tf.Variable(tf.random_normal([1]), name='bias4')
# hypothesis = tf.sigmoid(tf.matmul(layer3,W4)+b4)

# cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1 - hypothesis))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

# with tf.Session() as sess:
#      sess.run(tf.global_variables_initializer())

#      for step in range(20000):
#           sess.run(train, feed_dict={X:x_data, Y: y_data})
#           if step % 100 == 0:
#                print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data})) 

#      h,c,a = sess.run([hypothesis,cost,accuracy],
#           feed_dict={X:x_data, Y:y_data})
#      print("H = ",h,"C = ",c,"A = ",a)



########## MNIST

# import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
# import random

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# # plt.imshow(
# #      mnist.test.images[0].reshape(28,28),
# #      cmap='Greys',
# #      interpolation='nearest'
# #      )
# # plt.show()

# nb_classes=10

# X = tf.placeholder(tf.float32, [None,784])
# Y = tf.placeholder(tf.float32, [None,nb_classes])

# W = tf.Variable(tf.random_normal([784,nb_classes]))
# b = tf.Variable(tf.random_normal([nb_classes]))

# hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

# # epoch, batch_size, iterations
# # epoch : 데이터 전체를 학습하는 횟수
# # batch_size : 한번에 학습할 트레이닝 데이터 사이즈
# # iterations : 1 epoch를 하기 위해서 몇 바퀴나 해야되나
# training_epochs = 15
# batch_size = 100

# with tf.Session() as sess:
#      sess.run(tf.global_variables_initializer())
     
#      for epoch in range(training_epochs):
#           avg_cost = 0
#           #iterations 수 구하기
#           total_batch = int(mnist.train.num_examples / batch_size)

#           for i in range(total_batch):
#                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#                c, _ = sess.run([cost, optimizer],
#                     feed_dict={X:batch_xs, Y:batch_ys})
#                avg_cost = avg_cost + (c/total_batch)
#           print (epoch,avg_cost)

#      acc = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
#      print (acc)

#      r = random.randint(0, mnist.test.num_examples -1)
#      print ("Label :", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
#      print ("Prediction :", sess.run(tf.argmax(hypothesis,1), 
#           feed_dict={X: mnist.test.images[r:r+1]}))

#      plt.imshow(
#           mnist.test.images[r:r+1].reshape(28,28),
#           cmap='Greys',
#           interpolation='nearest'
#      )
#      plt.show()


########## Data Normalization / Linear Regression

# def MinMaxScalar(data):
#      num = data - np.min(data,0)
#      denominator = np.max(data,0) - np.min(data,0)
#      result = num / denominator
#      return result


# xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
#                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
#                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
#                [816, 820.958984, 1008100, 815.48999, 819.23999],
#                [819.359985, 823, 1188100, 818.469971, 818.97998],
#                [819, 823, 1198100, 816, 820.450012],
#                [811.700012, 815.25, 1098100, 809.780029, 813.669983],
#                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

# xy = MinMaxScalar(xy)

# x_data = xy[:,0:-1]
# y_data = xy[:,[-1]]

# X = tf.placeholder(tf.float32, shape=[None, 4])
# Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([4, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.matmul(X, W) + b

# cost = tf.reduce_mean(tf.square(hypothesis - Y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# for step in range(2000):
#     cost_val, hy_val, _ = sess.run(
#         [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
#     print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)


########## Softmax Classification 복습 ################ 
# x_data = [[1, 2, 1],
#           [1, 3, 2],
#           [1, 3, 4],
#           [1, 5, 5],
#           [1, 7, 5],
#           [1, 2, 5],
#           [1, 6, 6],
#           [1, 7, 7]]
# y_data = [[0, 0, 1],
#           [0, 0, 1],
#           [0, 0, 1],
#           [0, 1, 0],
#           [0, 1, 0],
#           [0, 1, 0],
#           [1, 0, 0],
#           [1, 0, 0]]

# x_test = [[2, 1, 1],
#           [3, 1, 2],
#           [3, 3, 4]]
# y_test = [[0, 0, 1],
#           [0, 0, 1],
#           [0, 0, 1]]

# X = tf.placeholder("float", [None, 3])
# Y = tf.placeholder("float", [None, 3])

# W = tf.Variable(tf.random_normal([3, 3]))
# b = tf.Variable(tf.random_normal([3]))

# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(
#     learning_rate=1e-10).minimize(cost)

# prediction = tf.arg_max(hypothesis, 1)
# is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     for step in range(201):
#         cost_val, W_val, _ = sess.run(
#             [cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
#         print(step, cost_val, W_val)

#     print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
#     print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))