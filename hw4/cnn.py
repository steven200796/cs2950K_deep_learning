import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist_data/", one_hot=True)

batch_size = 50
train_steps = 2000
learning_rate = 1e-4
stddev = 0.1
kernel = [1,2,2,1]
padding = 'SAME'
#input and calculated label vectors
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])

#2d convolution
x_t = tf.reshape(x,[-1,28,28,1])

# conv layer 1
W_c1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=stddev))
b_c1 = tf.Variable(tf.zeros([32]))
out_c1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(x_t, W_c1, strides=[1,1,1,1], padding=padding) + b_c1), ksize=kernel, strides=kernel, padding=padding)

# conv layer 2
W_c2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=stddev))
b_c2 = tf.Variable(tf.zeros([64]))
out_c2 = tf.reshape(tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(out_c1, W_c2, strides=[1,1,1,1], padding=padding) + b_c2), ksize=kernel, strides=kernel, padding=padding), [-1,7*7*64])

# forward layer 1
W_f1 = tf.Variable(tf.truncated_normal([7*7*64,1024], stddev=stddev))
b_f1 = tf.Variable(tf.zeros([1024]))

# dropout
keep_prob = tf.placeholder(tf.float32)
out_f1 = tf.nn.dropout(tf.nn.relu(tf.matmul(out_c2, W_f1) + b_f1), keep_prob)

# forward layer 2
W_f2 = tf.Variable(tf.truncated_normal([1024,10], stddev=stddev))
b_f2 = tf.Variable(tf.zeros([10]))
y = tf.matmul(out_f1, W_f2) + b_f2

#actual label vector
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session() 
sess.run(init)

with sess.as_default():
    for i in xrange(train_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
