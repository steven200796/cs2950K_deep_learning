import tensorflow as tf
with open("../text_data/train.txt", 'r') as train, open("../text_data/test.txt", 'r') as test:
    
    batch_size = 20
    h_size = 100
    vocab_size = 50000
    embedding_size = 30
    learning_rate = 1e-4
    stddev = 0.1

    #input and calculated label vectors
    x = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    embedding = tf.nn.embedding_lookup(embeddings, x)

    # forward layer 1
    W_f1 = tf.Variable(tf.truncated_normal([embedding_size, h_size], stddev=stddev))
    b_f1 = tf.Variable(tf.zeros([h_size]))

    out_f1 = tf.nn.relu(tf.matmul(embedding, W_f1) + b_f1)

    W_f2 = tf.Variable(tf.truncated_normal([h_size,batch_size], stddev=stddev))
    b_f2 = tf.Variable(tf.zeros([batch_size]))
    
    out = tf.Variable(tf.matmul(out_f1,W_f2) + b_f2)


    #actual label vector
    y_ = tf.placeholder(tf.float32, [batch_size, 1])
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
