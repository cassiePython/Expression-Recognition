import tensorflow as tf
import os
import model
import data_provider

MODEL_SAVE_PATH = "path/to/model"
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
    
MODEL_NAME = "face.ckpt"
global_step = tf.Variable(0, trainable=False)

image, label = data_provider.get_data("train")
num_classes = 2
learning_rate = 0.001
batch_size = 30
TRAINING_ROUNDS = int(40000 / batch_size)

capacity = 1000 + 3 * batch_size

image_batch, label_batch = tf.train.batch(
    [image, label], batch_size=batch_size, capacity=capacity,
    allow_smaller_final_batch=True)

print (image_batch, label_batch)

x = tf.placeholder(tf.float32, [None, 64*64])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

logit = model.create_model(x, num_classes, keep_prob)
loss = model.create_loss(y, logit)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_loss = 0
    train_acc = 0
    for i in range(TRAINING_ROUNDS):
        #print (image_batch, label_batch)
        cur_image_batch, cur_label_batch = sess.run(
            [image_batch, label_batch])
        _, cost, acc = sess.run([train_step,loss,accuracy],
                                feed_dict={x: cur_image_batch, y: cur_label_batch, keep_prob: 1.})
        train_loss += cost
        train_acc += acc
        print ("After %d iterator(s): the train-loss is %f & train-acc is %f" %(i+1,
                                                                train_loss/(i+1), train_acc/(i+1)) ) 

        if (i+1) % 10 == 0:
            #print (output[0].shape)
            #print (output[0])
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    coord.request_stop()
    coord.join(threads)
   
