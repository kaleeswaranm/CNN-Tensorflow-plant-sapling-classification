import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd

species = sorted(os.listdir('/home/kaleeswaran/Desktop/Sapling/all (1)/train'))

image_path = [os.listdir('/home/kaleeswaran/Desktop/Sapling/all (1)/train/' + s) for s in species]

def create_HSV_masked_images(img):
    timg = cv2.imread(img)
    timg = cv2.cvtColor(timg, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(timg, np.array([60-35,100,50]),np.array([60+35,255,255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    timg = cv2.bitwise_and(timg, timg, mask = mask)
    timg = cv2.resize(timg, (64, 64))
    return(timg)

images = [create_HSV_masked_images('/home/kaleeswaran/Desktop/Sapling/all (1)/train/'+species[i]+'/'+image_path[i][j]) for i in range(len(species))
                                                                                                                       for j in range(len(image_path[i]))]

rimages = [x.reshape((1,64,64,3)) for x in images]
npimages = np.concatenate(rimages, axis=0)
npimages = npimages / 255

pickle.dump(npimages , open('/home/kaleeswaran/Desktop/Sapling/all (1)/imagePickle.p', 'wb'))

npimages = pickle.load(open('/home/kaleeswaran/Desktop/Sapling/all (1)/imagePickle.p', 'rb'))

len_path = [len(os.listdir('/home/kaleeswaran/Desktop/Sapling/all (1)/train/' + s)) for s in species]

l = range(12)
y_labels = np.repeat(np.array(l), np.array(len_path))
ohv = np.zeros((4750, 12))
ohv[np.arange(4750), y_labels] = 1

idxes = np.random.permutation(4750)
simages = npimages[idxes]
sylabels = ohv[idxes]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(simages, sylabels, test_size=0.10, random_state=42)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None,64,64,3])
Y = tf.placeholder(tf.float32, shape=[None,12])

def convlayer(con, shpe, num):
    w = tf.get_variable('w'+str(num), shape = shpe, initializer = tf.contrib.layers.xavier_initializer(seed=0))
    b = tf.zeros(shpe[-1])
    conv = tf.nn.conv2d(con, w, [1,1,1,1], padding="SAME")
    conv = tf.nn.bias_add(conv, b)
    return(tf.nn.relu(conv))

conv1 = convlayer(X, [3,3,3,16], 1)
conv2 = convlayer(conv1, [3,3,16,16], 2)
maxp1 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='SAME')

conv3 = convlayer(maxp1, [3,3,16,32], 3)
conv4 = convlayer(conv3, [3,3,32,32], 4)
maxp2 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding='SAME')

bn2   = tf.layers.batch_normalization(maxp2)

conv5 = convlayer(bn2, [3,3,32,64], 5)
conv6 = convlayer(conv5, [3,3,64,64], 6)
maxp3 = tf.nn.max_pool(conv6, [1,2,2,1], [1,2,2,1], padding='SAME')

fc    = tf.contrib.layers.flatten(maxp3)
fc1   = tf.contrib.layers.fully_connected(fc, 4096, activation_fn=None)
bn3   = tf.layers.batch_normalization(fc1)
Fc1   = tf.nn.relu(bn3)
kp    = tf.placeholder(tf.float32)
do    = tf.nn.dropout(Fc1, keep_prob=kp)
fc2   = tf.contrib.layers.fully_connected(do, 1000)
fc3   = tf.contrib.layers.fully_connected(fc2, 12, activation_fn=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)

correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

at = tf.summary.scalar('accuracy_train', accuracy)
ct = tf.summary.scalar('cost_train', cost)
ate = tf.summary.scalar('accuracy_test', accuracy)
cte = tf.summary.scalar('cost_test', cost)
merge_train = tf.summary.merge([at, ct])
merge_test = tf.summary.merge([ate, cte])

init = tf.global_variables_initializer()
saver = tf.train.Saver()

trco = []
teco = []
trac = []
teac = []
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("/home/kaleeswaran/Desktop/Sapling/all (1)/output",sess.graph)
    Writer = tf.summary.FileWriter("/home/kaleeswaran/Desktop/Sapling/all (1)/output1",sess.graph)
    for epoch in range(8):
        for batch in range(int(4275/32)):
            _, trcos, tracc, summary = sess.run([optimizer, cost, accuracy, merge_train], feed_dict = {X: X_train[batch * 32:(batch + 1) * 32], Y: y_train[batch * 32:(batch + 1) * 32], kp: 0.75})
            print('epoch: ' + str(epoch) + ' batch: ' + str(batch))
            print('train cost: ' + str(trcos))
            print('train accuracy: ' + str(tracc))
            writer.add_summary(summary, epoch * 133 + batch)
        trco.append(trcos)
        trac.append(tracc)
        tecos, teacc, summary1 = sess.run([cost, accuracy, merge_test], feed_dict = {X: X_test, Y: y_test, kp: 1.0})
        Writer.add_summary(summary1, epoch)
        saver.save(sess,'/home/kaleeswaran/Desktop/Sapling/all (1)/chkpnt/model.ckpt')
        teco.append(tecos)
        teac.append(teacc)
        print('test cost: ' + str(tecos))
        print('test accuracy: ' + str(teacc))
    writer.close()
    Writer.close()
        
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(npimages[400+i])









