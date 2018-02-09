# Load pickled data
import pickle
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

dataset_path="./dataset/"
training_file = dataset_path+"train2.p"
validation_file = dataset_path+"valid2.p"
testing_file = dataset_path+"test2.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


# pickle.dump(train, open(dataset_path+"train2.p","wb"), protocol=2)
# pickle.dump(valid, open(dataset_path+"valid2.p","wb"), protocol=2)
# pickle.dump(test, open(dataset_path+"test2.p","wb"), protocol=2)


X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y_train.reshape([-1,1]))
ohe.n_values_, ohe.feature_indices_, ohe.active_features_
y_train_oh = ohe.transform(y_train.reshape([-1,1])).toarray()
y_valid_oh = ohe.transform(y_valid.reshape([-1,1])).toarray()
y_test_oh = ohe.transform(y_test.reshape([-1,1])).toarray()

y_train_oh = np.squeeze(np.asarray(y_train_oh))
#y_train_oh = np.asarray(y_train_oh)
print(y_train_oh)
print(y_train_oh.shape)


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.asarray(X_train[0]).shape

# TODO: How many unique classes/labels there are in the dataset.
n_labels = []
def get_num_classes():
    for _ in train['labels']:
        if _ not in n_labels:
            n_labels.append(_)
    return len(n_labels)
n_classes = get_num_classes()
print("labels :", n_labels)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Visualizations will be shown in the notebook.

# I will draw figures of one for each class, i.e. 'n_classes' figures.
figure_idx=np.empty([n_classes], dtype=np.int)
for i in range(n_classes):
    c = n_labels[i]
    for l in range(n_train):
        if y_train[l]==c:
            figure_idx[i]=l
            break
print (figure_idx)

cols = 10
rows = int(n_classes/cols)+1
print(cols, rows)
fig, ax = plt.subplots(rows, cols, figsize=(rows, cols))#, dpi=1200
plt.tight_layout()
for i in range(rows):
    for k in range(cols):
        ax[i][k].set_axis_off()
        if (i*cols+k)<n_classes:
            ax[i][k].imshow(X_train[figure_idx[i*cols+k]])



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# calculate mean w.r.t. channels (R,G,B)
mean = np.mean(X_train, axis=(0,1,2))
#print(mean)

# Preprocess --> normalize image pixel value between -1.0~1.0
def img_preprocess(img):
    return (img-mean)/128.
X_train = img_preprocess(X_train)
X_valid = img_preprocess(X_valid)
X_test = img_preprocess(X_test)
#print(X_train[0])

def img_restore(img):
    return int(img*128.+mean)


from sklearn.utils import shuffle
def shuffle_data(x, y):
    return shuffle(x, y)
    #return x, y

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
print(tf.__version__)


# I will use simple CNN 

# define default convolution layer with batch_norm and activation
def conv(name, input_layer, input_depth, out_depth, kernel_size=3, strides=[1, 2, 2, 1], padding="SAME", activation='relu'):
    F_W = tf.get_variable('weight_'+name, [kernel_size, kernel_size, input_depth, out_depth], initializer=tf.truncated_normal_initializer())
    F_b = tf.get_variable('offset'+name, [out_depth], initializer=tf.constant_initializer(0.0))
    _layer = tf.nn.conv2d(input_layer, F_W, strides, padding, name=name) + F_b
    if activation=='relu':
        _layer = tf.nn.relu(_layer)
    elif activation=='lrelu':
        _layer = tf.nn.leaky_relu(_layer)
    return _layer


X = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], image_shape[2]), name='X')
y = tf.placeholder(tf.float32, (None, n_classes), name='y')
dropout = tf.placeholder(tf.float32, name='dropout')
lr = tf.placeholder(tf.float32, name='learning_rate')


# base_depth = 32
# conv_1 = conv("conv_1", X, 3,            base_depth,   strides=[1,1,1,1])
# conv_2 = conv("conv_2", conv_1, base_depth,   base_depth*2, strides=[1,1,1,1])
# conv_3 = conv("conv_3", conv_2, base_depth*2, base_depth*4, strides=[1,1,1,1])
# conv_4 = conv("conv_4", conv_3, base_depth*4, base_depth*8, strides=[1,1,1,1])
# conv_5 = conv("conv_5", conv_4, base_depth*8, base_depth*16,strides=[1,2,2,1])#16x16
# conv_6 = conv("conv_6", conv_5, base_depth*16,base_depth*32,strides=[1,2,2,1])#8x8
# conv_7 = conv("conv_7", conv_6, base_depth*32,base_depth*32,strides=[1,2,2,1])#4x4

# base_depth = 32
# conv_1 = conv("conv_1", X, 3,            base_depth,   strides=[1,1,1,1])
# conv_2 = conv("conv_2", conv_1, base_depth,   base_depth*2, strides=[1,1,1,1])
# conv_5 = conv("conv_5", conv_2, base_depth*2, base_depth*4,strides=[1,2,2,1])#16x16
# conv_6 = conv("conv_6", conv_5, base_depth*4,base_depth*8,strides=[1,2,2,1])#8x8
# conv_7 = conv("conv_7", conv_6, base_depth*8,base_depth*16,strides=[1,2,2,1])#4x4

# fc0 = tf.contrib.layers.flatten(conv_7)
# fc1_w = tf.Variable(tf.truncated_normal([4*4*base_depth*16, n_classes]))
# fc1_b = tf.Variable(tf.zeros(n_classes))
# logits = tf.matmul(fc0, fc1_w) + fc1_b


# base_depth = 64
# conv_1_weight   = tf.Variable(tf.truncated_normal([filter_size, filter_size, 3, base_depth], mean=0.0, stddev=0.1))
# conv_1_bias     = tf.Variable(tf.zeros(base_depth))
# conv_1          = tf.nn.conv2d(X, conv_1_weight, [1,1,1,1], padding="SAME", name='conv_1') + conv_1_bias
# conv_1_relu     = tf.nn.relu(conv_1)
# conv_1_pool     = tf.nn.max_pool(conv_1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# conv_2_weight   = tf.Variable(tf.truncated_normal([filter_size, filter_size, base_depth, base_depth*2], mean=0.0, stddev=0.1))
# conv_2_bias     = tf.Variable(tf.zeros(base_depth*2))
# conv_2          = tf.nn.conv2d(conv_1_pool, conv_2_weight, [1,1,1,1], padding="SAME", name='conv_2') + conv_2_bias
# conv_2_relu     = tf.nn.relu(conv_2)
# conv_2_pool     = tf.nn.max_pool(conv_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# conv_3_weight   = tf.Variable(tf.truncated_normal([filter_size, filter_size, base_depth*2, base_depth*4], mean=0.0, stddev=0.1))
# conv_3_bias     = tf.Variable(tf.zeros(base_depth*4))
# conv_3          = tf.nn.conv2d(conv_2_pool, conv_3_weight, [1,1,1,1], padding="SAME", name='conv_3') + conv_3_bias
# conv_3_relu     = tf.nn.relu(conv_3)
# conv_3_pool     = tf.nn.max_pool(conv_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# fc0 = tf.contrib.layers.flatten(conv_3_pool)#flatten
# fc1_w = tf.Variable(tf.truncated_normal([4*4*base_depth*4, n_classes], mean=0.0, stddev=0.1))
# fc1_b = tf.Variable(tf.zeros(n_classes))
# logits = tf.matmul(fc0, fc1_w) + fc1_b

'''
base_depth = 16
conv_1_weight   = tf.Variable(tf.truncated_normal([3, 3, 3, base_depth], mean=0.0, stddev=0.1))
conv_1_bias     = tf.Variable(tf.zeros(base_depth))
conv_1_1        = tf.nn.conv2d(X, conv_1_weight, [1,1,1,1], padding="SAME", name='conv_1') + conv_1_bias

conv_1_2_weight = tf.Variable(tf.truncated_normal([5,5,3, base_depth], mean=0.0, stddev=0.1))
conv_1_2_bias   = tf.Variable(tf.zeros(base_depth))
conv_1_2        = tf.nn.conv2d(X, conv_1_2_weight, [1,1,1,1], padding="SAME", name='conv_1_2') + conv_1_2_bias

conv_1_3_weight = tf.Variable(tf.truncated_normal([7,7,3, base_depth], mean=0.0, stddev=0.1))
conv_1_3_bias   = tf.Variable(tf.zeros(base_depth))
conv_1_3        = tf.nn.conv2d(X, conv_1_3_weight, [1,1,1,1], padding="SAME", name='conv_1_3') + conv_1_3_bias

conv_1          = tf.concat([conv_1_1,conv_1_2,conv_1_3],-1)

conv_1_relu     = tf.nn.relu(conv_1)
conv_1_pool     = tf.nn.max_pool(conv_1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



conv_2_weight   = tf.Variable(tf.truncated_normal([3, 3, base_depth*3, base_depth*4], mean=0.0, stddev=0.1))
conv_2_bias     = tf.Variable(tf.zeros(base_depth*4))
conv_2          = tf.nn.conv2d(conv_1_pool, conv_2_weight, [1,1,1,1], padding="SAME", name='conv_2') + conv_2_bias
conv_2_relu     = tf.nn.relu(conv_2)
conv_2_pool     = tf.nn.max_pool(conv_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv_3_weight   = tf.Variable(tf.truncated_normal([3, 3, base_depth*4, base_depth*8], mean=0.0, stddev=0.1))
conv_3_bias     = tf.Variable(tf.zeros(base_depth*8))
conv_3          = tf.nn.conv2d(conv_2_pool, conv_3_weight, [1,1,1,1], padding="SAME", name='conv_3') + conv_3_bias
conv_3_relu     = tf.nn.relu(conv_3)
conv_3_pool     = tf.nn.max_pool(conv_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
fc0     = tf.contrib.layers.flatten(conv_3_pool)#flatten
fc0_dropout = tf.nn.dropout(fc0, dropout)

fc1_w   = tf.Variable(tf.truncated_normal([4*4*base_depth*8, 512], mean=0.0, stddev=0.1))
fc1_b   = tf.Variable(tf.zeros(512))
fc1     = tf.matmul(fc0, fc1_w) + fc1_b

fc2_w = tf.Variable(tf.truncated_normal([512, 256], mean=0.0, stddev=0.1))
fc2_b = tf.Variable(tf.zeros(256))
fc2   = tf.matmul(fc1, fc2_w) + fc2_b

fc3_w = tf.Variable(tf.truncated_normal([256, n_classes], mean=0.0, stddev=0.1))
fc3_b = tf.Variable(tf.zeros(n_classes))
logits = tf.matmul(fc2, fc3_w) + fc3_b
'''
'''
base_depth = 16
conv_1_weight   = tf.Variable(tf.truncated_normal([3, 3, 3, base_depth], mean=0.0, stddev=0.1))
conv_1_bias     = tf.Variable(tf.zeros(base_depth))
conv_1_1        = tf.nn.conv2d(X, conv_1_weight, [1,1,1,1], padding="SAME", name='conv_1') + conv_1_bias
conv_1_relu     = tf.nn.relu(conv_1_1)

conv_1_2_weight = tf.Variable(tf.truncated_normal([3, 3, base_depth, base_depth*2], mean=0.0, stddev=0.1))
conv_1_2_bias   = tf.Variable(tf.zeros(base_depth*2))
conv_1_2        = tf.nn.conv2d(conv_1_relu, conv_1_2_weight, [1,1,1,1], padding="SAME", name='conv_1_2') + conv_1_2_bias
conv_1_2_relu   = tf.nn.relu(conv_1_2)

conv_1_3_weight = tf.Variable(tf.truncated_normal([3, 3, base_depth*2, base_depth*4], mean=0.0, stddev=0.1))
conv_1_3_bias   = tf.Variable(tf.zeros(base_depth*4))
conv_1_3        = tf.nn.conv2d(conv_1_2_relu, conv_1_3_weight, [1,1,1,1], padding="SAME", name='conv_1_3') + conv_1_3_bias
conv_1_3_relu   = tf.nn.relu(conv_1_3)

conv_1_3_pool   = tf.nn.max_pool(conv_1_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


conv_2_weight   = tf.Variable(tf.truncated_normal([3, 3, base_depth*4, base_depth*8], mean=0.0, stddev=0.1))
conv_2_bias     = tf.Variable(tf.zeros(base_depth*8))
conv_2          = tf.nn.conv2d(conv_1_3_pool, conv_2_weight, [1,1,1,1], padding="SAME", name='conv_2') + conv_2_bias
conv_2_relu     = tf.nn.relu(conv_2)
conv_2_pool     = tf.nn.max_pool(conv_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv_3_weight   = tf.Variable(tf.truncated_normal([3, 3, base_depth*8, base_depth*16], mean=0.0, stddev=0.1))
conv_3_bias     = tf.Variable(tf.zeros(base_depth*16))
conv_3          = tf.nn.conv2d(conv_2_pool, conv_3_weight, [1,1,1,1], padding="SAME", name='conv_3') + conv_3_bias
conv_3_relu     = tf.nn.relu(conv_3)
conv_3_pool     = tf.nn.max_pool(conv_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


fc0     = tf.contrib.layers.flatten(conv_3_pool)#flatten
fc0_dropout = tf.nn.dropout(fc0, dropout)

fc1_w   = tf.Variable(tf.truncated_normal([4*4*base_depth*16, 1024], mean=0.0, stddev=0.1))
fc1_b   = tf.Variable(tf.zeros(1024))
fc1     = tf.matmul(fc0, fc1_w) + fc1_b

fc2_w = tf.Variable(tf.truncated_normal([1024, 256], mean=0.0, stddev=0.1))
fc2_b = tf.Variable(tf.zeros(256))
fc2   = tf.matmul(fc1, fc2_w) + fc2_b

fc3_w = tf.Variable(tf.truncated_normal([256, n_classes], mean=0.0, stddev=0.1))
fc3_b = tf.Variable(tf.zeros(n_classes))
logits = tf.matmul(fc2, fc3_w) + fc3_b
'''

base_depth = 8
conv_1_weight   = tf.Variable(tf.truncated_normal([3, 3, 3, base_depth], mean=0.0, stddev=0.1))
conv_1_bias     = tf.Variable(tf.zeros(base_depth))
conv_1_1        = tf.nn.conv2d(X, conv_1_weight, [1,1,1,1], padding="SAME", name='conv_1') + conv_1_bias
conv_1_relu     = tf.nn.relu(conv_1_1)

conv_1_2_weight = tf.Variable(tf.truncated_normal([3, 3, base_depth, base_depth*2], mean=0.0, stddev=0.1))
conv_1_2_bias   = tf.Variable(tf.zeros(base_depth*2))
conv_1_2        = tf.nn.conv2d(conv_1_relu, conv_1_2_weight, [1,1,1,1], padding="SAME", name='conv_1_2') + conv_1_2_bias
conv_1_2_relu   = tf.nn.relu(conv_1_2)

conv_1_3_weight = tf.Variable(tf.truncated_normal([3, 3, base_depth*2, base_depth*4], mean=0.0, stddev=0.1))
conv_1_3_bias   = tf.Variable(tf.zeros(base_depth*4))
conv_1_3        = tf.nn.conv2d(conv_1_2_relu, conv_1_3_weight, [1,1,1,1], padding="SAME", name='conv_1_3') + conv_1_3_bias
conv_1_3_relu   = tf.nn.relu(conv_1_3)

conv_1_3_pool   = tf.nn.max_pool(conv_1_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv_2_weight   = tf.Variable(tf.truncated_normal([5, 5, base_depth*4, base_depth*8], mean=0.0, stddev=0.1))
conv_2_bias     = tf.Variable(tf.zeros(base_depth*8))
conv_2          = tf.nn.conv2d(conv_1_3_pool, conv_2_weight, [1,1,1,1], padding="VALID", name='conv_2') + conv_2_bias
conv_2_relu     = tf.nn.relu(conv_2)
#conv_2_pool     = tf.nn.max_pool(conv_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

conv_3_weight   = tf.Variable(tf.truncated_normal([5, 5, base_depth*8, base_depth*16], mean=0.0, stddev=0.1))
conv_3_bias     = tf.Variable(tf.zeros(base_depth*16))
conv_3          = tf.nn.conv2d(conv_2_relu, conv_3_weight, [1,1,1,1], padding="VALID", name='conv_3') + conv_3_bias
conv_3_relu     = tf.nn.relu(conv_3)
#conv_3_pool     = tf.nn.max_pool(conv_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

fc0     = tf.contrib.layers.flatten(conv_3_relu)
fc0_dropout = tf.nn.dropout(fc0, dropout)

fc1_w   = tf.Variable(tf.truncated_normal([8*8*base_depth*16, 1024], mean=0.0, stddev=0.1))
fc1_b   = tf.Variable(tf.zeros(1024))
fc1     = tf.matmul(fc0, fc1_w) + fc1_b

fc2_w = tf.Variable(tf.truncated_normal([1024, 256], mean=0.0, stddev=0.1))
fc2_b = tf.Variable(tf.zeros(256))
fc2   = tf.matmul(fc1, fc2_w) + fc2_b

fc3_w = tf.Variable(tf.truncated_normal([256, n_classes], mean=0.0, stddev=0.1))
fc3_b = tf.Variable(tf.zeros(n_classes))
logits = tf.matmul(fc2, fc3_w) + fc3_b



# Cross entropy
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)

#loss
regular = tf.nn.l2_loss(conv_1_weight) + tf.nn.l2_loss(conv_1_2_weight) + tf.nn.l2_loss(conv_1_3_weight) + \
          tf.nn.l2_loss(conv_2_weight) + tf.nn.l2_loss(conv_3_weight) + tf.nn.l2_loss(fc1_w) + \
          tf.nn.l2_loss(fc2_w) + tf.nn.l2_loss(fc3_w)
loss = tf.reduce_mean(cross_entropy) + 0.0001*regular

train_step = tf.train.AdamOptimizer(lr).minimize(loss)    

epoch = 35
batch_size = 32
num_batch = int(n_train/batch_size)
#print(num_batch)

def eval(sess, _X, _y):
    prediction = tf.nn.softmax(logits)
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    eval_size = len(_X)
    acc = 0.
    num_data=0
    for i in range(0, eval_size, batch_size):
        start = i
        end = start+batch_size
        X_this_batch = _X[start:end]
        y_this_batch = _y[start:end]
        this_acc = sess.run(accuracy, feed_dict={X: X_this_batch, y: y_this_batch, dropout:1.0})
        acc += (this_acc * len(X_this_batch))
        #num_data+=len(X_this_batch)
    #print( eval_size, num_data )
    return acc / eval_size

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
base_lr = 0.0005
for e in range(epoch):
    this_lr = base_lr
    if e>40:
        this_lr = this_lr*0.1
    if e>70:
        this_lr = this_lr*0.1

    shuffled_X, shuffled_y = shuffle_data(X_train, y_train_oh)
    for i in range(num_batch):
        start = i*batch_size
        end = start+batch_size
        X_this_batch = shuffled_X[start:end]
        y_this_batch = shuffled_y[start:end]
        # _, this_loss = sess.run([train_step, loss], feed_dict={X:X_this_batch, y:y_this_batch})
        sess.run(train_step, feed_dict={X:X_this_batch, y:y_this_batch, dropout:0.8, lr:this_lr})
    this_loss, reg = sess.run([loss,regular], feed_dict={X:X_this_batch, y:y_this_batch, dropout:1.0})
    print("epoch : {}".format(e), "batch: {}".format(i), "Loss : {}".format(this_loss), "L2 Reg : {}".format(reg))
    val_accuracy = eval(sess, X_valid, y_valid_oh)
    print("validation accuracy : {}".format(val_accuracy))
test_accuracy =eval(sess, X_test, y_test_oh)
print("test accuracy : {}".format(test_accuracy))
train_accuracy =eval(sess, X_train, y_train_oh)
print("train accuracy : {}".format(train_accuracy))
saver.save(sess, 'train_data')    

#session.run(loss, feed_dict=valid_feed_dict)
#session.run(loss, feed_dict=test_feed_dict)
#biases_data = session.run(biases)