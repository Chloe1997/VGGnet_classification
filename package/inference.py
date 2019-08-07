import tensorflow as tf
import tensorflow.contrib.slim as slim

#parameter
num_classes = 2
total_batch = 500
#CNN structure
def inference(input,is_training):

    weight_decay = 0.0005
    keep_prob = 0.5
    ##batch normalization 參數定義
    batch_norm_decay = 0.996
    batch_norm_epsilon = 1e-5
    batch_norm_scale = True
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }
    ## CNN 架構
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.dropout], keep_prob=keep_prob, is_training=is_training):
            with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=[2, 2]):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
                    net = slim.repeat(input, 2, slim.conv2d, 4, [3, 3])  # conv1
                    net = slim.max_pool2d(net)  # pool1
                    net = slim.repeat(net, 2, slim.conv2d, 8, [3, 3])  # conv2
                    net = slim.max_pool2d(net)  # pool2
                    net = slim.conv2d(net, 16, [3, 3])  # conv3
                    net = slim.max_pool2d(net)  # pool3

                    flatten = slim.flatten(net)
                    net = slim.fully_connected(flatten, 1024)
                    net = slim.dropout(net)
                    output = slim.fully_connected(net, num_classes, activation_fn=None)
                    logits = tf.nn.softmax(output)

    return logits,output

def inception(input,is_training):
    weight_decay = 0.0005
    keep_prob = 0.5
    ##batch normalization 參數定義
    batch_norm_decay = 0.996
    batch_norm_epsilon = 1e-5
    batch_norm_scale = True
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
    }
    ## CNN 架構
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.dropout], keep_prob=keep_prob, is_training=is_training):
            with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=[2, 2]):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
                    net = slim.conv2d(input, 4, [3, 3])
                    net = slim.conv2d(net, 8, [3, 3])
                    net = slim.conv2d(net, 16, [3, 3])
                    net = slim.max_pool2d(net)
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                with tf.variable_scope('Mixed_1'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 8, [1, 1])
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 16, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 32, [3, 3])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 8, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 16, [5, 5])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net,[2,2],stride=[1,1],padding='SAME')
                        branch_3 = slim.conv2d(branch_3, 16, [1, 1])
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                with tf.variable_scope('Mixed_2'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 16, [1, 1])
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 32, [1, 1])
                        branch_1 = slim.conv2d(branch_1, 64, [3, 3])
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 32, [1, 1])
                        branch_2 = slim.conv2d(branch_2, 64, [5, 5])
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.max_pool2d(net,[2,2],stride=[1,1],padding='SAME')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1])
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)
                net = slim.conv2d(net,2,[1,1],activation_fn=None)
                net = slim.avg_pool2d(net,kernel_size=[net.shape[1], net.shape[2]],stride=[1,1],padding='VALID')
                net = tf.reshape(net,[-1,2])
                logits = tf.nn.softmax(net)

                return logits
def loss(logits,labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
    return loss

def confusion_matrix(label,output):
    confusion_matrix = tf.confusion_matrix(tf.argmax(label,1),tf.argmax(output,1), num_classes=2,dtype=tf.int32,name='output',weights=None)
    return confusion_matrix

def train(loss,global_step):
    # learning rate parameters
    init_learning_rate = 0.05
    decay_rate = 0.9

    learning_rate = tf.train.exponential_decay(init_learning_rate ,global_step , total_batch ,decay_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op

def accuracy(logits , labels):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy

def classes(logits):
    classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)

    return classes