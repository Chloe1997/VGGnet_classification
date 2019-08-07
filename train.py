from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import time
import Data
import package.inference
import Data_overlap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # close the warning
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # use gpu 0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

model_path = "C:/Users/student1/Desktop/LDCT_CT/logistic_logs/save/model.ckpt"  # save path
# parameter
data_sample = 150000
total_batch = 500
batch_size = 300
epoch = 5000
display_step = 100
keep_prob = 0.5
global_step = tf.Variable(0, trainable=False)

# Shuffle coefficient for all data
allran_num = np.arange(0,data_sample)
np.random.shuffle(allran_num)
allInt = allran_num.astype(int)
Index= allInt.tolist()

traindata,trainlabel,testdata,testlabel = Data.Data()
# print(traindata.shape)
# print(testdata.shape)



# placeholder
input = tf.placeholder(tf.float32, [None, 32,32])
label = tf.placeholder(tf.float32, [None, 2])
is_training = tf.placeholder(tf.bool)

inputs = tf.reshape(input, shape=[-1, 32, 32, 1])
inputs = tf.convert_to_tensor(inputs,dtype=np.float32)


#train
logits,output = package.inference.inference(inputs,is_training=True)
loss = package.inference.loss(logits,label)
train_op = package.inference.train(loss,global_step)

# show prediction
classes = package.inference.classes(logits)
accuracy = package.inference.accuracy(logits,label)

# confusion_matrix
confusion_matrix = package.inference.confusion_matrix(label,logits)

# 初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# show result
best_test_accuracy = 0
count = 0
counter = 0
avg_cost = 0
i=0


with tf.Session(config=config) as sess:
    sess.run(init)
    for epoch in range(1,epoch+1):
        result_train = 0
        result_test = 0
        print("--------------Epoch" + str(epoch) + "----------------")
        traindata = traindata[Index]
        trainlabel = trainlabel[Index]

        for step in range(1, total_batch+1):
            feed_dict = {input: traindata[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                        label: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                        is_training: True}
            sess.run(train_op, feed_dict=feed_dict)
            # acc = sess.run(accuracy, feed_dict={
            #     input: traindata[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
            #     label: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
            #     is_training: False})
            # result_train = result_train + acc * (1 / total_batch)

            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={input: traindata[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                                    label: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
                                                    is_training: False})
                print("Step " + str(step) + ", Training Accuracy= " + "{:.4f}".format(acc))


        for x in range(15):
            result_train_1=sess.run(accuracy,
                                    feed_dict={input: traindata[x*10000:(x+1)*10000],
                                               label: trainlabel[x*10000:(x+1)*10000],
                                               is_training: False})
            result_train = result_train + result_train_1 * (1 / 15)

        result_test = sess.run(accuracy,
                               feed_dict={input: testdata,
                                          label: testlabel,
                                          is_training: False})
        test_class = sess.run(classes,feed_dict={input:testdata,
                                                 label:testlabel,
                                                 is_training:False})


        # early stop
        count = count + 1
        counter = counter + 1
        if best_test_accuracy <= result_test and result_train - result_test < 0.1:
            best_test_accuracy = result_test
            print("***** best test  accuracy :" + "{:.4f}".format(best_test_accuracy) + " ****")
            print("***** best train accuracy :" + "{:.4f}".format(result_train) + " ****")
            count = 0
            save_path = saver.save(sess, model_path, write_meta_graph=False)
        else:
            print("Testing  accuracy=" + "{:.4f}".format(result_test))
            print("Training accuracy=" + "{:.4f}".format(result_train))

        if count == 20:
            break
        if result_train - result_test < 0.1:
            counter = 0
        if counter == 20:
            break

    # confusion matrix
    Sensitivity_train = 0
    Specificity_train = 0
    TP=0
    FN=0
    TN=0
    FP=0
    for x in range(15):
        confusion_matrix1 = sess.run(confusion_matrix,
                                        feed_dict={input: traindata[x * 10000:(x + 1) * 10000],
                                        label: trainlabel[x * 10000:(x + 1) * 10000],
                                        is_training: False})
        TP_train = confusion_matrix1[0][0]
        FN_train = confusion_matrix1[0][1]
        TN_train = confusion_matrix1[1][1]
        FP_train = confusion_matrix1[1][0]
        N_train = TP_train + TN_train + FP_train + FN_train
        TP = TP +TP_train
        FN = FN +FN_train
        TN = TN +TN_train
        FP = FP +FP_train
        Accuracy_train = round((TP_train + TN_train) / N_train, 3)
        Sensitivity_train = round(TP_train / (TP_train + FP_train), 3)+Sensitivity_train
        Specificity_train = round(TN_train / (TN_train + FP_train), 3)+Specificity_train
        PPV_train = round(TP_train / (TP_train + FP_train), 3)
        NPV_train = round(TN_train / (TN_train + FN_train), 3)
        pa = (TP_train + TN_train) / N_train
        pe = ((TP_train + FP_train) * (TP_train + FN_train) + (FN_train + TN_train) * (FP_train + TN_train)) / (N_train * N_train)
        Kappa_index_train = round((pa - pe) / (1 - pe), 3)

    print(Sensitivity_train/15, Specificity_train/15)
    print(TP,FN,TN,FP)




# Restore weights & biases variable
print("----------------------Restore------------------------------")
with tf.Session() as sess:

    saver.restore(sess, model_path)
    result_test = 0
    result_train = 0

    # for step in range(1, total_batch + 1):
        # acc = sess.run(accuracy, feed_dict={
        #     input: traindata[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
        #     label: trainlabel[0 + (batch_size) * (step - 1):(batch_size) + (batch_size) * (step - 1)],
        #     is_training: False})
        # result_train = result_train + acc * (1 / total_batch)

    for x in range(15):
        result_train_1 = sess.run(accuracy,
                                  feed_dict={input: traindata[x * 10000:(x + 1) * 10000],
                                             label: trainlabel[x * 10000:(x + 1) * 10000],
                                             is_training: False})
        result_train = result_train + result_train_1 * (1 / 15)

    result_test = sess.run(accuracy,
                           feed_dict={input: testdata,
                                      label: testlabel,
                                      is_training: False})
    print("Best testing  accuracy=" + "{:.4f}".format(result_test))
    print("Best training accuracy=" + "{:.4f}".format(result_train))










