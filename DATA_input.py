import os
import h5py
import numpy as np


def CT_seg ():
    os.chdir('C:/Users/student1/Desktop/test_data/seg_image/P')
    path = 'C:/Users/student1/Desktop/test_data/seg_image/P'
    files = os.listdir(path)
    num = 0
    for file in files:
        # print(num)
        num = num + 1
        # 這裡的矩陣是matlab打開時矩陣的轉置
        CT_image = h5py.File(file)
        #轉置回來
        CT_image = np.transpose(CT_image['CT_seg'])
        CT_image = np.reshape(CT_image, [-1, 32, 32])
        if num == 1 :
             CT = CT_image
        else:
             CT = np.vstack((CT,CT_image))
    return CT

# CT = CT_seg()
# print(CT.shape)

def LDCT_seg ():
    os.chdir('C:/Users/student1/Desktop/test_data/seg_image/L')
    path = 'C:/Users/student1/Desktop/test_data/seg_image/L'
    files = os.listdir(path)
    num = 0
    for file in files:
        num = num+1
        # 這裡的矩陣是matlab打開時矩陣的轉置
        LDCT_image = h5py.File(file)
        #轉置回來
        LDCT_image = np.transpose(LDCT_image['LDCT_seg'])
        LDCT_image = np.reshape(LDCT_image, [-1, 32, 32])
        if num == 1 :
             LDCT = LDCT_image
        else:
             LDCT = np.vstack((LDCT,LDCT_image))
    return LDCT

# LDCT = LDCT_seg()
# print(LDCT.shape)

def CT_overlap ():
    os.chdir('C:/Users/student1/Desktop/test_data/seg_image_overlap/P')
    path = 'C:/Users/student1/Desktop/test_data/seg_image_overlap/P'
    files = os.listdir(path)
    num = 0
    for file in files:
        num+=1
        # 這裡的矩陣是matlab打開時矩陣的轉置
        CT_image = h5py.File(file)
        #轉置回來
        CT_image = np.transpose(CT_image['CT_seg_overlap'])
        CT_image = np.reshape(CT_image, [-1, 32, 32])
        if num == 1 :
             CT = CT_image
        else:
             CT = np.vstack((CT,CT_image))
    return CT
# CT = CT_overlap()
# print(CT.shape)

def LDCT_overlap ():
    os.chdir('C:/Users/student1/Desktop/test_data/seg_image_overlap/L')
    path = 'C:/Users/student1/Desktop/test_data/seg_image_overlap/L'
    files = os.listdir(path)
    num = 0
    for file in files:
        num+=1
        # 這裡的矩陣是matlab打開時矩陣的轉置
        LDCT_image = h5py.File(file)
        #轉置回來
        LDCT_image = np.transpose(LDCT_image['LDCT_seg_overlap'])
        LDCT_image = np.reshape(LDCT_image, [-1, 32, 32])
        if num == 1 :
             LDCT = LDCT_image
        else:
             LDCT = np.vstack((LDCT,LDCT_image))
    return LDCT
# LDCT = LDCT_overlap()
# print(LDCT.shape)

def Data_labels ():
    # zero_1 = np.zeros([138012,1])
    # one_1 = np.ones([138012,1])
    # zero_2 = np.zeros([68009,1])
    # one_2 = np.ones([68009,1])
    #
    # LDCT_labels = np.dstack((zero_1,one_1))
    # LDCT_labels = np.reshape(LDCT_labels,[138012,2])
    # CT_labels = np.dstack((one_2,zero_2))
    # CT_labels = np.reshape(CT_labels,[68009,2])

    LDCT_labels = []
    for i in range(138012):
        label = ([0,1])
        LDCT_labels.append(label)
    LDCT_labels = np.concatenate(LDCT_labels)
    LDCT_labels = LDCT_labels.reshape([138012,2])

    CT_labels = []
    for j in range(68009):
        label2 = [1,0]
        CT_labels.append(label2)
    CT_labels = np.concatenate(CT_labels)
    CT_labels = CT_labels.reshape([68009, 2])

    return LDCT_labels,CT_labels

# LDCT_labels,CT_labels = Data_labels()
# print(LDCT_labels)
# print(CT_labels)

def Data_labels_overlap ():
    LDCT_labels = []
    for i in range(2012869):
        label = [0,1]
        LDCT_labels.append(label)
    LDCT_labels = np.concatenate(LDCT_labels)
    LDCT_labels = LDCT_labels.reshape([2012869,2])

    CT_labels = []
    for j in range(1014007):
        label2 = [1,0]
        CT_labels.append(label2)
    CT_labels = np.concatenate(CT_labels)
    CT_labels = CT_labels.reshape([1014007, 2])

    return LDCT_labels,CT_labels

