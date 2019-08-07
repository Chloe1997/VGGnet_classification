import numpy as np
import DATA_input

def Data():
    # Shuffle coefficient for all data
    allran_num = np.arange(0, 68009)
    np.random.shuffle(allran_num)
    allInt = allran_num.astype(int)
    Index = allInt.tolist()
    # Shuffle coefficient for all data
    allran_num = np.arange(0, 138012)
    np.random.shuffle(allran_num)
    allInt = allran_num.astype(int)
    Indexx = allInt.tolist()

    #load data from DATA_input
    LDCT = DATA_input.LDCT_seg()
    LDCT = LDCT[Indexx]
    CT = DATA_input.CT_seg()
    CT = CT[Index]
    LDCT_labels,CT_labels = DATA_input.Data_labels()

    #seperate Train & Test data
    Train_LDCT = LDCT[0:100000,:,:]
    Train_LDCT_labels = LDCT_labels[0:100000,:]
    Test_LDCT = LDCT[100000:,:,:]
    Test_LDCT_labels = LDCT_labels[100000:,:]
    # Test_LDCT = LDCT[1:1000, :, :]
    # Test_LDCT_labels = LDCT_labels[1:1000, :]



    Train_CT = CT[0:50000,:,:]
    Train_CT_labels = CT_labels[0:50000,:]
    Test_CT = CT[50000:,:,:]
    Test_CT_labels = CT_labels[50000:,:]
    # Test_CT = CT[1:1000, :, :]
    # Test_CT_labels = CT_labels[1:1000, :]

    Train_Data = np.vstack((Train_CT,Train_LDCT))
    # Train_Data = np.concatenate((Train_CT,Train_LDCT),axis=0)
    # Train_Data = np.reshape(Train_Data,[-1,32,32,1])
    Train_labels = np.concatenate((Train_CT_labels,Train_LDCT_labels),axis=0)
    Test_Data = np.vstack((Test_CT, Test_LDCT))
    # Test_Data = np.concatenate((Test_CT,Test_LDCT),axis=0)
    # Test_Data = np.reshape(Test_Data, [-1, 32, 32, 1])
    Test_labels = np.concatenate((Test_CT_labels,Test_LDCT_labels),axis=0)


    return Train_Data,Train_labels,Test_Data,Test_labels

Train_Data,Train_labels,Test_Data,Test_labels = Data()
# print(Train_Data.shape,Test_Data.shape,Train_labels.shape,Test_labels.shape)
# print(Test_labels[5].shape,Test_labels[5])
# print(Test_labels[50000].shape,Test_labels[50000])
# print(Train_Data[7].shape,Train_labels[7],Train_labels[7].shape,Train_Data[7])

