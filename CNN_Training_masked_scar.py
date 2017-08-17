# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:39:19 2017

@author: fusta
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:25:02 2017

@author: fusta
"""
import numpy as np
import keras
import SimpleITK
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy
import SimpleITK as sitk
K.set_image_dim_ordering('th')
from skimage.util import view_as_windows
from skimage.util import view_as_blocks
import random

patch_size = 8
window_size = 32
nclasses = 2
epochs = 20

filter_size = 2
datapopfraction = 0.80
modelname= 'CNN_scar_1.h5'
#pid = '0329'
#sl = 30
#datapath = 'DataCNNScarNorm/' #for sharcnet work directory
datapath = 'C:\\Users\\fusta\\Dropbox\\1_Machine_Learning\\DataCNNScarNorm\\'
skip=4
pid_train = np.array(['0329','0364'])#,'0417', '0424', '0450', '0473', '0493', '0494', '0495', '0515', '0519', '0529', '0546', '0562', '0565', '0574', '0578', '0587', '0591', '0601'])#, '0632', '0715', '0730', '0917', '0921', '0953', '1036', '1073', '1076', '1115', '1166', '1168', '1171', '1179'])
pid_test = np.array([('0485')])#for pid in pids:

patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
test_slice = range(30,31)#15-45

def PatchMaker(patch_size, window_size, nclasses, pid_train, pid_test, test_slice, datapath, skip):  
    pid_all = np.concatenate((pid_train, pid_test))
    patch_labels_training=[]
    patch_labels_testing=[]    
    window_intensities_training=[]
    window_intensities_testing=[]
    test_img_shape = []#np.empty((len(pid_test),2))
    pads=[]
    LGE_padded_slice = np.empty(())
            #make windows size and patch size evenly dvideble 
    if (window_size-patch_size)%2 != 0:
        window_size +=1 
    for pid in pid_train:
        LGE = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-LGE-cropped.mhd')
        scar = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-scar-cropped.mhd')
        myo = SimpleITK.ReadImage(datapath + pid + '//' + pid + '-myo-cropped.mhd')
        
        #convert a SimpleITK object into an array
        LGE_3D = SimpleITK.GetArrayFromImage(LGE)
        myo_3D = SimpleITK.GetArrayFromImage(myo)
        scar_3D = SimpleITK.GetArrayFromImage(scar) 
        #masking the LGE
        LGE_3D = np.multiply(LGE_3D,myo_3D)
        
        d_LGE = LGE_3D.shape[0]
        h_LGE = LGE_3D.shape[1]
        w_LGE = LGE_3D.shape[2] 

        nonzero = 0   
        #calculate the amount of padding for heaght and width of a slice for patches
        rem_w = w_LGE%patch_size
        w_pad=patch_size-rem_w      
        rem_h = h_LGE%patch_size
        h_pad=patch_size-rem_h    
        pads.append((h_pad,w_pad))
        #calculate len of LGE Patches  estimate)
        len_samples = (h_LGE*w_LGE)/(patchsize_sq)
        datapopnumber=len_samples*datapopfraction

        all_slice = range(0, d_LGE, skip)#15,5)#30,60,2)        for sl in all_slice:

        for sl in all_slice:           
            if sl<d_LGE:
                LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
                scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
                LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
                LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size))
                #find background patches
                LGE_patches_bg = LGE_patches[np.where(LGE_patches==0)]
                #re-pad your padded image before you make your windows
                padding = int((window_size - patch_size)/2)
                LGE_repadded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
                LGE_padded_slice = []
                #done with the labels, now we will do our windows, 
                LGE_windows = view_as_windows(LGE_repadded_slice, (window_size,window_size), step=patch_size)
                LGE_repadded_slice=[]
                LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))        
                #find background windows
                LGE_windows_bg = LGE_windows[np.where(LGE_patches==0)]
                rang=[]
       
                #delete random samples to resuce the amount of data
#                for each patches: pop some patches and corresponding windows
                randomrange=random.sample(range(1, len(LGE_patches)), int(datapopnumber))
                LGE_patches = np.delete(LGE_patches, randomrange, axis = 0) 
                LGE_windows = np.delete(LGE_windows, randomrange, axis = 0) 
                #delete background 
                #sum the window intensities       
                for r in range(0,len(LGE_patches)):
                    if(np.sum(LGE_patches[r])==0):
                        rang.append(r)
                        
                LGE_patches = np.delete(LGE_patches, rang, axis = 0) 
                LGE_windows = np.delete(LGE_windows, rang, axis = 0)   
                
                for p in range(0,len(LGE_patches)):            
                    label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))
                    label = numpy.reshape(label, (1,1))           
                    if label==nclasses:
                        label -=1 #mmake sure the range for the classes do not exceed the maximum
                    #making your window  intensities a single row
                    intensities = numpy.reshape(LGE_windows[p],(window_size*window_size))
                    intensities = numpy.reshape(intensities, (1,window_size*window_size))

                    patch_labels_training.append(label)
                    window_intensities_training.append(intensities)
                     
            else:
                print('sl>=d_LGE %d >= %d' % (sl,d_LGE))
                break
        if pid in pid_test:
            print(pid)
            test_img_shape.append(LGE_padded_slice.shape)        #Now, intensities=windows and patches labels will be comboined

        training_data= list(zip(numpy.uint8(window_intensities_training),numpy.uint8(patch_labels_training)))
        testing_data= list(zip(numpy.uint8(window_intensities_testing),numpy.uint8(patch_labels_testing)))  
        
    return training_data, testing_data, test_img_shape, pads, pid_all
    numpy.savetxt('training.csv', training_data ,fmt='%s', delimiter=',' ,newline='\r\n') 
    numpy.savetxt('testing.csv', testing_data ,fmt='%s', delimiter=',' ,newline='\r\n') 
    print(nonzero)      

#Dice Calculation
def DiceIndex(BW1, BW2):
    BW1 = BW1.astype('float32')
    BW2 = BW2.astype('float32')
    #elementwise multiplication
    t= (np.multiply(BW1,BW2))
    total = np.sum(t)
    DI=2*total/(np.sum(BW1)+np.sum(BW2))
    DI=DI*100
    return DI

def runCNNModel(dataset_training, dataset_testing, test_img_shape, pads, epochs, patch_size, window_size, nclasses, pid_test, datapath):
    print('\nsample size: %d' % len(dataset_training))
    #count the samp;les with scar in it
    # preprocessing
    X_training = np.zeros((len(dataset_training),windowsize_sq))
    Y_training = np.zeros((len(dataset_training),1))
    X_testing = np.zeros((len(dataset_testing),windowsize_sq))
    Y_testing = np.zeros((len(dataset_testing),1))

    for p in range(0,len(dataset_testing)):
        X_testing[p]=dataset_testing[p][0]
        Y_testing[p]=dataset_testing[p][1]

    for p in range(0,len(dataset_training)):
        X_training[p]=dataset_training[p][0]
        Y_training[p]=dataset_training[p][1]

    X_training_scar = X_training[np.where(Y_training>=1)]
    Y_training_scar = Y_training[np.where(Y_training>=1)] 
    
    print('\nonly scar sample size: %d' % len(X_training_scar))
    print('\nscar is %d percent of entire data' %  ( len(X_training_scar) / len(X_training)*100))

    #Reshape my dataset for my model       
    X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
    X_testing = X_testing.reshape(X_testing.shape[0], 1, window_size, window_size)
    X_training = X_training.astype('float32')
    X_testing = X_testing.astype('float32')
    X_training /= 255
    X_testing /= 255
    
    Y_training = np_utils.to_categorical(Y_training, nclasses)
    Y_testing = np_utils.to_categorical(Y_testing, nclasses)

#    X_training_scar = X_training[np.where(Y_training>=1)]
#    Y_training_scar = Y_training[np.where(Y_training>=1)] 
    
    multiply_data = int((len(X_training)/len(X_training_scar))/2)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    model = Sequential()
    model.add(Convolution2D(16, filter_size, filter_size, activation='relu', input_shape=(1,window_size,window_size), dim_ordering='th'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, filter_size, filter_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nclasses, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_training, Y_training, epochs=epochs, batch_size=100, shuffle=True, verbose = 2)     
    model.summary()

    #save your model
    model.save(modelname)#path to  save  "C:\Users\fusta\Dropbox\1_Machine_Learning\Machine Learning\KerasNN\Neural_Network_3D_Scar\2D\Data Augmentation\Model.h5"    
    y_pred_scaled_cropped = []#.append(y_pred_scaled[p][:-pads[p+len(pid_train)][0],:-pads[p+len(pid_train)][1]])
    return y_pred_scaled_cropped

#to do a rough segmentation, save the ,model
(dataset_training, dataset_testing, test_img_shape, pads, pid_all) = PatchMaker(patch_size, window_size, nclasses, pid_train, pid_test, test_slice, datapath, skip)
y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, test_img_shape, pads, epochs, patch_size, window_size, nclasses, pid_test, datapath)
#to do a finer segmentation, save the mpodel
patch_size = 2
window_size = 16
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
modelname= 'CNN_scar_2.h5'
(dataset_training, dataset_testing, test_img_shape, pads, pid_all) = PatchMaker(patch_size, window_size, nclasses, pid_train, pid_test, test_slice, datapath, skip)
y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, test_img_shape, pads, epochs, patch_size, window_size, nclasses, pid_test, datapath)
