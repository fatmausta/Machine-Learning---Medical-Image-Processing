# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:53:54 2017

@author: fusta
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:59:08 2017

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
#PARAMETERS TO ADJUST
nclasses = 2
randomly_drop = 0
#desired ratio of true positives, for scar in this case
desired_ratio_balance = 0.50
datapopfraction = 0.80

patch_size = 4
window_size = 32
filter_size = 2
epochs = 1
skip = 4
modelname= 'CNN_scar_1.h5'
pid_train = np.array(['0329','0364','0417'])#, '0424', '0450', '0473', '0485','0493', '0494', '0495', '0515', '0519', '0529', '0546', '0562', '0565', '0574', '0578', '0587', '0591'])

#datapath = 'DataCNNScarNorm/' #for sharcnet work directory
datapath = 'C:\\Users\\fusta\\Dropbox\\1_Machine_Learning\\DataCNNScarNorm\\'
#TRAINING

patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)

def PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip):  
    patch_labels_training = []
    patch_labels_testing = []    
    window_intensities_training = []
    window_intensities_testing = []
    pads = []
    LGE_patches_scar = []
    LGE_patches_bg = []
    LGE_windows_scar = []
    LGE_windows_bg = []
    LGE_padded_slice = []
    
    #make windows size and patch size evenly dvideble 
    if (window_size-patch_size)%2 != 0:
        window_size +=1 

    for pid in pid_train:
        nskippedslice=0

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
        #calculate the amount of padding for height and width of a slice for patches
        rem_w = w_LGE%patch_size
        w_pad=patch_size-rem_w      
        rem_h = h_LGE%patch_size
        h_pad=patch_size-rem_h    
        pads.append((h_pad,w_pad))
        #calculate len of LGE Patches  estimate)
        len_samples = (h_LGE*w_LGE)/(patchsize_sq)
        random_datapopnumber=len_samples*datapopfraction
        all_slice = range(0, d_LGE, skip)#15,5)#30,60,2)
    
        for sl in all_slice:           
            if sl<d_LGE:
                LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
                scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
                LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
                LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size))
                #calculate padding and pad the images one more time before making windows
                padding = int((window_size - patch_size)/2)
                LGE_repadded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
                LGE_padded_slice = []
                #done with the labels, now we will do it for our windows, 
                LGE_windows = view_as_windows(LGE_repadded_slice, (window_size,window_size), step=patch_size)
                LGE_repadded_slice=[]
                LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))        
                rang=[]
                #sum the window intensities       
                for r in range(0,len(LGE_windows)):
                    if(np.sum(LGE_windows[r])==0):
                        rang.append(r)
                #remove samples from outside of myocardium. 
                LGE_patches = np.delete(LGE_patches, rang, axis = 0) 
                LGE_windows = np.delete(LGE_windows, rang, axis = 0)  
                
                #BALANCE YOUR DATA IN A CONTROLLED WAY
                #1) SEPERATE SCAR FROM BACKGROUND
                for r in range(0,len(LGE_patches)):
                    if(np.sum(LGE_patches[r])/patch_size>=1):#scar 
                        LGE_patches_scar.append(LGE_patches[r])
                        LGE_windows_scar.append(LGE_windows[r])
                    else: #background
                        LGE_patches_bg.append(LGE_patches[r])
                        LGE_windows_bg.append(LGE_windows[r])
                #2) CALCULATE AMOUNT OF DATA TO BE DROPPED
                if len(LGE_patches_bg) == 0 or len(LGE_patches_scar) == 0:
                    print('\nskipping slice %d:' %sl)
                    nskippedslice=nskippedslice+1
                    continue
                else:
                    ratio_imbalance = len(LGE_patches_scar)/(len(LGE_patches_bg)+len(LGE_patches_scar))#re-pad your padded image before you make your windows    
                    #delete the imbalanced data in a controlled way
                    if len(LGE_patches_bg)>len(LGE_patches_scar):#drop from background data only
                        #formula to decide how many samples to drop from background
                        controlled_datapopnumber = (desired_ratio_balance-ratio_imbalance)*len(LGE_patches)/desired_ratio_balance
                        
                        if controlled_datapopnumber<=0 or controlled_datapopnumber>=len(LGE_patches_bg):
                            continue
                        randomrange=random.sample(range(0, len(LGE_patches_bg)), int(controlled_datapopnumber))
                        #delete from background
                        LGE_patches_bg = np.delete(LGE_patches_bg, randomrange, axis = 0) 
                        LGE_windows_bg = np.delete(LGE_windows_bg, randomrange, axis = 0) 
                        #just reshape scar samples
                        LGE_patches_scar = np.reshape(LGE_patches_scar, (len(LGE_patches_scar), patch_size, patch_size))
                        LGE_windows_scar = np.reshape(LGE_windows_scar, (len(LGE_windows_scar), window_size, window_size))
    
                    else:#more scar than bg, so drop some from the scar region
                        controlled_datapopnumber = (desired_ratio_balance-ratio_imbalance)*len(LGE_patches)/desired_ratio_balance                 
                        randomrange=random.sample(range(1, len(LGE_patches_scar)), int(controlled_datapopnumber))
                        #delete from scar samples
                        LGE_patches_scar = np.delete(LGE_patches_scar, randomrange, axis = 0) 
                        LGE_windows_scar = np.delete(LGE_windows_scar, randomrange, axis = 0) 
                        #just reshape background samples
                        LGE_patches_bg = np.reshape(LGE_patches_bg, (len(LGE_patches_bg), patch_size, patch_size))
                        LGE_windows_bg = np.reshape(LGE_windows_bg, (len(LGE_windows_bg), window_size, window_size))
                    
                    #combine left-over desired scar and background patches together
                    LGE_patches = np.concatenate((LGE_patches_scar,LGE_patches_bg),axis=0)
                    LGE_windows = np.concatenate((LGE_windows_scar,LGE_windows_bg),axis=0)

                    LGE_patches_scar=[]
                    LGE_patches_bg=[]
                    LGE_windows_scar=[]
                    LGE_windows_bg=[]
                    
                    #delete random samples to reduce the amount of data
                    if randomly_drop==1:    
                        randomrange=random.sample(range(1, len(LGE_patches)), int(random_datapopnumber))
                        LGE_patches = np.delete(LGE_patches, randomrange, axis = 0) 
                        LGE_windows = np.delete(LGE_windows, randomrange, axis = 0) 
                    #calculate the label values for the patches
                    for p in range(0,len(LGE_patches)):            
                        label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))

                        label=int(numpy.divide( numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))

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
        print('number of skipped slices for patient %s: %d'%(pid, nskippedslice))

        training_data= list(zip(numpy.uint8(window_intensities_training),numpy.uint8(patch_labels_training)))
        testing_data= list(zip(numpy.uint8(window_intensities_testing),numpy.uint8(patch_labels_testing)))  
        print('\n\nsize of training data %d'%len(training_data))
        
    return training_data, testing_data, pads
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

def runCNNModel(dataset_training, dataset_testing, pads, epochs, patch_size, window_size, nclasses, datapath):

    # preprocessing
    X_training = np.zeros((len(dataset_training),windowsize_sq)).astype('int16')
    Y_training = np.zeros((len(dataset_training),1)).astype('int16')
    X_testing = np.zeros((len(dataset_testing),windowsize_sq)).astype('int16')
    Y_testing = np.zeros((len(dataset_testing),1)).astype('int16')

    for p in range(0,len(dataset_testing)):
        X_testing[p]=dataset_testing[p][0]
        Y_testing[p]=dataset_testing[p][1]

    for p in range(0,len(dataset_training)):
        X_training[p]=dataset_training[p][0]
        Y_training[p]=dataset_training[p][1]
    #count the samples with scar in it
    X_training_scar = X_training[np.where(Y_training==1)]
    X_training_bg = X_training[np.where(Y_training==0)]

    print('\ntotal number of samples: %d' % len(X_training))        
    print('number of scar samples in the training data: %d' % len(X_training_scar))
    print('background is %d percent of entire data' %  ( len(X_training_bg) / len(X_training)*100))    
    print('scar is %d percent of entire data' %  ( len(X_training_scar) / len(X_training)*100))

    #Reshape my dataset for my model       
    X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
    X_testing = X_testing.reshape(X_testing.shape[0], 1, window_size, window_size)
    X_training = X_training.astype('float32')
    X_testing = X_testing.astype('float32')
    X_training /= 255
    X_testing /= 255
    
    Y_training = np_utils.to_categorical(Y_training, nclasses)
    Y_testing = np_utils.to_categorical(Y_testing, nclasses)

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
(dataset_training, dataset_testing, pads) = PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip)
y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, pads, epochs, patch_size, window_size, nclasses, datapath)


#to do a finer segmentation, save the mpodel
#patch_size = 2
#window_size = 16
#patchsize_sq = np.square(patch_size)
#windowsize_sq = np.square(window_size)
#numpy.random.seed(windowsize_sq-1)
#modelname= 'CNN_scar_2.h5'
#(dataset_training, dataset_testing, pads) = PatchMaker(patch_size, window_size, nclasses, pid_train, datapath, skip)
#y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, pads, epochs, patch_size, window_size, nclasses, datapath)
