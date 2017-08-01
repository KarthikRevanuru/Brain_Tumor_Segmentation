import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import os
import cv2
import mxnet as mx
import pandas as pd
import xgboost as xgb
import math

import numpy as np
import nibabel as nib

folder_names_train=[]
folder_names_test=[]

def get_extractor():
    model = mx.model.FeedForward.load('../Data/Resnet/resnet-50', 0, ctx=mx.cpu(), numpy_batch_size=1)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(), symbol=fea_symbol, numpy_batch_size=64,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    return feature_extractor


def get_3d_data(path):
    epi_img = nib.load(path)
    epi_img_data = epi_img.get_data()
    return epi_img_data


def get_data_id(path):
    sample_image = get_3d_data(path)
    
    sample_image[sample_image == -2000] = 0
    

    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))

        
    batch = np.array(batch)
    return batch


def calc_features():
    net = get_extractor()
    
    for folder in glob.glob('seg/*'):
        batch =get_data_id(folder)
        print (batch.shape)
        feats = net.predict(batch)
        print(feats.shape)
        np.save(folder, feats)
        
   
def folder_names():
    if(folder_names_train==[]):
        for folder in glob.glob('training/*.npy'):
            folder_names_train.append((folder.split('/')[-1].split('.')[0]).rsplit("_", 1)[0])
   
    if(folder_names_test==[]):
        for folder in glob.glob('testing/*.npy'):
            folder_names_test.append((folder.split('/')[-1].split('.')[0]).rsplit("_", 1)[0])


def train_xgboost():
    df = pd.read_csv('survival_data.csv', index_col=0, encoding = 'UTF-7')
    p = np.array([np.mean(np.load('training/%s_flair.nii.gz.npy' % str(id)), axis=0) for id in folder_names_train])
    q = np.array([np.mean(np.load('training/%s_t1.nii.gz.npy' % str(id)), axis=0) for id in folder_names_train])
    r = np.array([np.mean(np.load('training/%s_t1ce.nii.gz.npy' % str(id)), axis=0) for id in folder_names_train])
    s = np.array([np.mean(np.load('training/%s_t2.nii.gz.npy' % str(id)), axis=0) for id in folder_names_train])
    
    y=np.array([])
    t=0
    z=np.array([])
    for ind in range(len(folder_names_train)):
        try:
            temp = df.get_value(str(folder_names_train[ind]),'Survival')
            y=np.append(y,temp)
            temp = df.get_value(str(folder_names_train[ind]),'Age')
            z=np.append(z,np.array([temp]))
        except Exception as e:
            t+=1 
            print (t,str(e),"Label Not found, deleting entry")
            y=np.append(y,0)
    
    z=np.array([[v] for v in z])
    
    t=np.concatenate((p,q),axis=1)
    u=np.concatenate((r,s),axis=1)
    x=np.concatenate((t,u),axis=1) 
    #print(x.shape)
    #print (x)
    #print (x.shape,z.shape)
    x=np.concatenate((x,z),axis=1)
    #print (x)
    #clf=linear_model.LogisticRegression(C=1e5)
    #clf = RandomForestRegressor()
    clf = xgb.XGBRegressor()
    clf.fit(x,y)
    return clf


def make_submit():
    clf = train_xgboost()
    df = pd.read_csv('survival_data.csv', index_col=0)
    p = np.array([np.mean(np.load('testing/%s_flair.nii.gz.npy' % str(id)), axis=0) for id in folder_names_test])
    q = np.array([np.mean(np.load('testing/%s_t1.nii.gz.npy' % str(id)), axis=0) for id in folder_names_test])
    r = np.array([np.mean(np.load('testing/%s_t1ce.nii.gz.npy' % str(id)), axis=0) for id in folder_names_test])
    s = np.array([np.mean(np.load('testing/%s_t2.nii.gz.npy' % str(id)), axis=0) for id in folder_names_test])
    y=np.array([])
    z=np.array([])
    for file1 in folder_names_test:
        try:
            temp = df.get_value(str(file1),'Survival')
            y=np.append(y,temp)
            temp = df.get_value(str(file1),'Age')
            z=np.append(z,temp)
        except Exception as e:
            print (str(e),"Label Not found, deleting entry")
            pass 
            
    z=np.array([[v] for v in z])
    #print (x)
    #print (x.shape,z.shape)
    #x=np.concatenate((x,z),axis=1)
    t=np.concatenate((p,q),axis=1)
    u=np.concatenate((r,s),axis=1)
    x=np.concatenate((t,u),axis=1) 
    x=np.concatenate((x,z),axis=1)
    pred = clf.predict(x)
    
    
    for x in range(len(pred)):
        print (pred[x],",",y[x])
        
    print (np.mean(abs(pred-y)))
        #print ("Prediction of Patient",folder_names_test[x],"is",pred[x]," and actual is",y[x],"\n")
        

if __name__ == '__main__':
   #calc_features()
    folder_names()
    make_submit()
