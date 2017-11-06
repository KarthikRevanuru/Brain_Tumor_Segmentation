import time
print(time.strftime('%a %H:%M:%S'))
import numpy as np
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import math
import nibabel as nib
from scipy import ndimage
import cv2
import random

#leng=8928000 #240*240*155
	
def get_train_names():
	imgNames = []
	with open("MinMax.txt") as f:
		#f.readline()
		hashMap = {}
		for lines in f.readlines():
			[name,X_min,Y_min,Z_min,X_max,Y_max,Z_max] = lines.split()[:7]
			X_min,Y_min,Z_min,X_max,Y_max,Z_max = int(X_min),int(Y_min),int(Z_min),int(X_max),int(Y_max),int(Z_max)
			try:
				hashMap[name].append([X_min,Y_min,Z_min,X_max,Y_max,Z_max])
			except:
				hashMap[name] = [[X_min,Y_min,Z_min,X_max,Y_max,Z_max]]
				imgNames.append(name)
	
	return imgNames

def get_3d_data(path):
	epi_img = nib.load(path)
	epi_img_data = epi_img.get_data()
	return epi_img_data
    
def concat(nparray):
	conc=[]
	dim=nparray.shape
	a=dim[1]
	b=dim[2]
	c=dim[3]
	for i in range(0,a):
		for j in range(0,b):
			for k in range(0,c):
				conc.append(nparray[0][i][j][k])
		
	return conc

def concat1(nparray):
	conc=[]
	dim=nparray.shape
	a=dim[0]
	b=dim[1]
	c=dim[2]
	for i in range(0,a):
		for j in range(0,b):
			for k in range(0,c):
				conc.append(nparray[i][j][k])
		
	return conc
	
	
def reshape_feat(p,pg,pcx,pcy,leng):
	temp=[]
	for i in range(0,leng):
		temp.append([p[i],pg[i],pcx[i],pcy[i]])
		
	return temp

def reshape_seg(y,leng):
	temp=[]
	for i in range(0,leng):
		if(y[i]==0):
			temp.append(0)
		else:
			temp.append(1)
		
	return temp
    
def seg(path):
	p=np.array([get_3d_data('../../../Cut_Brats_Training_Data/Train/'+"cut"+path+"_flair.nii.gz")])
	
	y=np.array([get_3d_data('../../../Cut_Brats_Training_Data/Train/'+"cut"+path[4:]+"_seg.nii.gz")])
	
	
	shap=p[0].shape
	print (shap)
	
	leng=shap[0]*shap[1]*shap[2]
	
	#pix=get_pixels(path)
	pc=concat(p)
	
	yc=concat(y)
	print (p[0].shape)
	px = cv2.Sobel(p[0],cv2.CV_64F,1,0,ksize=5)
	py = cv2.Sobel(p[0],cv2.CV_64F,0,1,ksize=5)
	
	print(time.strftime('%a %H:%M:%S'))
	pcx=concat1(px)
	pcy=concat1(py)
	
	
	print(time.strftime('%a %H:%M:%S'))
	pa=ndimage.filters.convolve(p[0],np.full((5, 5, 5), 1.0/125),mode='constant')
	
	print(time.strftime('%a %H:%M:%S'))
	pg=concat1(pa)
	
	print(time.strftime('%a %H:%M:%S'))
	X=reshape_feat(pc,pg,pcx,pcy,leng)
	Y=reshape_seg(yc,leng)
	print(time.strftime('%a %H:%M:%S'))
	
	return X,Y
	
def pre_process_train():
	train_names=get_train_names()
	print (train_names)	
	X,Y=[],[]
	for item in train_names:
		tempx,tempy=seg(item)
		X=X+tempx
		Y=Y+tempy
		
	X=np.asarray(X)
	Y=np.asarray(Y)
	#x=np.load("location.npy")
	X_f=X#np.concatenate((x,X),axis=1)
	
	print("pre-processing done")
	print(X_f.shape)
	print(Y.shape)
	np.save('train_x', X_f)
	np.save('train_y',Y)
	print(time.strftime('%a %H:%M:%S'))
	
if __name__ == '__main__':
	pre_process_train()
