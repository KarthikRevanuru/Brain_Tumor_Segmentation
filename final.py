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

leng=8928000 #240*240*155
	
def get_train_names():
	train_names=[]
	for folder in glob.glob('1/names/*.gz'):
		train_names.append((folder.split('/')[-1].split('.')[0]).rsplit("_", 1)[0])
		
	
	return train_names

def get_3d_data(path):
	epi_img = nib.load(path)
	epi_img_data = epi_img.get_data()
	return epi_img_data
	
def get_pixels(nparray):
	conc=[]
	dim=nparray.shape
	a=dim[1]
	b=dim[2]
	c=dim[3]
	for i in range(0,a):
		for j in range(0,b):
			for k in range(0,c):
				conc.append(np.array([i,j,k]))
		
	return conc
    
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
	
def get_avg(nparray):
	output=[]
	dim=nparray.shape
	a=dim[1]
	b=dim[2]
	c=dim[3]
	
	
def reshape_feat(pix,p,q,r,s,pg,qg,rg,sg,pcx,qcx,rcx,scx,pcy,qcy,rcy,scy):
	temp=[]
	for i in range(0,leng):
		temp.append([pix[i][0],pix[i][1],pix[i][2],p[i],q[i],r[i],s[i],pg[i],qg[i],rg[i],sg[i],pcx[i],qcx[i],rcx[i],scx[i],pcy[i],qcy[i],rcy[i],scy[i]])
		
	return temp

def reshape_seg(y):
	temp=[]
	for i in range(0,leng):
		temp.append(y[i])
		
	return temp
    
def seg(tt,path):
	p=np.array([get_3d_data(tt+"/"+path+"_flair.nii.gz")])
	q=np.array([get_3d_data(tt+"/"+path+"_t1.nii.gz")])
	r=np.array([get_3d_data(tt+"/"+path+"_t1ce.nii.gz")])
	s=np.array([get_3d_data(tt+"/"+path+"_t2.nii.gz")])
	y=np.array([get_3d_data(tt+"/"+path+"_seg.nii.gz")])
	
	
	pix=get_pixels(p)
	pc=concat(p)
	qc=concat(q)
	rc=concat(r)
	sc=concat(s)
	yc=concat(y)
	print (p[0].shape)
	px = cv2.Sobel(p[0],cv2.CV_64F,1,0,ksize=5)
	py = cv2.Sobel(p[0],cv2.CV_64F,0,1,ksize=5)
	qx = cv2.Sobel(q[0],cv2.CV_64F,1,0,ksize=5)
	qy = cv2.Sobel(q[0],cv2.CV_64F,0,1,ksize=5)
	rx = cv2.Sobel(r[0],cv2.CV_64F,1,0,ksize=5)
	ry = cv2.Sobel(r[0],cv2.CV_64F,0,1,ksize=5)
	sx = cv2.Sobel(s[0],cv2.CV_64F,1,0,ksize=5)
	sy = cv2.Sobel(s[0],cv2.CV_64F,0,1,ksize=5)
	print(time.strftime('%a %H:%M:%S'))
	pcx=concat1(px)
	pcy=concat1(py)
	qcx=concat1(qx)
	qcy=concat1(qy)
	rcx=concat1(rx)
	rcy=concat1(ry)
	scx=concat1(sx)
	scy=concat1(sy)
	
	print(time.strftime('%a %H:%M:%S'))
	pa=ndimage.filters.convolve(p[0],np.full((5, 5, 5), 1.0/125),mode='constant')
	qa=ndimage.filters.convolve(q[0],np.full((5, 5, 5), 1.0/125),mode='constant')
	ra=ndimage.filters.convolve(r[0],np.full((5, 5, 5), 1.0/125),mode='constant')
	sa=ndimage.filters.convolve(s[0],np.full((5, 5, 5), 1.0/125),mode='constant')
	print(time.strftime('%a %H:%M:%S'))
	pg=concat1(pa)
	qg=concat1(qa)
	rg=concat1(ra)
	sg=concat1(sa)
	print(time.strftime('%a %H:%M:%S'))
	X=reshape_feat(pix,pc,qc,rc,sc,pg,qg,rg,sg,pcx,qcx,rcx,scx,pcy,qcy,rcy,scy)
	Y=reshape_seg(yc)
	print(time.strftime('%a %H:%M:%S'))
	xl0=[]
	yl0=[]
	xl1=[]
	yl1=[]
	xl2=[]
	yl2=[]
	xl4=[]
	yl4=[]
	
	for i in range(0,len(Y)):
		if(Y[i]==0):
			xl0.append(X[i])
			yl0.append(Y[i])
		elif(Y[i]==1):
			xl1.append(X[i])
			yl1.append(Y[i])
		elif(Y[i]==2):
			xl2.append(X[i])
			yl2.append(Y[i])
		elif(Y[i]==4):
			xl4.append(X[i])
			yl4.append(Y[i])
	
	a=len(yl0)
	b=len(yl1)
	c=len(yl2)
	d=len(yl4)	
	print(time.strftime('%a %H:%M:%S'))
	min1=min(a,b,c,d)
	fin_X=[]
	fin_Y=[]
	print ("here")
	print (a)
	print (b)
	print (c)
	print (d)
	print (min1)
	print ("end\n")
	ind=random.sample(xrange(0,a), min1)
	for x in ind:
		fin_X.append(xl0[x])
		fin_Y.append(yl0[x])
	
	ind=random.sample(xrange(0,b), min1)
	for x in ind:
		fin_X.append(xl1[x])
		fin_Y.append(yl1[x])
	ind=random.sample(xrange(0,c), min1)
	for x in ind:
		fin_X.append(xl2[x])
		fin_Y.append(yl2[x])
	ind=random.sample(xrange(0,d), min1)
	for x in ind:
		fin_X.append(xl4[x])
		fin_Y.append(yl4[x])
	
	print(time.strftime('%a %H:%M:%S'))
	return fin_X,fin_Y
	
def pre_process_train():
	train_names=get_train_names()
	print (train_names)	
	X,Y=[],[]
	for item in train_names:
		tempx,tempy=seg("1",item)
		X=X+tempx
		Y=Y+tempy
		
	X=np.asarray(X)
	Y=np.asarray(Y)
	print("pre-processing done")
	print(X.shape)
	print(Y.shape)
	np.save('train1_x', X)
	np.save('train1_y',Y)

if __name__ == '__main__':
	pre_process_train()
