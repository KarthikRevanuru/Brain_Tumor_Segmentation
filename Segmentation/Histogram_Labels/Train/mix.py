#Required Packages
import numpy as np #Version '1.13.0'
import nibabel as nib #Version '2.1.0'
import glob

def Hist_Lab(name):
	#Loading .nii.gz image
	epiImg = nib.load('../../../Normalised_Brats_Training_Data/Train/'+name+"_flair.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()

	#Converting to Numpy Format 
	imgNpFormat=np.array(epiImgData)
	#print (np.amax(imgNpFormat))

	#Dimensions of Image
	xLen, yLen, zLen = imgNpFormat.shape

	histAllYZ = []
	histAllXZ = []
	histAllXY = []

	#Keep X const iterate through everything else
	for _ in range(xLen):
		ytemp=[]
		for y in range(yLen):
			ztemp=[]
			for z in range(zLen):
				ztemp.append(imgNpFormat[_][y][z])
			
			ytemp.append(ztemp)
		
			
		imgYZ = np.array(ytemp) #Keep x constant take only y & z
		
		histImgYZ = np.histogram(imgYZ,bins = range(257))[0] #Ouput Format (hist,bin_edges)
		histAllYZ.append(histImgYZ)	
	
	#Keep Y const iterate through everything else
	for _ in range(yLen):
		xtemp=[]
		for x in range(xLen):
			ztemp=[]
			for z in range(zLen):
				ztemp.append(imgNpFormat[x][_][z])
			
			xtemp.append(ztemp)
	
	
		imgXZ = np.array(xtemp) #Keep y constant take only x & z
		
		
		histImgXZ = np.histogram(imgXZ,bins = range(257))[0] #Ouput Format (hist,bin_edges)
		histAllXZ.append(histImgXZ)	

	#Keep Z const iterate through everything else
	for _ in range(zLen):
		xtemp=[]
		for x in range(xLen):
			ytemp=[]
			for y in range(yLen):
				ytemp.append(imgNpFormat[x][y][_])
			
			xtemp.append(ytemp)
		
		imgXY = np.array(xtemp)  #Keep z constant take only x & y
		
		
		histImgXY = np.histogram(imgXY,bins = range(257))[0] #Ouput Format (hist,bin_edges)
		histAllXY.append(histImgXY)	

	saveHistAllYZ = np.array(histAllYZ)	
	saveHistAllXZ = np.array(histAllXZ)
	saveHistAllXY = np.array(histAllXY)

	#Loading .nii.gz image
	epiImg = nib.load('../../../Normalised_Brats_Training_Data/Train/seg/'+name[4:]+"_seg.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()

	#Converting to Numpy Format 
	imgNpFormat=np.array(epiImgData)

	#Dimensions of Image
	xLen, yLen, zLen = imgNpFormat.shape
	#print (xLen, yLen, zLen)

	histAllYZ = []
	histAllXZ = []
	histAllXY = []

	labelAllYZ = []
	labelAllXZ = []
	labelAllXY = []
	#Keep X const iterate through everything else
	for _ in range(xLen):
		ytemp=[]
		for y in range(yLen):
			ztemp=[]
			for z in range(zLen):
				ztemp.append(imgNpFormat[_][y][z])
			
			ytemp.append(ztemp)
		
			
		imgYZ = np.array(ytemp) #Keep x constant take only y & z
		#print (imgYZ.shape)
		histImgYZ = np.histogram(imgYZ,bins = range(6))[0] #Ouput Format (hist,bin_edges)
		#print (_,histImgYZ[0])
		if(histImgYZ[0]==37200):
			labelAllYZ.append(0)
		else:
			labelAllYZ.append(1)
		#print ("X",_,histImgYZ[0],labelAllYZ[-1])
		
	
	#Keep Y const iterate through everything else
	for _ in range(yLen):
		xtemp=[]
		for x in range(xLen):
			ztemp=[]
			for z in range(zLen):
				ztemp.append(imgNpFormat[x][_][z])
			
			xtemp.append(ztemp)
	
	
		imgXZ = np.array(xtemp) #Keep y constant take only x & z
		#print (imgXZ.shape)
		histImgXZ = np.histogram(imgXZ,bins = range(6))[0] #Ouput Format (hist,bin_edges)
		if(histImgXZ[0]==37200):
			labelAllXZ.append(0)
		else:
			labelAllXZ.append(1)
		#print ("Y",_,histImgXZ[0],labelAllXZ[-1])	
	
	#Keep Z const iterate through everything else
	for _ in range(zLen):
		xtemp=[]
		for x in range(xLen):
			ytemp=[]
			for y in range(yLen):
				ytemp.append(imgNpFormat[x][y][_])
			
			xtemp.append(ytemp)
		
		imgXY = np.array(xtemp)  #Keep z constant take only x & y
		#print (imgXY.shape)
		histImgXY = np.histogram(imgXY,bins = range(6))[0] #Ouput Format (hist,bin_edges)
		if(histImgXY[0]==57600):
			labelAllXY.append(0)
		else:
			labelAllXY.append(1)
		#print ("Z",_,histImgXY[0],labelAllXY[-1])
		
	
	savelabelAllYZ = np.array(labelAllYZ)	
	savelabelAllXZ = np.array(labelAllXZ)
	savelabelAllXY = np.array(labelAllXY)
	
	posYZ=[]
	posXZ=[]
	posXY=[]
	
	for i in range(0,savelabelAllYZ.size):
		posYZ.append([i,"0","0"])
		print (name,i,"0","0",savelabelAllYZ[i],end=" ")
		#for x in saveHistAllYZ[i]:
		#	print (x, end=" ")
		print ('', end = '\n')
		
	for i in range(0,savelabelAllXZ.size):
		posXZ.append(["0",i,"0"])
		print (name,"0",i,"0",savelabelAllXZ[i],end=" ")
		#for x in saveHistAllXZ[i]:
		#	print (x, end=" ")
		print ('', end = '\n')

	for i in range(0,savelabelAllXY.size):
		posXY.append(["0","0",i])
		print (name,"0","0",i,savelabelAllXY[i],end=" ")
		#for x in saveHistAllXY[i]:
		#	print (x, end=" ")
		print ('', end = '\n')
		
	np_pos_YZ=np.asarray(posYZ)
	np_pos_XZ=np.asarray(posXZ)
	np_pos_XY=np.asarray(posXY)
	
	temp_pos=np.concatenate((np_pos_YZ,np_pos_XZ), axis=0)
	final_pos=np.concatenate((temp_pos,np_pos_XY), axis=0)
		
	thist=np.concatenate((saveHistAllYZ,saveHistAllXZ), axis=0)
	X=np.concatenate((thist,saveHistAllXY), axis=0)
	
	X_f=np.concatenate((final_pos,X),axis=1)
	
	tlabel=np.concatenate((savelabelAllYZ,savelabelAllXZ), axis=0)
	y=np.concatenate((tlabel,savelabelAllXY), axis=0)

	return X_f.tolist(),y.tolist()
	
		
		
def get_train_names():
	train_names=[]
	for folder in glob.glob('../../../Normalised_Brats_Training_Data/Train/*.gz'):
		temp=(folder.split('/')[-1].split('.')[0]).rsplit("_", 1)[0]
		if temp not in train_names:
			train_names.append(temp)
		
	
	return train_names	
	
def do():
	X,Y=[],[]
	train_names=get_train_names()
	#print (train_names)
	for x in train_names:
		#print (x)
		tempx,tempy=Hist_Lab(x)
		X=X+tempx
		Y=Y+tempy
		#print ("One Scan Done")

	X=np.asarray(X)
	Y=np.asarray(Y)
	#print("pre-processing done")
	#print(X.shape)
	#print(Y.shape)
	np.save('hist', X)
	np.save('label', Y)
		
if __name__ == '__main__':
	do()
