#Required Packages
import time
print(time.strftime('%a %H:%M:%S'))
import numpy as np #Version '1.13.0'
import nibabel as nib #Version '2.1.0'
import glob

def normalise(name,lm,rm):
	#Loading .nii.gz image
	epiImg = nib.load('../../../Brats_Training_Data/Test/'+name+"_flair.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()
	array=[]
	
	imgNpFormat=np.array(epiImgData)
	xLen, yLen, zLen = imgNpFormat.shape

	array = imgNpFormat.flatten()

	array.sort()
	min_ele=array[1]
	max_ele=array[len(array)-10000]
	
	for i in range(0,xLen):
		for j in range(0,yLen):
			for k in range(0,zLen):
				if(imgNpFormat[i][j][k]==0):
					continue
				else:
					if(imgNpFormat[i][j][k]<lm):
						min_ele=1
						max_ele=lm
						try:
							imgNpFormat[i][j][k]= 1+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*rm)
						except ZeroDivisionError:
    							imgNpFormat[i][j][k]= 1
						
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255
					else:
						min_ele=lm
						max_ele=array[len(array)-10000]
						try:
							imgNpFormat[i][j][k]= rm+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*(float(255-rm)))
						except ZeroDivisionError:
							imgNpFormat[i][j][k]= rm
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255
						

	#Converting to Numpy Format 
	
	array = nib.Nifti1Image(np.array(imgNpFormat),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	nib.save(array, "norm"+name+"_flair.nii.gz")

	#Loading .nii.gz image
	epiImg = nib.load('../../../Brats_Training_Data/Test/'+name+"_t1.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()
	array=[]
	
	imgNpFormat=np.array(epiImgData)
	xLen, yLen, zLen = imgNpFormat.shape

	array = imgNpFormat.flatten()

	array.sort()
	min_ele=array[1]
	max_ele=array[len(array)-100]
	
	for i in range(0,xLen):
		for j in range(0,yLen):
			for k in range(0,zLen):
				if(imgNpFormat[i][j][k]==0):
					continue
				else:
					if(imgNpFormat[i][j][k]<lm):
						min_ele=1
						max_ele=lm
						try:
							imgNpFormat[i][j][k]= 1+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*rm)
						except ZeroDivisionError:
    							imgNpFormat[i][j][k]= 1
						
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255
					else:
						min_ele=lm
						max_ele=array[len(array)-10000]
						try:
							imgNpFormat[i][j][k]= rm+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*(float(255-rm)))
						except ZeroDivisionError:
							imgNpFormat[i][j][k]= rm
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255

	#Converting to Numpy Format 
	
	array = nib.Nifti1Image(np.array(imgNpFormat),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	nib.save(array, "norm"+name+"_t1.nii.gz")

	#Loading .nii.gz image
	epiImg = nib.load('../../../Brats_Training_Data/Test/'+name+"_t1ce.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()
	array=[]
	
	imgNpFormat=np.array(epiImgData)
	xLen, yLen, zLen = imgNpFormat.shape

	array = imgNpFormat.flatten()

	array.sort()
	min_ele=array[1]
	max_ele=array[len(array)-100]
	
	for i in range(0,xLen):
		for j in range(0,yLen):
			for k in range(0,zLen):
				if(imgNpFormat[i][j][k]==0):
					continue
				else:
					if(imgNpFormat[i][j][k]<lm):
						min_ele=1
						max_ele=lm
						try:
							imgNpFormat[i][j][k]= 1+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*rm)
						except ZeroDivisionError:
    							imgNpFormat[i][j][k]= 1
						
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255
					else:
						min_ele=lm
						max_ele=array[len(array)-10000]
						try:
							imgNpFormat[i][j][k]= rm+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*(float(255-rm)))
						except ZeroDivisionError:
							imgNpFormat[i][j][k]= rm
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255

	#Converting to Numpy Format 
	
	array = nib.Nifti1Image(np.array(imgNpFormat),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	nib.save(array, "norm"+name+"_t1ce.nii.gz")

	#Loading .nii.gz image
	epiImg = nib.load('../../../Brats_Training_Data/Test/'+name+"_t2.nii.gz") #TODO Change filename as required
	epiImgData = epiImg.get_data()
	array=[]
	
	imgNpFormat=np.array(epiImgData)
	xLen, yLen, zLen = imgNpFormat.shape

	array = imgNpFormat.flatten()

	array.sort()
	min_ele=array[1]
	max_ele=array[len(array)-100]
	
	for i in range(0,xLen):
		for j in range(0,yLen):
			for k in range(0,zLen):
				if(imgNpFormat[i][j][k]==0):
					continue
				else:
					if(imgNpFormat[i][j][k]<lm):
						min_ele=1
						max_ele=lm
						try:
							imgNpFormat[i][j][k]= 1+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*rm)
						except ZeroDivisionError:
    							imgNpFormat[i][j][k]= 1
						
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255
					else:
						min_ele=lm
						max_ele=array[len(array)-10000]
						try:
							imgNpFormat[i][j][k]= rm+(((float(imgNpFormat[i][j][k])-float(min_ele))/(float(max_ele)-float(min_ele)))*(float(255-rm)))
						except ZeroDivisionError:
							imgNpFormat[i][j][k]= rm
						if(imgNpFormat[i][j][k]>255):
							imgNpFormat[i][j][k]=255

	#Converting to Numpy Format 
	
	array = nib.Nifti1Image(np.array(imgNpFormat),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
	
	nib.save(array, "norm"+name+"_t2.nii.gz")
	
	
		
def get_train_names():
	train_names=[]
	for folder in glob.glob('../../../Brats_Training_Data/Test/*.gz'):
		temp=(folder.split('/')[-1].split('.')[0]).rsplit("_", 1)[0]
		if temp not in train_names:
			train_names.append(temp)
		
	
	return train_names	
	
def do():
	lm=0
	rm=0
	train_names=get_train_names()
	lm=np.load('lm.npy')
	with open('rm.txt', 'r') as f:
		rm = f.read()
		#print (lm)
	for i in range(0,len(train_names)):
		normalise(train_names[i],lm[i],float(rm))
		print(time.strftime('%a %H:%M:%S'))
		print("One_Image Done")
		
if __name__ == '__main__':
	do()
