#Required Packages
import time
print(time.strftime('%a %H:%M:%S'))
import numpy as np #Version '1.13.0'
import nibabel as nib #Version '2.1.0'
import glob
import collections
    
def lmp(name,mean):
	epiImg = nib.load(name) #TODO Change filename as required
	epiImgData = epiImg.get_data()
	array=[]
	
	imgNpFormat=np.array(epiImgData)
	xLen, yLen, zLen = imgNpFormat.shape

	array = imgNpFormat.flatten()

	array.sort()
	min_ele=1
	max_ele=array[len(array)-10000]
	#print (array)
	#print (min_ele,max_ele,mean)
	#print (1+(((float(mean)-float(min_ele))/(float(max_ele)-float(min_ele)))*255))
	return (1+(((float(mean)-float(min_ele))/(float(max_ele)-float(min_ele)))*255))
	
def list_names():
	names = glob.glob('../../../Brats_Training_Data/Test/*.nii.gz')
	lm=np.load('lm.npy')
	max_list = []
	with open("lmp.txt","w") as f:
		for i in range(0,len(names)):
			max_list.append(lmp(names[i],lm[i]))
			f.write("%s %f\n" %(names[i],max_list[-1]))
			#print (max_list[-1])
	f.close()
	np.save('lmp', max_list)	
	mean_val = sum(max_list)/len(max_list)
	#print (mean_val)
	with open("rm.txt","w") as f:
		f.write("%f\n" %(mean_val))
		
	f.close()

if __name__ == '__main__':
	list_names()
