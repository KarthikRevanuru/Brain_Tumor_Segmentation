#Required Packages
import time
print(time.strftime('%a %H:%M:%S'))
import numpy as np #Version '1.13.0'
import nibabel as nib #Version '2.1.0'
import glob
import collections

def lm(name):
    #Loading .nii.gz image
    epiImg = nib.load(name) #TODO Change filename as required
    epiImgData = epiImg.get_data()
    array=[]
    
    imgNpFormat=np.array(epiImgData)
    array = imgNpFormat.flatten()
    counter = collections.Counter(array)
    ll = counter.most_common(5)
    #print (ll)
    return ll[1][0]
	
def list_names():
	names = glob.glob('../../../Brats_Training_Data/Test/*.nii.gz')
	max_list = []
	with open("lm.txt","w") as f:
		for name in names:
			max_list.append(lm(name))
			f.write("%s %f\n" %(name,max_list[-1]))
			#print (max_list[-1])
	f.close()
	np.save('lm', max_list)	
	'''#print (max_list)
	mean_val = sum(max_list)/len(max_list)
	#print (mean_val)
	with open("lm.txt","w") as f:
		f.write("%f\n" %(mean_val))
	f.close()'''

if __name__ == '__main__':
	list_names()
