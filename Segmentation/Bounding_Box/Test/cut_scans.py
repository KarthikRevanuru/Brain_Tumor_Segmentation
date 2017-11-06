import time
print(time.strftime('%a %H:%M:%S'))
import nibabel as nib
import numpy as np

with open("MinMax.txt") as f:
    #f.readline()
    hashMap = {}
    imgNames = []
    for lines in f.readlines():
        [name,X_min,Y_min,Z_min,X_max,Y_max,Z_max] = lines.split()[:7]
        X_min,Y_min,Z_min,X_max,Y_max,Z_max = int(X_min),int(Y_min),int(Z_min),int(X_max),int(Y_max),int(Z_max)
        try:
            hashMap[name].append([X_min,Y_min,Z_min,X_max,Y_max,Z_max])
        except:
            hashMap[name] = [[X_min,Y_min,Z_min,X_max,Y_max,Z_max]]
            imgNames.append(name)
            
    
    location=[]            
    for name in imgNames:
        arrVal = hashMap[name]
        X_min,Y_min,Z_min,X_max,Y_max,Z_max=arrVal[0][0],arrVal[0][1],arrVal[0][2],arrVal[0][3],arrVal[0][4],arrVal[0][5]
        
        for i in range(X_min,X_max+1):
                for j in range(Y_min,Y_max+1):
                        for k in range(Z_min,Z_max+1):
                                location.append([i,j,k])
                
       	p = nib.load('../../../Normalised_Brats_Training_Data/Test/'+name+"_flair.nii.gz")
        p_data = np.array(p.get_data())
        tempx=[]
        for i in range(X_min,X_max+1):
                tempy=[]
                for j in range(Y_min,Y_max+1):
                        tempz=[]
                        for k in range(Z_min,Z_max+1):
                                tempz.append(p_data[i][j][k])
                        tempy.append(tempz)
                tempx.append(tempy)
                
        array = nib.Nifti1Image(np.array(tempx),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        nib.save(array, "cut"+name+"_flair.nii.gz")
        
        
        
        p = nib.load('../../../Normalised_Brats_Training_Data/Test/seg/'+name[4:]+"_seg.nii.gz")
        p_data = np.array(p.get_data())
        tempx=[]
        for i in range(X_min,X_max+1):
                tempy=[]
                for j in range(Y_min,Y_max+1):
                        tempz=[]
                        for k in range(Z_min,Z_max+1):
                                tempz.append(p_data[i][j][k])
                        tempy.append(tempz)
                tempx.append(tempy)
                
        array = nib.Nifti1Image(np.array(tempx),[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        nib.save(array, "cut"+name[4:]+"_seg.nii.gz")
        print ("One Scan Done")
	print(time.strftime('%a %H:%M:%S'))
        
        
    np.save('location', np.array(location))       
    print(time.strftime('%a %H:%M:%S'))      	
