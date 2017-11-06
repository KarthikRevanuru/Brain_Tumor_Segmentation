import numpy as np
import nibabel as nib
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

#X=np.load('prediction.npy')
X=np.load('label.npy')
for x in X:
	print(x)
