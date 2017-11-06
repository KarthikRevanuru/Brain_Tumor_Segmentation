import time
print(time.strftime('%a %H:%M:%S'))
import numpy as np
import nibabel as nib
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

X=np.load('train_x.npy')
y=np.load('train_y.npy')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print ("splitting-done")
clf = RandomForestClassifier()
clf.fit(X,y)

print("model-fitting done")
print(time.strftime('%a %H:%M:%S'))
joblib.dump(clf, 'model.pkl')

clf = joblib.load('model.pkl')
pred=clf.predict(X)
print("testing done")
np.save('prediction', pred)

print (confusion_matrix(y, pred))
print(time.strftime('%a %H:%M:%S'))
