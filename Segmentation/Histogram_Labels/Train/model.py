import numpy as np
import nibabel as nib
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

X=np.load('hist.npy')
y=np.load('label.npy')

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print ("splitting-done")
clf = RandomForestClassifier()
clf.fit(X,y)

print("model-fitting done")
joblib.dump(clf, 'model.pkl')

clf = joblib.load('model.pkl')
pred=clf.predict(X)
print("testing done")
np.save('prediction', pred)

print (confusion_matrix(y, pred))
