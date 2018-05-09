import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

datasetArr = sys.argv

# Importing the datasets 1, 2
# dataset_train = pd.read_table(datasetArr[1])
# dataset_train_labels = pd.read_table(datasetArr[2])
# dataset_test_data = pd.read_table(datasetArr[3])

#importing the datasets 4,5
dataset_train = pd.read_table(datasetArr[1], delim_whitespace=True, header=None)
dataset_train_labels = pd.read_table(datasetArr[2], delim_whitespace=True, header=None)
dataset_test_data = pd.read_table(datasetArr[3], delim_whitespace=True, header=None)

# uncomment line below for data set group 3
# dataset_test_data = pd.read_table(datasetArr[3], sep = ',', header=None)

dataset_train.replace(1.0000000000000001e+99, np.NaN, inplace=True)
dataset_test_data.replace(1.0000000000000001e+99, np.NaN, inplace=True)
#fill the missing values
imp = Imputer(strategy='mean')

X = pd.DataFrame(imp.fit_transform(dataset_train))
y = dataset_train_labels.iloc[:, :].values
Z = pd.DataFrame(imp.fit_transform(dataset_test_data))



# Feature Scaling
sc = StandardScaler()
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_Ztrain = sc.fit_transform(X)
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_Ztrain,y.ravel())
Z_test = sc.transform(Z)
Z_pred = classifier.predict(Z_test)

thefile = open(datasetArr[3][:len(datasetArr[3]) - 4] + '_KNNresult_labels.txt', 'w')
for item in Z_pred:
    thefile.write("%s\n" % item)
# Fitting K-NN to the Training set
test_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
test_classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = test_classifier.predict(X_test)

count = 0
sum = 0
for i, j in zip(y_pred, y_test):
    if i == j:
        count += 1
print(count /len(y_pred))