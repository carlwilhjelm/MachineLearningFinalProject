
# Decision Tree Classification
# Importing the libraries

import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

datasetArr = sys.argv

# Importing the datasets 1, 2
# dataset_train = pd.read_table(datasetArr[1])
# dataset_train_labels = pd.read_table(datasetArr[2])
# dataset_test_data = pd.read_table(datasetArr[3])

#importing the datasets 4,5
dataset_train = pd.read_table(datasetArr[1], delim_whitespace=True, header=None)
dataset_train_labels = pd.read_table(datasetArr[2], delim_whitespace=True, header=None)
dataset_test_data = pd.read_table(datasetArr[3], delim_whitespace=True, header=None)

# uncomment for data set group 3
# dataset_test_data = pd.read_table(datasetArr[3], sep = ',')


dataset_train.replace(1.0000000000000001e+99, np.NaN, inplace=True)
dataset_test_data.replace(1.0000000000000001e+99, np.NaN, inplace=True)
#fill the missing values
imp = Imputer(strategy='median')

X = pd.DataFrame(imp.fit_transform(dataset_train))
y = dataset_train_labels.iloc[:, :].values
Z = pd.DataFrame(imp.fit_transform(dataset_test_data))

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

classifier.fit(X_train, y_train.ravel())
Z_test = sc.transform(Z)
Z_pred = classifier.predict(Z_test)

thefile = open(datasetArr[3][:len(datasetArr[3]) - 4] + '_RFresult_labels.txt', 'w')
for item in Z_pred:
    thefile.write("%s\n" % item)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

count = 0
sum = 0
for i, j in zip(y_pred, y_test):
    if i == j:
        count += 1
print(count / len(y_pred))


