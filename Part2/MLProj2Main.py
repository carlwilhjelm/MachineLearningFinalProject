# Importing the libraries
import sys
import numpy as np
import pandas as pd
import Pr.knn as knn
# Importing the dataset

dataset = pd.read_table(sys.argv[1], header=None)
finalData = dataset.copy()
dataShape = dataset.shape
rows = dataShape[0]
columns = dataShape[1]
knn.setRowCol(dataset)
knn.nanSubstitution(dataset)


# show missing values
missingVals = dataset.isnull()
# for each value in the dataset thats missing
for i in range(rows):
    for j in range(columns):
        if missingVals.iloc[i, j] == True:
            # create a new dataframe with no missing values,
            print("i=" + str(i) + " j=" + str(j))
            tempData = knn.cleanSubset(dataset, i, j)
            if(j == 8):
                print()
            # find the 3 nearest neighbors and return determine value of empty space with weighted mean
            finalData.iloc[i, j] = knn.weightedKNN(dataset, i, j, tempData[0], tempData[1])
            print(finalData.iloc[i,j])
            print()

finalData.to_csv(sys.argv[1][:len(sys.argv[1]) - 4] + '_results.txt', float_format='%.16f', header=False, index=False, mode='w')