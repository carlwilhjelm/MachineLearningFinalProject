# K-Nearest Neighbors (K-NN)

import numpy as np
import pandas as pd

def setRowCol(data):
    global rows, columns
    dataShape = data.shape
    rows = dataShape[0]
    columns = dataShape[1]


def nanSubstitution(data):
    for i in range(rows):
        for j in range(columns):
            if data.iloc[i, j] > 1.0e+98:
                data.iloc[i, j] = np.nan

def cleanSubset(data, targetRow, targetCol):
    # remove all columns where row being filled is otherwise missing values
    columnNumbers = [x for x in range(columns)]
    for i in range(columns):
        if i != targetCol and pd.isna(data.iloc[targetRow, i]):
            columnNumbers.remove(i)
    # data = data.iloc[:, columnNumbers]

    # remove all rows where values are missing
    rowNumbers = [x for x in range(rows)]
    for i in range(rows):
        for j in columnNumbers:
            if pd.isna(data.iloc[i,j]):
                rowNumbers.remove(i)
                break
    # data = data.iloc[rowNumbers, :]

    # testVals = data.isnull()
    return [rowNumbers, columnNumbers]


def weightedKNN(data, missingValRow, missingValCol, consideredRows, consideredCols):
    k = 2
    distance = [0] * len(consideredRows)
    i = 0
    # for each remaining row calculate distance
    consideredCols.remove(missingValCol)
    for row in consideredRows:
        for col in consideredCols:
            distance[i] += abs(data.iloc[row, col] - data.iloc[missingValRow, col])
        i += 1

    distanceCopy = list(distance)

    # if k = 3 1st 2nd and 3rd element will represent the distances of the 3 nearest neighbors
    distanceCopy.sort()
    # distanceCopy = list(distanceCopy[(len(distance) - len(consideredRows)):])

    neighborsIndex = [0] * k
    weightDistances = [0] * k
    weights = [0] * k
    # make a list of indexes, return the index from distance[] of the value from distanceCopy[]
    for kIndex in range(0, k):
        for index, value in enumerate(distance):
            if value == distanceCopy[kIndex]:
                neighborsIndex[kIndex] = consideredRows[index]
                weightDistances[kIndex] = value
                break


    print(consideredRows)
    print(distance)
    print(weightDistances)
    # make a list of weights
    weightDenom = 0
    for i in range(k):
        weightDenom += 1 / weightDistances[i]
    for i in range(k):
        weights[i] = (1 / weightDistances[i]) / weightDenom

    #sum values*weights
    result = 0
    for i in range(k):
        temp = (data.iloc[neighborsIndex[i], missingValCol]*weights[i])
        result += temp
    return result