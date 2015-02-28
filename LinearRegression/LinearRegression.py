__author__ = 'Rohith'

import sys
import numpy as np
import csv
from numpy.linalg import inv

def computeLinearRegression(trainFile,testFile):
    trainFileName = trainFile
    testFileName = testFile
    Ein, Eout= calculateMSE(trainFileName, testFileName, 'Ein')

    outputFileName = 'output'+trainFileName
    lengthEin = len(Ein)
    with open(outputFileName,'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['Ein','Eout'])
        for val in range(0,lengthEin):
            writer.writerow([Ein[val], Eout[val]])

def returnXYMatrix(fileName):
    rownum = 0
    xList = []
    yList = []
    inputFile  = open(fileName, "rU")
    reader = csv.reader(inputFile)
    for row in reader:
        if rownum is 0:
            header = row
            rownum += 1
        else:
            row.insert(0,1)
            xList.append(row[:len(row)-1])
            yList.append(row[len(row)-1:])

            rownum += 1

    inputFile.close()
    return xList, yList

def calculateMSE(trainFileName,testFileName,inputType):

    final_Ein = []
    final_Eout = []

    matrixX, matrixY = returnXYMatrix(trainFileName)
    matrixX_, matrixY_ = returnXYMatrix(testFileName)

    # Matrix form for X and Y to calculate Ein
    matrixX_Ein = np.asarray(matrixX, dtype="float")
    matrixY_Ein = np.asarray(matrixY, dtype="float")

    # Matrix form for X and Y to calculate Eout
    matrixX_Eout = np.asarray(matrixX_, dtype="float")
    matrixY_Eout = np.asarray(matrixY_, dtype="float")

    if inputType is 'Ein':
        transposeX = matrixX_Ein.transpose()
        resultMatrix = transposeX.dot(matrixX_Ein)
        xTransposeY = transposeX.dot(matrixY_Ein)

    # Now finding w for all lambda values ranging from 0 to 150
    for lamdaCheck in range(0,150):
        midResult = inv(resultMatrix + lamdaCheck*np.identity(len(resultMatrix))).dot(xTransposeY)
        w = np.asarray(midResult,dtype="float")

        # predicted Y* for the obtained w for train data
        predictedMatrixY_Ein = matrixX_Ein.dot(w)

        # predicted Y* for the obtained w for test data
        predictedMatrixY_Eout = matrixX_Eout.dot(w)

        # Computing E(w) for Ein
        mse_Ein = predictedMatrixY_Ein - matrixY_Ein
        finalMSE_Ein = np.transpose(mse_Ein).dot(mse_Ein)
        Ein = finalMSE_Ein[0]/(len(matrixX_Ein))
        final_Ein.append(Ein[0].tolist())

        # Computing E(w) for Eout
        mse_Eout = predictedMatrixY_Eout - matrixY_Eout
        finalMSE_Eout = np.transpose(mse_Eout).dot(mse_Eout)
        Eout = finalMSE_Eout[0]/(len(matrixX_Eout))
        final_Eout.append(Eout[0].tolist())

    return final_Ein,final_Eout

computeLinearRegression('train-1000-100.csv','test-1000-100.csv')
computeLinearRegression('train-100-100.csv','test-100-100.csv')
computeLinearRegression('train-100-10.csv','test-100-10.csv')
computeLinearRegression('50(1000)_100_train.csv','test-1000-100.csv')
computeLinearRegression('100(1000)_100_train.csv','test-1000-100.csv')
computeLinearRegression('150(1000)_100_train.csv','test-1000-100.csv')


# if __name__=="__main__":
#     computeLinearRegression(sys.argv[1:])
#     computeLinearRegression(sys.argv[2:])