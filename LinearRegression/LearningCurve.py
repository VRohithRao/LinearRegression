__author__ = 'Rohith'

import sys
import numpy as np
import csv
from numpy.linalg import inv
import random


def learningCurve(trainFile,testFile,lambdaValue):
    trainFileName = trainFile
    testFileName = testFile
    Ein, Eout= calculateMSE(trainFileName, testFileName, lambdaValue)
    writeOutput(Ein,Eout, lambdaValue)

def writeOutput(Ein, Eout, lambdaValue):
    outputFileName = 'Output-Lambda'+str(lambdaValue)+'.csv'
    lengthEin = len(Ein)
    with open(outputFileName, 'wb') as f:
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

def calculateMSE(trainFileName,testFileName,lambdaValue):

    final_Ein = []
    final_Eout = []
    initialList = []
    returnEin = []
    returnEout = []


    matrixX_, matrixY_ = returnXYMatrix(testFileName)

    # Matrix form for X and Y to calculate Eout
    matrixX_Eout = np.asarray(matrixX_, dtype="float")
    matrixY_Eout = np.asarray(matrixY_, dtype="float")

    rownum = 0
    inputFile = open(trainFileName, "rU")
    reader = csv.reader(inputFile)
    for row in reader:
        if rownum is 0:
            header = row
            rownum += 1
        else:
            row.insert(0,1)
            initialList.append(row)
            rownum += 1

    totalRows = 20
    while totalRows < 1000:

    # Now finding w for all lambda values ranging from 0 to 150
        for i in range(0,10):
            matrixX = []
            matrixY = []

            randomChoice = random.sample(initialList, totalRows)

            for randomRow in randomChoice:
                matrixX.append(randomRow[:len(row)-1])
                matrixY.append(randomRow[len(row)-1:])

            # Matrix form for X and Y to calculate Ein
            matrixX_Ein = np.asarray(matrixX, dtype="float")
            matrixY_Ein = np.asarray(matrixY, dtype="float")

            transposeX = matrixX_Ein.transpose()
            resultMatrix = transposeX.dot(matrixX_Ein)
            xTransposeY = transposeX.dot(matrixY_Ein)

            midResult = inv(resultMatrix + lambdaValue*np.identity(len(resultMatrix))).dot(xTransposeY)
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

        someValueEin = sum(final_Ein)/float(len(final_Ein))
        someValueEout = sum(final_Eout)/float(len(final_Eout))

        # print(someValueEin, " : ", someValueEout)

        returnEin.append(someValueEin)
        returnEout.append(someValueEout)
        totalRows += 20

    return returnEin,returnEout

learningCurve('train-1000-100.csv','test-1000-100.csv',1)
learningCurve('train-1000-100.csv','test-1000-100.csv',25)
learningCurve('train-1000-100.csv','test-1000-100.csv',150)




