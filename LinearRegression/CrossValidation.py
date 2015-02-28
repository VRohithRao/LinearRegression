__author__ = 'Rohith'

import numpy as np
import csv
from numpy.linalg import inv

def crossValidation(trainFile):
    matrixX, matrixY = returnXYMatrix(trainFile)
    dataFolds = len(matrixX)/10
    # print(dataFolds, ':', len(matrixX))
    initialRange = 0
    final_Eout = []
    averageEout = []
    lambdaToEout = []
    result = []

    outerBound = int(dataFolds)
    fixedBound = int(dataFolds)
    matrixX_Eout = []
    matrixY_Eout = []

    for lambdaValue in range(0,150):
        outerBound = fixedBound
        initialRange = 0
        for fold in range(1, 11):
            if fold != 1:
                initialRange = outerBound
                outerBound = fixedBound * fold
            matrixX_duplicate = matrixX[:]
            matrixY_duplicate = matrixY[:]

            matrixX_Eout_intermediate = matrixX_duplicate[initialRange:outerBound]
            matrixY_Eout_intermediate = matrixY_duplicate[initialRange:outerBound]

             # test fold
            matrixX_Eout = np.asarray(matrixX_Eout_intermediate, dtype="float")
            matrixY_Eout = np.asarray(matrixY_Eout_intermediate, dtype="float")

            # print("Range:", initialRange, outerBound)
            del matrixX_duplicate[initialRange:outerBound]
            del matrixY_duplicate[initialRange:outerBound]

            # train fold
            matrixX_Ein = np.asarray(matrixX_duplicate, dtype="float")
            matrixY_Ein = np.asarray(matrixY_duplicate, dtype="float")
            # print(matrixX_Ein.shape, matrixY_Ein.shape)

            transposeX = matrixX_Ein.transpose()
            resultMatrix = transposeX.dot(matrixX_Ein)
            xTransposeY = transposeX.dot(matrixY_Ein)

            midResult = inv(resultMatrix + lambdaValue*np.identity(len(resultMatrix))).dot(xTransposeY)
            w = np.asarray(midResult,dtype="float")

            predictedMatrixY_Eout = matrixX_Eout.dot(w)

            mse_Eout = predictedMatrixY_Eout - matrixY_Eout
            finalMSE_Eout = np.transpose(mse_Eout).dot(mse_Eout)
            Eout = finalMSE_Eout[0]/(len(matrixX_Eout))
            final_Eout.append(Eout[0].tolist())

        average = sum(final_Eout) / 10
        averageEout.append(average)

        lambdaToEout.append([averageEout[0],lambdaValue])
        averageEout = []
        final_Eout = []

    lengthEout = len(lambdaToEout)
    outputFileName = 'Output-CrossValidation-'+str(trainFile)
    with open(outputFileName, 'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['Eout','Corresponding Lambda'])
        for val in range(0,lengthEout):
            writer.writerow([lambdaToEout[val][0], lambdaToEout[val][1]])
    print("For given dataset the Eout value, best lambda value:",min(lambdaToEout))

    lambdaToEout = []


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


crossValidation('50(1000)_100_train.csv')
crossValidation('100(1000)_100_train.csv')
crossValidation('150(1000)_100_train.csv')
crossValidation('train-1000-100.csv')
crossValidation('train-100-100.csv')
crossValidation('train-100-10.csv')

