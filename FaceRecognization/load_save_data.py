import cPickle
import numpy
from PIL import Image

import theano
import theano.tensor as T


def loadImageData(dataPath):
    """
        This function will load our original training data.
        Split the whole picture into pieces, the width and height can be recognized
        by file property. And we know that each small piece is 57 x 47.
        In this way we can load these 400 faces with labels. Here we will not only
        collect the training data, but also validation data and test data.
        Although the data is not that much, we can also get brillant result.
        When we finished to collect the data, we should also get them into the GPU,
        so that we can increase our speed of program.
        Lastly, return them.
    """
    img = Image.open(dataPath)
    imgDataArray = numpy.asarray(img, dtype = 'float64') / 256
    dataBase = numpy.empty((10 * 40, 57 * 47))
    for i in range(20):
        for j in range(20):
            dataBase[i * 20 + j] = numpy.ndarray.flatten(imgDataArray[57 * i : 57 * (i + 1), 47 * j : 47 * (j + 1)])
    manLabel = numpy.empty(10 * 40)

    for i in range(40):
        manLabel[i * 10 : (i + 1) * 10] = i


    trainingData = numpy.empty((8 * 40, 57 * 47))
    trainingLabel = numpy.empty(8 * 40)
    validationData = numpy.empty((40, 57 * 47))
    validationLabel = numpy.empty(40)
    testData = numpy.empty((40, 57 * 47))
    testLabel = numpy.empty(40)

    for i in range(40):
        trainingData[i * 8 : (i + 1) * 8] = dataBase[i * 10 : i * 10 + 8]
        trainingLabel[i * 8 : (i + 1) * 8] = manLabel[i * 10 : i * 10 + 8]
        validationData[i] = dataBase[i * 10 + 8]
        validationLabel[i] = manLabel[i * 10 + 8]
        testData[i] = dataBase[i * 10 + 9]
        testLabel[i] = manLabel[i * 10 + 9]

    #Get to GPU to be faster.
    trainingData_S = theano.shared(numpy.asarray(trainingData, dtype = theano.config.floatX), borrow = True)
    validationData_S = theano.shared(numpy.asarray(validationData, dtype = theano.config.floatX), borrow = True)
    testData_S = theano.shared(numpy.asarray(testData, dtype = theano.config.floatX), borrow = True)
    trainingLabel_S = T.cast(theano.shared(numpy.asarray(trainingLabel, dtype = theano.config.floatX), borrow = True), 'int32')
    validationLabel_S = T.cast(theano.shared(numpy.asarray(validationLabel, dtype = theano.config.floatX), borrow = True), 'int32')
    testLabel_S = T.cast(theano.shared(numpy.asarray(testLabel, dtype = theano.config.floatX), borrow = True), 'int32')

    return [(trainingData_S, trainingLabel_S), (validationData_S, validationLabel_S), (testData_S, testLabel_S)]

def saveData(prm0, prm1, prm2, prm3):
    """
        After training, we will save the relative parameters for later on.
    """
    infile = open('data.pkl', 'wb')
    cPickle.dump(prm0, infile, -1)
    cPickle.dump(prm1, infile, -1)
    cPickle.dump(prm2, infile, -1)
    cPickle.dump(prm3, infile, -1)
    infile.close()

def loadData(file):
    """
        Open the file and read the result which have been trained.
    """
    f = open(file, 'rb')
    prm0 = cPickle.load(f)
    prm1 = cPickle.load(f)
    prm2 = cPickle.load(f)
    prm3 = cPickle.load(f)
    f.close()
    return prm0, prm1, prm2, prm3
