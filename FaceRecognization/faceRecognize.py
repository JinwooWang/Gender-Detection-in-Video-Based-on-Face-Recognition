import time

import numpy
from PIL import Image

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

import CNN
import load_save_data

def DataTraining(learningRate = 0.05, epochs = 200, imageData = 'olivettifaces.gif',kernels = [8, 10], batchsize = 40):
    """
        This is the training function, kind of like relaxation step.
        First, give a number(here I choose 23333) to generate random number and load the data.
        Then construct the convolution neural network(It's a hard part, by debugging and debugging the parameters)
        After constructing, we will build three models, and use SGD in training model.
        Then training.
    """
    rng = numpy.random.RandomState(23333)
    imageData = load_save_data.loadImageData(imageData)
    trainX, trainY = imageData[0]
    validX, validY = imageData[1]
    testX, testY = imageData[2]
    numTrainBatches = trainX.get_value(borrow = True).shape[0] / batchsize
    numValidBatches = validX.get_value(borrow = True).shape[0] / batchsize
    numTestBatches = testX.get_value(borrow = True).shape[0] / batchsize

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    inputData = x.reshape((batchsize, 1, 57, 47))
    l0 = CNN.ConvPoolLayer(inputData = inputData, filterShape = (kernels[0], 1, 5, 5),
                          imageShape = (batchsize, 1, 57, 47), poolsize = (2, 2), rng = rng)
    # 26 = (57 - 5) / 2, 21 = (47 - 5) / 2
    l1 = CNN.ConvPoolLayer(inputData = l0.output, filterShape = (kernels[1], kernels[0], 5, 5),
                          imageShape = (batchsize, kernels[0], 26, 21), poolsize = (2, 2), rng = rng)
    # 11 = (26 - 5 + 1) / 2, 8 = (21 - 5) / 2
    l2 = CNN.HiddenLayer(inputData = l1.output.reshape((batchsize, kernels[1] * 11 * 8)),
                        numIn = kernels[1] * 11 * 8, numOut = 2000, activation = T.tanh, rng = rng)
    l3 = CNN.LogisticRegression(inputData = l2.output, numIn = 2000, numOut = 40)
    cost = l3.logLikelihood(y)
    """
        ================================================================================================================================
    """
    #Here I will give a brief introduction to how to build the model.
    #Using the given index, we can get the corresponding x and y(what given does).
    #Because l3.error() is applied, l2 - l1 - l0 were also applied. i.e. CNN is applied
    #In this way, model was built.
    testModel = theano.function([index], l3.error(y),
                                givens = {x: testX[index * batchsize : (index + 1) * batchsize], y: testY[index * batchsize : (index + 1) * batchsize]})
    validationModel = theano.function([index], l3.error(y),
                                      givens = {x: validX[index * batchsize : (index + 1) * batchsize], y: validY[index * batchsize : (index + 1) * batchsize]})
    #SGD(stochastic gradient decent)
    parameters = l0.parameters + l1.parameters + l2.parameters + l3.parameters
    grads = T.grad(cost, parameters)
    updates = [(p, p - learningRate * g) for p, g in zip(parameters, grads)]

    trainingModel = theano.function([index], cost, updates = updates,
                                    givens = {x: trainX[index * batchsize : (index + 1) * batchsize], y: trainY[index * batchsize : (index + 1) * batchsize]})

    """
        =================================================================================================================================
    """
    print 'training'
    ptc = 800
    ptcInc = 2
    threshold = 0.99
    #To ensure that always validate on validation data
    validationFrequency = min(numTrainBatches, ptc / 2)

    bstValidationL = numpy.inf
    bstIter = 0
    start = time.clock()

    epoch = 0
    done = False
    #Now it's time to train, really tricky.
    #Kind of like SVM, a epoch will go through all data set. For each while loop, we need to train
    #each batch by using the for loop. If the current loss is less than the best validation loss
    #we will update them, and if the current loss is less than the threshold x best validation loss
    #we will update the patience.
    while(epoch < epochs) and (not done):
        epoch += 1
        for ind in xrange(numTrainBatches):
            iter = (epoch - 1) * numTrainBatches + ind
            c = trainingModel(ind)
            if (iter + 1) % validationFrequency == 0:
                validationL = [validationModel(i) for i in xrange(numValidBatches)]
                currentL = numpy.mean(validationL)
                #print('epoch %i, validation error %f %%' % (epoch, currentL * 100.0))
                if currentL < bstValidationL:
                    if currentL < bstValidationL * threshold:
                        ptc = max(ptc, iter * ptcInc)
                    bstValidationL = currentL
                    bstIter = iter
                    load_save_data.saveData(l0.parameters, l1.parameters, l2.parameters, l3.parameters)
                    testL = [testModel(i) for i in xrange(numTestBatches)]
                    tstscore = numpy.mean(testL)
                    print(('epoch %i, model error: %f %%') % (epoch, tstscore * 100.0))
            if ptc <= iter:
                done = True
                break
    end = time.clock()
    print ('Optimization: ')
    print ('validation score: %f %% iter: %i' % (bstValidationL * 100.0, bstIter + 1))
    print ('time: %.2f ' % (end - start))

if __name__ == '__main__':
    DataTraining()
