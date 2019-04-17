import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv

#filterShape (filter num, number of input feature maps, filter height, filter width)
#imageShape (bach size, number of input feature maps, image height, image width)

class ConvPoolLayer(object):
    """
        Here we combine convolution layer and pooling layer as one "layer" called ConvPoolLayer.
        This will be more clear to see and write the code. rng is the abbreviation of "random number generator".
        You need to pay attention to the shape of input and the poolsize.
        inputDataShape (batchsize, #kernel, width, height)
    """
    def __init__(self, inputData, filterShape, imageShape, poolsize = (2, 2), weights = None, biases = None, rng = None):
        assert imageShape[1] == filterShape[1]
        #We need this assert statement, the program will go down if it returns True
        #By the way both imageShape[1] and filterShape[1] is the number of input feature maps.
        self.input = inputData
        #If weights is not passed by other parameters, we will initiate them.
        if weights == None:
            #Each neuron will connected to the maps of previous layer, so the number of connections should be
            # := #map x width x height
            ctnIn = numpy.prod(filterShape[1:])

            #As for output, it's kind of similar to the input, which the number of connections will be
            # := #filters x width x height
            ctnOut = (filterShape[0] * numpy.prod(filterShape[2:])) / numpy.prod(poolsize)

            #Using formula and get the weightBound of our initiation.
            #No matter what function you use(tanh or sigmoid or sth. else), its always suitable to use this bound.
            weightBound = numpy.sqrt(6.0 / (ctnIn + ctnOut))

            #As for compatiable with GPU, use theano.shared
            weights = theano.shared(numpy.asarray(rng.uniform(low = -weightBound, high = weightBound, size = filterShape),
                                        dtype = theano.config.floatX), borrow = True)
        #If biases is not passed by, than just initiate them into zero.
        if biases == None:
            biases = theano.shared(numpy.zeros((filterShape[0], ),
                                        dtype = theano.config.floatX), borrow = True)

        self.weights = weights
        self.biases = biases
        #Did not get the sigmoid and bias, it's a simplification.
        #But it works well.
        convOut = conv.conv2d(input = inputData, filters = self.weights, filter_shape = filterShape, image_shape = imageShape)
        pooledOut = pool.pool_2d(input = convOut, ds = poolsize, ignore_border = True)
        self.output = T.tanh(pooledOut + self.biases.dimshuffle('x', 0, 'x', 'x'))
        self.parameters = [self.weights, self.biases]

class HiddenLayer(object):
    """
        In HiddenLayer it's just one single layer, and because of it's fully meshed, the number of weights
        are #input x #output. Obviously, the number of biases is #output. rng is the same as the previous class.
        inputDataShape (#example, #neuron)
    """
    def __init__(self, inputData, numIn, numOut, weights = None, biases = None, activation = T.tanh, rng = None):
        self.input = inputData
        #The initiation of weights and baises is similar as before.
        if weights == None:
            #Here I need to declare that we are using (tanh) and it's uniform in interval section
            #     [-sqrt(6.0 / (#inConnections + #outConnections)), sqrt(6.0 / (#inConnections + #outConnections))]
            #Of course, if you choose the sigmoid, you should multiply 4 to this interval section.
            weightsBfSt = numpy.asarray(rng.uniform(low = -numpy.sqrt(6.0 / (numIn + numOut)), high = numpy.sqrt(6.0 / (numIn + numOut)),
                                        size = (numIn, numOut)), dtype = theano.config.floatX)
            if activation == T.nnet.sigmoid:
                weightsBfSt *= 4
            weights = theano.shared(value = weightsBfSt, name = 'weights', borrow = True)
        if biases == None:
            biasesBfSt = numpy.zeros((numOut, ), dtype = theano.config.floatX)
            biases = theano.shared(value = biasesBfSt, name = 'biases', borrow = True)
        self.weights = weights
        self.biases = biases
        #In HiddenLayer we calculate the output by inner product of matrix input and matrix weights, then add the biases to them.
        tmpOut = T.dot(self.input, self.weights) + self.biases
        self.output = (tmpOut if activation is None else activation(tmpOut))
        self.parameters = [self.weights, self.biases]

class LogisticRegression(object):
    """
        Although this layer is called LogisticRegression, you know, we will use softmax, and to a certain extend,
        it's reasonable to use this name. I won't say too much about softmax, it's the most basic machine learning
        algorithm, and we just choose the biggest possibility from the outputs.
        In this class I will add two small but useful function, to calculate logLikelihood and the errorRate.
        inputShape (#example, #neuron)
    """
    def __init__(self, inputData, numIn, numOut, weights = None, biases = None):
        if weights == None:
            weights = theano.shared(value = numpy.zeros((numIn, numOut), dtype = theano.config.floatX), name = 'weights', borrow = True)
        if biases == None:
            biases = theano.shared(value = numpy.zeros((numOut, ), dtype = theano.config.floatX), name = 'biases', borrow = True)
        self.weights = weights
        self.biases = biases
        #Using softmax to get the possibility.
        self.pGiven = T.nnet.softmax(T.dot(inputData, self.weights) + self.biases)
        #Attention! argmax will return the index instead of label.
        self.yRslt = T.argmax(self.pGiven, axis = 1)
        self.parameters = [self.weights, self.biases]
    def logLikelihood(self, y):
        return -T.mean(T.log(self.pGiven)[T.arange(y.shape[0]), y])
    def error(self, y):
        if y.ndim != self.yRslt.ndim:
            raise TypeError('TypeError!')
        if y.dtype.startswith('int'):
            #Here is an interesting function -neq, which will give you a list
            #for example, a = [1, 2, 3, 3] b = [1, 5, 5, 3]
            #Then neq(a, b) = [0, 1, 1, 0] "0" means equal, "1" means unequal
            return T.mean(T.neq(self.yRslt, y))
        else:
            raise NotImplementedError()
