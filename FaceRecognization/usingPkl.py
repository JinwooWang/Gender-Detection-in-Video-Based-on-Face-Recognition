import load_save_data
import CNN
import numpy
from PIL import Image, ImageDraw
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import matplotlib.pyplot as plt
import pylab

import os
import os.path
import cv2.cv as cv
import cv2
#opend the csv files
old_matrix = numpy.loadtxt(open("new.csv","rb"),delimiter=",",skiprows=0)
gender_matrix = numpy.loadtxt(open("gender.csv","rb"),delimiter=",",skiprows=0)

def ResizeImage(filein, fileout, width, height, type):
    """
        This function can resize your photo to any new size you want.
    """
    img = Image.open(filein)
    out = img.resize((width, height),Image.ANTIALIAS) #resize image with high-quality
    out.save(fileout, type)


def deal_with():
    """
        Resize the face you captured, and change the images to the grey one.
    """
    i = 0
    k = 0
    g = os.walk("captured")
    for path,d,filelist in g:
        for filename in filelist:
          ResizeImage(os.path.join(path, filename),r'deal/testout_%d.png'%(i),47, 57, 'png')
          i += 1
    
    g = os.walk("deal")
    for path,d,filelist in g:
        for filename in filelist:
          image_file = Image.open(os.path.join(path, filename)) # open colour image
          image_file = image_file.convert('L') # convert image to grey
          image_file.save(r'deal/testout_%d.png'%(k))
          k += 1
    



def capture():
    """
        Using the intel training set to capture the face in the video.
        Most of them are frameworks in OpenCV.
    """
    j = 0
    g = os.walk("origin")
    for path,d,filelist in g:
        for filename in filelist:
            img = cv.LoadImage(os.path.join(path, filename));
            image_size = cv.GetSize(img)
            greyscale = cv.CreateImage(image_size, 8, 1)
            cv.CvtColor(img, greyscale, cv.CV_BGR2GRAY)
            storage = cv.CreateMemStorage(0)

            cv.EqualizeHist(greyscale, greyscale)
            cascade = cv.Load('haarcascade_frontalface_alt2.xml')


            faces = cv.HaarDetectObjects(greyscale, cascade, storage, 1.2, 2,
                                       cv.CV_HAAR_DO_CANNY_PRUNING,
                                       (50, 50))


            for (x,y,w,h),n in faces:
              j+=1
              cv.SetImageROI(img,(x,y,w,h))
              cv.SaveImage("captured/face"+str(j)+".png",img);



def loading(path):
    """
        A very simple function, which can load the picture and put the label on it.
    """
    img = Image.open(path)
    imgArray = numpy.asarray(img, dtype = 'float64') / 256
    dataBase = numpy.empty((1, 57 * 47))
    """
    for i in range(20):
        for j in range(20):
            dataBase[i * 20 + j] = numpy.ndarray.flatten(imgArray[i * 57: (i + 1) * 57, j * 47:(j + 1) * 47])

    manLabel = numpy.empty(400)
    for i in range(40):
        manLabel[i * 10:(i + 1) * 10] = i

"""
    manLabel = numpy.empty(1)
    dataBase[0] = numpy.ndarray.flatten(imgArray[0:57, 0:47])
    manLabel[0] = 20
    return dataBase, manLabel

def recognizing(imageData, loadedFile = 'data.pkl', kernels = [8, 10]):
    """
        Load the data and build the CNN which have been trained.
        Then print the result.
    """
    dataBase, manLabel = loading(imageData)
    prm0, prm1, prm2, prm3 = load_save_data.loadData(loadedFile)
    m = T.matrix('m')
    inputData = m.reshape((dataBase.shape[0], 1, 57, 47))
    l0 = CNN.ConvPoolLayer(inputData = inputData, filterShape = (kernels[0], 1, 5, 5),
                            imageShape = (dataBase.shape[0], 1, 57, 47), poolsize = (2, 2),
                            weights = prm0[0], biases = prm0[1])
    l1 = CNN.ConvPoolLayer(inputData = l0.output, filterShape = (kernels[1], kernels[0], 5, 5),
                            imageShape = (dataBase.shape[0], kernels[0], 26, 21), poolsize = (2, 2),
                             weights = prm1[0], biases = prm1[1])
    l2 = CNN.HiddenLayer(inputData = l1.output.reshape((dataBase.shape[0], kernels[1] * 11 * 8)),
                            numIn = kernels[1] * 11 * 8, numOut = 2000, weights = prm2[0], biases = prm2[1],activation = T.tanh)
    l3 = CNN.LogisticRegression(inputData = l2.output, numIn = 2000, numOut = 40, weights = prm3[0], biases = prm3[1])

    func = theano.function([m], l3.yRslt)
    result = func(dataBase)
    """for ind in range(dataBase.shape[0]):
        if result[ind] != manLabel[ind]:
            print ('%i is mis-predicted as %i' % (manLabel[ind], result[ind]))
            img1 = dataBase[ind].reshape((57, 47))
            plt.subplot(121)
            plt.imshow(img1)
            pylab.gray()
            img2 = dataBase[result[ind] * 10].reshape((57, 47))
            plt.subplot(122)
            plt.imshow(img2)
            pylab.gray()
            pylab.show()

    """
    #first term is age and second term is gender
    if int(gender_matrix[result[0]]) == 0:
        gender = "male"
    else:
        gender = "female"
    print old_matrix[result[0]], ' ', gender


if __name__ == '__main__':
    g = os.walk("deal")
    capture()
    deal_with()
    for path,d,filelist in g:
        for filename in filelist:
            print filename, ':'
            recognizing(imageData = os.path.join(path, filename))
