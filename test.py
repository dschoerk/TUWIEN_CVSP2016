
#import pylab
import cv2
import imageio
#import matplotlib
#import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.color import rgb2hsv
from skimage.feature import hog
from skimage import data, color, exposure
import numpy

cv2.startWindowThread()
cv2.namedWindow("video")
cv2.namedWindow("fg")
cv2.namedWindow("color")
print(cv2.__version__)

fgbg = cv2.BackgroundSubtractorMOG2(50, 16, False)
#hog = cv2.HOGDescriptor()

filename = 'count-hard.mp4'
vid = imageio.get_reader(filename,  'ffmpeg')
nframes = vid._meta["nframes"]
nums = range(0, nframes)
for num in nums:
    image = vid.get_data(num)
    image_small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    image_intensity = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    image_hsv = rgb2hsv(image_small)
    fd, image_hog = hog(image_intensity, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
    #image_edge = cv2.Canny(numpy.uint8((1-image_hsv[:,:,1]) * 255), 300, 400)
    image_edge = cv2.Laplacian(numpy.uint8((1-image_hsv[:,:,1]) * 255), cv2.CV_8UC1)
    #print(image_hog)
    #image_hog_re = exposure.rescale_intensity(image_hog, in_range=(0, 0.02))
    #image_hog = hog.compute(image_intensity)

    #print(image_intensity.dtype)
    #print(image_hsv.dtype)
    #print(image_small.shape)
    #print(fd.shape)
    

    #tmp = cv2.convertScaleAbs(image_hsv[:,:,1])
    fgmask = fgbg.apply(image_intensity)
    fgmask = fgbg.apply(numpy.uint8((1-image_hsv[:,:,1]) * 255))
    kernel = numpy.ones((3,3),numpy.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_ERODE, kernel)

    #f, axarr = plt.subplots(3, 1, figsize=(20, 20))
    #axarr[0].imshow(image_hog)
    #axarr[1].imshow(image_small)
    #axarr[2].imshow(image_hsv)

    print(image_hog.shape)
    cv2.imshow("video", image_intensity) #
    cv2.imshow("fg", numpy.uint8((image_hsv[:,:,1]) * 255)) #
    cv2.imshow("color", image_edge) #
    
    #cv2.waitKey() 
    
    #fig = pylab.figure()
    #fig.suptitle('image #{}'.format(num), fontsize=20)
    #pylab.imshow(image_hog)
#pylab.show()

