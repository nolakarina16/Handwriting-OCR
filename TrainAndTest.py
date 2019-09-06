import cv2
import numpy as np
import operator
import os
import sys

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 50

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True

###################################################################################################
def main():
    allContoursWithData = []                # declare empty lists,
    validContoursWithData = []              # we will fill these shortly

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print ("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print ("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    kNearest = cv2.ml.KNearest_create()                   # instantiate KNN object

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = cv2.imread(str(sys.argv[1]))

    # imgTestingNumbers = cv2.imread(os.path.dirname(__file__)+"/static/images/"+'hamzen_zen.png')

    if imgTestingNumbers is None:                           # if image was not read successfully
        print ("error: image not read from file \n\n")      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)       # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                    # blur

                                                        # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()        # make a copy of the thresh image, this in necessary b/c findContours modifies the image
    # cv2.imshow("thresholded", imgThresh)
    # cv2.waitKey(0)

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data
    # end for

    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    # validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    arrayCharHeight = []       # array to store the height of ROI
    arrayCharWidth = []
    arrayChars = []

    arrayIntRectX = []
    old_nilaiX = 0

    for contourWithData in validContoursWithData: # getting mean distance between each characters
        nilaiX = contourWithData.intRectX
        if old_nilaiX == 0:
            old_nilaiX = nilaiX
        else:
            arrayIntRectX.append(abs(nilaiX - old_nilaiX))
            old_nilaiX = 0
        # endif
    # endfor
    # print(sum(arrayIntRectX))
    meanDistanceX = float(sum(arrayIntRectX))/float(len(arrayIntRectX))
    # meanDistanceX = max(arrayIntRectX)
    # print('mean: '+str(meanDistanceX) + '<br />')
    old_nilaiX = 0

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness
        # cv2.drawContours(imgTestingNumbers, npaContours, -1, (0, 0, 0), 2)
        # cv2.waitKey(0)
        # cv2.imwrite("imgTestingNumbers.png", imgTestingNumbers)

        arrayCharHeight.append(int(contourWithData.intRectY)) # append ROI height
        arrayCharWidth.append(int(contourWithData.intRectX))

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        # nilaiX = contourWithData.intRectX
        # if (nilaiX > 0): # Searching for space between chars
        #     distance = abs(nilaiX - old_nilaiX)
        #     old_nilaiX = nilaiX
        #     if distance >= meanDistanceX:
        #         strCurrentChar = " " + strCurrentChar

        arrayChars.append(strCurrentChar)
    # end for

    arrayCharHeight, arrayChars, arrayCharWidth = zip(*sorted(zip(arrayCharHeight, arrayChars, arrayCharWidth))) # Sort by char's height
    arrayCharData = np.vstack((arrayCharHeight, arrayChars, arrayCharWidth)).T # Combine arrayCharHeight, arrayChars, arrayCharWidth into one array

    # for a in arrayCharData:
    #     print(str(a[1]) + ': -height: ' + str(a[0]) + ' -width: ' + str(a[2]) + '  -sum: ' + str(int(a[0])+int(a[2])))

    prevHeight = 0
    prevWidth = 0
    charsOnSameLine = []
    charPosition = []
    charWidth = []
    chars = []
    index = 0
    for a in arrayCharData:
        if (abs(int(a[0]) - int(prevHeight)) > 20) and prevHeight != 0: # if height difference is too big(there's new line)
            charPosition, charsOnSameLine, charWidth = zip(*sorted(zip(charPosition, charsOnSameLine, charWidth))) # sort by chars positions
            chars = np.vstack((charPosition, charsOnSameLine, charWidth)).T
            for b in chars:
                if (abs(int(b[2]) - int(prevWidth)) > meanDistanceX):
                    # print(b[1])
                    # print(abs(int(b[2]) - int(prevWidth)))
                    strFinalString = strFinalString + ' ' + b[1] # add space
                else:
                    strFinalString = strFinalString + b[1]
                prevWidth = b[2]
            # endfor
            strFinalString = strFinalString + '<br />'

            # reset variables
            prevHeight = a[0]
            prevWidth = 0
            charPosition = []
            charsOnSameLine = []
            charWidth = []
            chars = []

            charsOnSameLine.append(a[1])
            charPosition.append(int(a[0]) + int(a[2]))
            charWidth.append(a[2])
        elif (index+1 ==len(arrayCharData)):
            charsOnSameLine.append(a[1])
            charPosition.append(int(a[0]) + int(a[2]))
            charWidth.append(a[2])

            charPosition, charsOnSameLine, charWidth = zip(*sorted(zip(charPosition, charsOnSameLine, charWidth)))  # sort by chars positions
            chars = np.vstack((charPosition, charsOnSameLine, charWidth)).T
            for b in chars:
                if (abs(int(b[2]) - int(prevWidth)) > meanDistanceX):
                    # print(b[1])
                    # print(abs(int(b[2]) - int(prevWidth)))
                    strFinalString = strFinalString + ' ' + b[1] # add space
                else:
                    strFinalString = strFinalString + b[1]
                prevWidth = b[2]
            # endfor
        else: # same line
            prevHeight = a[0]

            charsOnSameLine.append(a[1])
            charPosition.append(int(a[0])+int(a[2]))
            charWidth.append(a[2])
        # endif
        index+=1
    # endfor

    print(strFinalString)                  # show the full string

    # print ("\n" + printString + "\n")
    # cv2.imshow("imgTestingNumbers", imgTestingNumbers)      # show input image with green boxes drawn around found digits
    # cv2.waitKey(0)                                          # wait for user key press

    # cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if
