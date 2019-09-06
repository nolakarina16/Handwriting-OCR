import sys
import numpy as np
import cv2
import os

# module level variables ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
###################################################################################################

class PanZoomWindow(object):
    """ Controls an OpenCV window. Registers a mouse listener so that:
        1. right-dragging up/down zooms in/out
        2. right-clicking re-centers
        3. trackbars scroll vertically and horizontally
    You can open multiple windows at once if you specify different window names.
    You can pass in an onLeftClickFunction, and when the user left-clicks, this
    will call onLeftClickFunction(y,x), with y,x in original image coordinates."""

    def __init__(self, img, window_name='PanZoomWindow', on_left_click_function=None):
        self.WINDOW_NAME = window_name
        self.H_TRACKBAR_NAME = 'x'
        self.V_TRACKBAR_NAME = 'y'
        self.img = img
        self.onLeftClickFunction = on_left_click_function
        self.TRACKBAR_TICKS = 1000
        self.panAndZoomState = PanAndZoomState(img.shape, self)
        self.lButtonDownLoc = None
        self.mButtonDownLoc = None
        self.rButtonDownLoc = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.redrawImage()
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)
        cv2.createTrackbar(self.H_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onHTrackbarMove)
        cv2.createTrackbar(self.V_TRACKBAR_NAME, self.WINDOW_NAME, 0, self.TRACKBAR_TICKS, self.onVTrackbarMove)

    def on_mouse(self, event, x, y, _ignore1, _ignore2):
        """ Responds to mouse events within the window.
        The x and y are pixel coordinates in the image currently being displayed.
        If the user has zoomed in, the image being displayed is a sub-region, so you'll need to
        add self.panAndZoomState.ul to get the coordinates in the full image."""
        if event == cv2.EVENT_MOUSEMOVE:
            return
        elif event == cv2.EVENT_RBUTTONDOWN:
            # record where the user started to right-drag
            self.mButtonDownLoc = np.array([y, x])
        elif event == cv2.EVENT_RBUTTONUP and self.mButtonDownLoc is not None:
            # the user just finished right-dragging
            dy = y - self.mButtonDownLoc[0]
            pixelsPerDoubling = 0.2 * self.panAndZoomState.shape[0]  # lower = zoom more
            changeFactor = (1.0 + abs(dy) / pixelsPerDoubling)
            changeFactor = min(max(1.0, changeFactor), 5.0)
            if changeFactor < 1.05:
                dy = 0  # this was a click, not a draw. So don't zoom, just re-center.
            if dy > 0:  # moved down, so zoom out.
                zoomInFactor = 1.0 / changeFactor
            else:
                zoomInFactor = changeFactor
            #            print "zoomFactor:",zoomFactor
            self.panAndZoomState.zoom(self.mButtonDownLoc[0], self.mButtonDownLoc[1], zoomInFactor)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # the user pressed the left button.
            coordsInDisplayedImage = np.array([y, x])
            if np.any(coordsInDisplayedImage < 0) or np.any(coordsInDisplayedImage > self.panAndZoomState.shape[:2]):
                print("you clicked outside the image area")
            else:
                # print("you clicked on", coordsInDisplayedImage, "within the zoomed rectangle")
                coordsInFullImage = self.panAndZoomState.ul + coordsInDisplayedImage
                # print("this is", coordsInFullImage, "in the actual image")
                # print("this pixel holds ", self.img[coordsInFullImage[0], coordsInFullImage[1]])
                if self.onLeftClickFunction is not None:
                    self.onLeftClickFunction(coordsInFullImage[0], coordsInFullImage[1])
        # you can handle other mouse click events here

    def onVTrackbarMove(self, tickPosition):
        self.panAndZoomState.setYFractionOffset(float(tickPosition) / self.TRACKBAR_TICKS)

    def onHTrackbarMove(self, tickPosition):
        self.panAndZoomState.setXFractionOffset(float(tickPosition) / self.TRACKBAR_TICKS)

    def redrawImage(self):
        pzs = self.panAndZoomState
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(self.WINDOW_NAME, self.img[pzs.ul[0]:pzs.ul[0] + pzs.shape[0], pzs.ul[1]:pzs.ul[1] + pzs.shape[1]])

class PanAndZoomState(object):
    """ Tracks the currently-shown rectangle of the image.
    Does the math to adjust this rectangle to pan and zoom."""
    MIN_SHAPE = np.array([50, 50])

    def __init__(self, imShape, parentWindow):
        self.ul = np.array([0, 0])  # upper left of the zoomed rectangle (expressed as y,x)
        self.imShape = np.array(imShape[0:2])
        self.shape = self.imShape  # current dimensions of rectangle
        self.parentWindow = parentWindow

    def zoom(self, relativeCy, relativeCx, zoomInFactor):
        self.shape = (self.shape.astype(np.float) / zoomInFactor).astype(np.int)
        # expands the view to a square shape if possible. (I don't know how to get the actual window aspect ratio)
        self.shape[:] = np.max(self.shape)
        self.shape = np.maximum(PanAndZoomState.MIN_SHAPE, self.shape)  # prevent zooming in too far
        c = self.ul + np.array([relativeCy, relativeCx])
        self.ul = c - self.shape / 2
        self._fixBoundsAndDraw()

    def _fixBoundsAndDraw(self):
        """ Ensures we didn't scroll/zoom outside the image.
        Then draws the currently-shown rectangle of the image."""
        #        print "in self.ul:",self.ul, "shape:",self.shape
        self.ul = np.maximum(0, np.minimum(self.ul, self.imShape - self.shape))
        self.shape = np.minimum(np.maximum(PanAndZoomState.MIN_SHAPE, self.shape), self.imShape - self.ul)
        #        print "out self.ul:",self.ul, "shape:",self.shape
        yFraction = float(self.ul[0]) / max(1, self.imShape[0] - self.shape[0])
        xFraction = float(self.ul[1]) / max(1, self.imShape[1] - self.shape[1])
        cv2.setTrackbarPos(self.parentWindow.H_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,
                           int(xFraction * self.parentWindow.TRACKBAR_TICKS))
        cv2.setTrackbarPos(self.parentWindow.V_TRACKBAR_NAME, self.parentWindow.WINDOW_NAME,
                           int(yFraction * self.parentWindow.TRACKBAR_TICKS))
        self.parentWindow.redrawImage()

    def setYAbsoluteOffset(self, yPixel):
        self.ul[0] = min(max(0, yPixel), self.imShape[0] - self.shape[0])
        self._fixBoundsAndDraw()

    def setXAbsoluteOffset(self, xPixel):
        self.ul[1] = min(max(0, xPixel), self.imShape[1] - self.shape[1])
        self._fixBoundsAndDraw()

    def setYFractionOffset(self, fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way down the image."""
        self.ul[0] = int(round((self.imShape[0] - self.shape[0]) * fraction))
        self._fixBoundsAndDraw()

    def setXFractionOffset(self, fraction):
        """ pans so the upper-left zoomed rectange is "fraction" of the way right on the image."""
        self.ul[1] = int(round((self.imShape[1] - self.shape[1]) * fraction))
        self._fixBoundsAndDraw()

def main():
    imgTrainingNumbers = cv2.imread(
        os.path.dirname(__file__) + "/static/images/hw2.png")  # read in training numbers image

    if imgTrainingNumbers is None:  # if image was not read successfully
        print("error: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,
                                      # invert so foreground will be white, background will be black
                                      11,  # size of a pixel neighborhood used to calculate threshold value
                                      2)  # constant subtracted from the mean or weighted mean

    cv2.imshow("imgThresh", imgThresh)      # show threshold image for reference

    imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                              # input image, make sure to use a copy since the function will modify this image in the course of finding contours

                                                              cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                              cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points

    # declare empty numpy array, we will use this to write to file later
    # zero rows, enough cols to hold all image data
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []  # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'), ord('a'), ord('b'), ord('c'), ord('d'),
                     ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'), ord('k'), ord('l'), ord('m'), ord('n'),
                     ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'), ord('u'), ord('v'), ord('w'), ord('x'),
                     ord('y'), ord('z'), ord('.'), ord(','), ord('?'), ord('!'), ord('?'), ord(',')]

    for npaContour in npaContours:  # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

            # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,  # draw rectangle on original training image
                          (intX, intY),  # upper left corner
                          (intX + intW, intY + intH),  # lower right corner
                          (0, 0, 255),  # red
                          2)  # thickness

            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH,
                                                RESIZED_IMAGE_HEIGHT))  # resize image, this will be more consistent for recognition and storage

            # cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference

            # cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            PanZoomWindow(imgTrainingNumbers, "training numbers.png")
            cv2.imshow("imgROIResized", imgROIResized)  # show resized image for reference

            intChar = cv2.waitKey(0)  # get key press

            if intChar == 226:
                # if shift key pressed, wait for another key
                intShiftChar = cv2.waitKey(0)
                if intShiftChar == 49:
                    intChar = int(33)
                if intShiftChar == 47:
                    intChar = int(63)

            if intChar == 27:  # if esc key was pressed
                sys.exit()  # exit program
            elif ((intChar in intValidChars) and (
            not intChar == 226)):  # else if the char is in the list of chars we are looking for . . .
                intClassifications.append(
                    intChar)  # append classification char to integer list of chars (we will convert to float later before writing to file)
                npaFlattenedImage = imgROIResized.reshape((1,
                                                           RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage,
                                               0)  # add current flattened impage numpy array to list of flattened image numpy arrays
            # end if
        # end if
        else:
            # Delete later, only for development
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect
            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]  # crop char out of threshold image
            cv2.imshow("imgROI failed", imgROI)
            # print('countor area: '+ str(cv2.contourArea(npaContour)))
    # end for

    fltClassifications = np.array(intClassifications,
                                  np.float32)  # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape(
        (fltClassifications.size, 1))  # flatten numpy array of floats to 1d so we can write to file later

    print("\n\ntraining complete !!\n")

    np.savetxt(os.path.dirname(__file__) + "/classifications.txt", npaClassifications)  # write flattened images to file
    np.savetxt(os.path.dirname(__file__) + "/flattened_images.txt", npaFlattenedImages)  #

    cv2.destroyAllWindows()  # remove windows from memory

    return


###################################################################################################
if __name__ == "__main__":
    main()
# end if
