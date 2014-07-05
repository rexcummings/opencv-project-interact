"""
File:    hand.py

Purpose: Module containing utilities for identifying and tracking 
         a hand based on depth images using OpenCV and OpenNI.    

Authors: Rex Cummings, Nathan Sprague
Version: 1.0, July 2, 2014 

Credit:  Architecture based pyblob.py by Nathan Sprague.  
  
Note:    This will only work if OpenCV has been compiled 
         with OpenNI support.   
"""

import cv2
import numpy as np 

class Hand(object):
    """
    Hand -- Represents a region of interest (ROI) from a
            depth image.     
    """
    
    def __init__(self, contour):
        """
        Construct a Hand object from an identified 
        contour object.   
        """
        self.contour = contour

    def draw(self,img, color=(255,0,0)):
        """
        draw -- Draw a box around the ROI on a specified image.  
        """        
        cv2.drawContours(img, [self.contour], 0, color,
                        thickness=cv2.cv.CV_FILLED)
        rect = self.bounding_rectangle()
        cv2.rectangle(img, (rect[0], rect[1]), 
                     (rect[0] + rect[2], rect[1] + rect[3]),
                     (0,255,0), 2)

    def contour_outline(self, img, color=(255, 0, 0)):
        """
        Draw contour outline of the hand on a specified image.  
        """
        cv2.drawContours(img, [self.contour], 0, color,
                        thickness=cv2.cv.CV_FILLED)

    def mean_depth(self, img):
        """
        Returns mean depth of a contour object based on a mask.  
        """
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask, [self.contour], 0, 1, thickness=cv2.cv.CV_FILLED)
        masked_depth = mask * img  
        length = np.sum(mask) #Alternative can be cv2.countNonZero(mask)          
        total_depth = np.sum(masked_depth)
        avg_depth = float(total_depth/length)
        
        return avg_depth    

    def area(self):
        """
        Return number of pixels contained in the Hand.  
        """
        return abs(cv2.contourArea(self.contour))

    def bounding_rectangle(self):
        """
        Return bounding rectangle as tuple (x, y, width, height).  
        """
        return cv2.boundingRect(self.contour)

    def center(self):
        """
        Return the center of the blob as tuple (x, y).  
        """
        rect = self.bounding_rectangle()
        x = rect[0] + rect[2] // 2
        y = rect[1] + rect[3] // 2  
        return (x, y)
   
    def centroid(self):
        """
        Return the centroid of a blob as tuple (cx, cy)
        """
        M = cv2.moments(self.contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00']) 
        return (cx, cy)

    def __cmp__(self, other):
        """ 
        Default comparisons for blobs are based on area.
        """
        assert isinstance(other, Hand)
        if self is other:
            return 0
        elif self.area() < other.area():
            return -1
        elif self.area() > other.area():
            return 1
        elif id(self) < id(other):
            return -1
        else:
            return 1

def find_color_blobs(img, min_color, max_color, min_size, max_size):
    """  
    find_color_blobs -- Perform color blob finding in an image.   

    Parameters:
       img - the color image to search.
       min_color, max_color -Color values representing the color range
                             to search for.  For example in an RGB image,
                             min_color=(10, 20, 30), max_color=(100,100,100)
                             will search for regions of the image in which
                             the red value is between 10 and 100, the blue is
                             between 20 and 100 and the green is between 
                             30 and 100. 
       min_size, max_size   -Only return blobs between these two sizes. 

     Returns:
        A list of blob objects sorted from largest to smallest. 

    """
    color_mask = cv2.inRange(img, np.array(min_color), np.array(max_color))
    return find_blobs(color_mask, min_size, max_size)

def find_blobs(img, min_size, max_size):
    """ 
    find_blobs -- Find all connected compenents in an image and return 
                  them as blob objects.  

    Parameters:
       img - Single channel image.  All non-0 pixels will be treated as 1's. 
       min_size, max_size - Only return blobs between these two sizes. 

     Returns:
        A list of blob objects sorted from largest to smallest. 
    """
    contours,h = cv2.findContours (img, cv2.cv.CV_RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    ret_blobs = []
    for contour in contours:
        if min_size < cv2.contourArea(contour) < max_size :
            ret_blobs.append(Hand(contour))

    ret_blobs.sort(reverse=True)
    return ret_blobs
