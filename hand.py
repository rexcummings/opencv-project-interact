"""
File:    hand.py

Purpose: Module containing utilities for identifying and tracking 
         a hand based on depth images using OpenCV and OpenNI.    

Authors: Rex Cummings, Nathan Sprague
Version: 1.7, August 22, 2014 

Credit:  Architecture based on pyblob.py by Nathan Sprague.  
  
Note:    This will only work if OpenCV has been compiled 
         with OpenNI support.   
"""

import cv2
import math
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
        
    def fit_ellipse(self, color_img, x_i, y_i):
        """Finds the ellipse that best fits the pixels for this blob.  

        Return value is a three entry tuple representing a rotated
        rectangle that encloses the ellipse:

         ( (x, y), (width, height), angle ) 

        Where x and y are the center of the rectangle, and the angle
        is in degrees.
        """
        pixels = self.pixels(color_img, x_i, y_i)

        return cv2.fitEllipse(pixels) 

    def draw_ellipse(self, initial_roi, roi, color_img, x_i, y_i, x_vertex, y_vertex, color=(0,255,0)):
        """Draw ellipse surrounding the hand object.
        
        Returns the position of the vertex of the best fit ellipse.        
        """
        purple = (153, 0, 153)        
        box = self.fit_ellipse(color_img, x_i, y_i)
        ( (x, y), (width, height), angle ) = box
        h, w = (height / 2, width / 2)
        theta = math.radians(angle)
        
        #Top-right ellipse vertex            
        x_ch1 = h * math.sin(theta)
        y_ch1 = -h * math.cos(theta)
        x1_f, y1_f = (int(x+x_ch1), int(y+y_ch1))
        dist1 = math.sqrt((x_vertex - x1_f)**2 + (y_vertex - y1_f)**2)            
        #Bottom-left ellipse vertex            
        x_ch2 = -x_ch1
        y_ch2 = -y_ch1
        x2_f, y2_f = (int(x+x_ch2), int(y+y_ch2))
        dist2 = math.sqrt((x_vertex - x2_f)**2 + (y_vertex - y2_f)**2)   
        if dist1 < dist2:
            x_vertex, y_vertex = x1_f, y1_f                
        else:
            x_vertex, y_vertex = x2_f, y2_f   
        cv2.circle(color_img, (x_vertex, y_vertex), 5, purple, thickness=-1)                      
        cv2.ellipse(color_img, box, color, thickness=2)
        return x_vertex, y_vertex

    def pixels(self, color_img, x_i, y_i):
        """ 
        Returns an Nx2 numpy array of pixel coordinates 
        Note that the pixel positions are returned in (x, y) 
        order NOT (row, col) order. 
        """
        rect = cv2.boundingRect(self.contour)     
        occupied = np.zeros((rect[3],rect[2]),dtype=np.uint8)
        
        # Shift the pixels in the contour so that they lie inside the
        # newly created image. 
        shifted_contour = self.contour - (rect[0], rect[1])
            
        cv2.drawContours(occupied, [shifted_contour], 0, 255,
                         thickness=cv2.cv.CV_FILLED)
        pixels = np.nonzero(occupied)
        pixels = np.array(zip(pixels[1], pixels[0])) + (rect[0], rect[1]) + (x_i, y_i)
        return pixels

    def contour_outline(self, img, color=(255, 0, 0)):
        """
        Draw contour outline of the hand on a specified image.  
        """
        cv2.drawContours(img, [self.contour], 0, color,
                        thickness=cv2.cv.CV_FILLED)

    def mean_depth(self, img, x_vertex, y_vertex, zero):
        """
        Returns mean depth around vertex of a contour object's
        best fit ellipse.  
        """
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask, [self.contour], 0, 1, thickness=cv2.cv.CV_FILLED)
        masked_depth = mask * img  
        box_img = find_box_img(img, x_vertex, y_vertex, zero)
        length = np.sum(mask)          
        total_depth = np.sum(masked_depth)
        avg_depth = float(total_depth/length)        
        return avg_depth   

    def reconstruct_img(self, img, img_data):
        """
        reconstruct_img -- Construct new image from thresholded img and image data.

        Parameters:
           img - thresholded image of hand object
           img_data - new data to construct a new image

        Returns:
           A new image with shape of mask and data from img_data.  
        """
        mask = np.zeros(img.shape,np.uint8)
        cv2.drawContours(mask, [self.contour], 0, 1, thickness=cv2.cv.CV_FILLED)
        img_data[:,:,0] = mask * img_data[:,:,0] 
        img_data[:,:,1] = mask * img_data[:,:,1] 
        img_data[:,:,2] = mask * img_data[:,:,2] 
        return img_data

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

def find_box_img(img, x_vertex, y_vertex, zero):
    """
    Returns box image around ellipse vertex.
    """
    h, w = img.shape
    box_length = w / 3
    box_height = h / 5
    shift_y = box_height / 2
    shift_x = box_length / 2
    pt1_x, pt1_y = (x_vertex - shift_x), (y_vertex - shift_y) 
    pt2_x, pt2_y = (x_vertex + shift_x), (y_vertex + shift_y) 

    #Check dimensions
    if pt1_x < zero:
        pt1_x = zero 
    elif pt1_y < zero:
        pt1_y = zero
    elif pt2_x > w:
        pt2_x = w
    elif pt2_y > h:
        pt2_y = h
    box_img = img[pt1_y:pt2_y, pt1_x:pt2_x]    
    return box_img

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
       min_size, max_size - Only return blobs between min and max contour area. 

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
