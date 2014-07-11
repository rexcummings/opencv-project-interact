#!/usr/bin/python
"""
Hand tracking using depth sensor. Demonstration of the hand module.  

Author: Rex Cummings
Version: 1.0, June 24, 2014

Note: Mouse Integration with hand tracking achieved through
      utilities from PyUserInput, specifically PyMouse.  
""" 
#Custom Import Statements
import hand 

#Import Statements
import cv
import cv2
import os
import os.path
import numpy as np  
import random
import time
import matplotlib.pyplot as plt
import mlpy
from pymouse import PyMouse
from pymouse import PyMouseEvent

def main():
    #Variable Declarations  
    thresh = 85
    title = ''
    data_Line = ''
    distances = np.ones(30, dtype=np.int)
    last = time.time()

    #Instantiate an mouse object 
    #m = PyMouse()
    
    #Remove file and create new template for the most recent data session
    filename = "/home/rex/opencv/projectCode/data/depth_data.txt"
    title = 'time(s)  avg_depth(mm)'
    if os.path.isfile(filename):
        os.remove(filename)        
        fn = open(filename, "a")
        fn.write(title + '\n')
        fn.close()
    
    #Depth camera used
    capture_depth = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
    
    while True:
        #Retrieve depth and color images
        capture_depth.grab()
        
        success_depth, depth_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_DEPTH_MAP)
        success_color, color_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_BGR_IMAGE)

        #Start time (seconds)
        t = time.clock()
    
        #Find minimum depth location and 2 inch buffer for threshold max
        depth_img[depth_img==0] = 2**16 - 1
        min_depth, max_depth, min_loc, max_loc = cv2.minMaxLoc(depth_img)

        #Create converted depth image having (float32 dtype)   
        converted_img = np.array(depth_img, dtype='float32')
 
        #Establish thresholded image 
        min_c, max_c, min_loc_c, max_loc_c = cv2.minMaxLoc(converted_img)
        thresh_max = min_c + thresh    
        ret, thresh_img = cv2.threshold(converted_img, thresh_max, thresh_max, cv2.THRESH_BINARY_INV) 
        thresh8 = np.array(thresh_img, dtype='uint8')

        #Find Hand ROI in thresholded depth image   
        hand_list = hand.find_blobs(thresh8, 550, 20000)
        print "Num Blobs: ", len(hand_list)  
        print "time: ", t
        """
        # See what size image is REALLY being captured (in case setting failed above.)
        HEIGHT = capture_depth.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        WIDTH = capture_depth.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        print "height: ", HEIGHT
        print "width:  ", WIDTH  
        """
        #Extract information from each detected hand object
        for hd in hand_list:
            area = hd.area()
            centroid = hd.centroid()
            hd.contour_outline(color_img, (255, 0,0))
            cv2.circle(color_img, centroid, 10, (0,0 ,255), thickness=-1)
            avg_depth = int(hd.mean_depth(thresh_img))     
            #print "area", area            
            print "Centroid Pose: ", centroid
            print 'Avg Depth: ' + repr(avg_depth)
            
            #Update array for most recent average distances
            distances = distances[1:]
            distances = np.append(distances, avg_depth)

            #Reformat depth values for proper Dynamic Time Warp checking
            
            min_avg = np.amin(distances)
            ones_arr = np.ones(distances.size, dtype=np.int)
            min_arr = ones_arr * min_avg
            updated_avg_distances = np.subtract(distances, min_arr)
            print updated_avg_distances            
            #Check if distances match standard click sequence  
            confirm = False
            confirm = is_mouse_click(updated_avg_distances)
            print "Sequences match: ", confirm
            
            #Format data line 
            data_Line = str(t) + ' ' + str(avg_depth) 
            #print data_Line

            #Write to file 
            data_file = open("/home/rex/opencv/projectCode/data/depth_data.txt", "a")
            data_file.write(data_Line + '\n')
            """
            #Control mouse with hand 
            pos = m.position()  
            print 'MOUSE POSITION: ' + repr(pos)
            cx, cy = centroid
            move_mouse(cx, cy, m)
            """                        
        #Display image versions
        #cv2.imshow("depth", 10 * depth_img) 
        #cv2.imshow("converted_img", 10 * (converted_img/(2**16-1)))
        #cv2.imshow("thresh_img", thresh_img)
        
        #Frame Rate
        """
        print "TIME:", 1./(time.time() - last)
        last = time.time()
        """
        cv2.imshow("color", color_img)

        #Wait 1 ms before repeating loop.
        c = cv2.waitKey(1)
    #Close file
    if os.path.exists("/home/rex/opencv/projectCode/data/depth_data.txt"):
        data_file.close()
    
def move_mouse(cx, cy, m):
    """
    move_mouse -- Moves mouse according to centroid of a Hand object 
                  and based on resolution ratios.    

    Parameters:
       xm - x-axis position of mouse
       ym - y-axis position of mouse 
       m - pymouse object 
    """
    x_ratio = 2.5
    y_ratio = 1.875
    
    MAKE DYNAMIC AND AVOID HARD CODING IN VALUES ITS BADDDDDD PRACTICE
    # get the screen size
m.screen_size()
# (1024, 768)


    xf = cx * x_ratio
    yf = cy * y_ratio
    m.move(xf, yf)

def is_mouse_click(updated_avg_distances):
    """
    is_mouse_click -- Check if sequence of avg distances matches standard
                      click wave sequence based on dynamic time warp.   

    Parameters: 
       updated_avg_distances - list of current distances
    Returns:
       True if distances match standard wave sequence, false if not.     
    """
    success = False
    std_sequence = [198, 198, 199, 200, 201, 202,196, 179, 109, 58, 5, 3, 0, 6, 7, 12, 21, 29, 47, 81, 97, 152, 175, 193, 195, 195]

    wave_diff = mlpy.dtw_std(updated_avg_distances, std_sequence, dist_only=True)
    print "wav diff: ", wave_diff
    if wave_diff > 5000:
        success = True
    return success

if __name__ == "__main__":
    main()



