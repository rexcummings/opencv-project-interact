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
from pymouse import PyMouse
from pymouse import PyMouseEvent

def main():
    #Variable Declarations
    thresh = 85
    title = ''
    data_Line = ''

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

        #Extract information from each detected hand object
        for hd in hand_list:
            area = hd.area()
            centroid = hd.centroid()
            cv2.circle(color_img, centroid, 10, (0,0 ,255), thickness=-1)
            avg_depth = hd.mean_depth(thresh_img)     
            print "area", area            
            print "Centroid Pose: ", centroid
            print 'Avg Depth: ' + repr(avg_depth)

            #Format data line 
            data_Line = str(t) + ' ' + str(avg_depth) 
            print data_Line

            #Write to file 
            data_file = open("/home/rex/opencv/projectCode/data/depth_data.txt", "a")
            data_file.write(data_Line + '\n')
            time.sleep(0.033)

            """
            #Control mouse with hand 
            pos = m.position()  
            print 'MOUSE POSITION: ' + repr(pos)
            cx, cy = centroid
            m.move(cx, cy)
            """                 
       
        #Display image versions
        #cv2.imshow("depth", 10 * depth_img) 
        #cv2.imshow("converted_img", 10 * (converted_img/(2**16-1)))
        #cv2.imshow("thresh_img", thresh_img)
        cv2.imshow("color", color_img)

        #Wait 33 ms before repeating loop.
        c = cv2.waitKey(33)
    #Close file
    if os.path.exists("/home/rex/opencv/projectCode/data/depth_data.txt"):
        data_file.close()

if __name__ == "__main__":
    main()
