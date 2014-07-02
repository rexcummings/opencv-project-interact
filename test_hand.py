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
import numpy as np  
import random
import time
import matplotlib.pyplot as plt
from pymouse import PyMouse
from pymouse import PyMouseEvent

def main():
    thresh = 85
    i = 0

    #Instantiate an mouse object
    #m = PyMouse()

    #Set up graph framework
    fig=plt.figure()
    plt.axis([0,210,500, 800])
    x=list()
    y=list()
    #ax.set_title("Average Depth vs. Time")
    #ax.set_xlabel("Time (ms)")
    #ax.set_ylabel("Depth (mm)") 

    #Enable graph integration
    plt.ion()
    #Create blank graph
    plt.show()      

    #Depth camera used
    capture_depth = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
    
    while True and i < 210:
        #Retrieve depth and color images
        capture_depth.grab()
        
        success_depth, depth_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_DEPTH_MAP)
        success_color, color_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_BGR_IMAGE)

        #Find minimum depth location and 2 inch buffer for threshold max
        depth_img[depth_img==0] = 2**16 - 1
        min_depth, max_depth, min_loc, max_loc = cv2.minMaxLoc(depth_img)

        #Create converted depth image (float32 dtype)   
        converted_img = np.array(depth_img, dtype='float32')
 
        #Establish threshold 
        min_c, max_c, min_loc_c, max_loc_c = cv2.minMaxLoc(converted_img)
        thresh_max = min_c + thresh
        #print 'min_c: ' + repr(min_c)
        #print 'threshold range: ' + repr(min_c) + ' to ' + repr(thresh_max)           
        ret, thresh_img = cv2.threshold(converted_img, thresh_max, thresh_max, cv2.THRESH_BINARY_INV) 
        thresh8 = np.array(thresh_img, dtype='uint8')

        #Obtain Hand ROI   
        blob_list = hand.find_blobs(thresh8, 550, 20000)
        print "Num Blobs: ", len(blob_list)  
        #print 'TIME: ' + repr(time.clock())
        for blob in blob_list:
            area = blob.area()
            print "area", area
            centroid = blob.centroid()
            print "Centroid Pose: ", centroid
            cv2.circle(color_img, centroid, 10, (0,0 ,255), thickness=-1)
            #blob.contour_outline(color_img,[random.randrange(0,256) for _ in range(3)])
            avg_depth = blob.mean_depth(thresh_img)     
            print 'Avg Depth: ' + repr(avg_depth)
            """
            #Control mouse with hand 
            pos = m.position()  
            print 'MOUSE POSITION: ' + repr(pos)
            cx, cy = centroid
            m.move(cx, cy)
            """   

            #Update graph
            if avg_depth > 450 and avg_depth < 800:
                x.append(i)
                y.append(avg_depth)
                plt.plot(x, y)
                i+=1
                plt.draw()
            if i == 125:
                plt.savefig('/home/rex/Pictures/plot1.png')
            if i == 150:
                plt.savefig('/home/rex/Pictures/plot2.png')
            if i == 175:
                plt.savefig('/home/rex/Pictures/plot3.png')
            if i == 200:
                plt.savefig('/home/rex/Pictures/plot4.png')
            
            time.sleep(0.25)
             
        #Display Images
        #cv2.imshow("depth", 10 * depth_img) 
        #cv2.imshow("converted_img", 10 * (converted_img/(2**16-1)))
        #cv2.imshow("thresh_img", thresh_img)
        cv2.imshow("color", color_img)
        c = cv2.waitKey(33)

if __name__ == "__main__":
    main()
