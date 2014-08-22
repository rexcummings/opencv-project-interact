#!/usr/bin/python
"""
Hand tracking using depth sensor. Demonstration of the hand module.  

Author: Rex Cummings, Nathan Sprague
Version: 1.7, August 22, 2014
 
Note: Mouse Integration with hand tracking achieved through
      utilities from PyUserInput, specifically PyMouse. \
      Also, this will only work if OpenCV has been compiled 
      with OpenNI support. 
""" 
#Custom Import Statements
import hand 
import kalman 

#Import Statements
import cv2.cv as cv
import cv2
import numpy as np  
import math
import time
import mlpy
from pymouse import PyMouse

#Global Variable Declarations
CENTROID = None
VERTEX = None
TIME_ELAPSED = None  

def main():
    global CENTROID, VERTEX, TIME_ELAPSED

    #Variable Declarations  
    ii, jj, baseline, coverage = 0, 5, 0, 0.075
    zero, thresh = 0, 85            # mm
    time_check = 1.5                # seconds
    min_size, max_size = 550, 20000 #number of pixels
    begin, initial_roi, hand_detected, suc_click = False, True, False, False
    upper_right_corner = (600, 0) 
    depth_img_shape = (400, 450)
    small_img_shape = (400, 450, 3)
    pcl_roi_shape, depth_roi_shape = (220, 200, 3), (220, 200)
    pt1, pt2 = (525, 25), (550, 50)
    black, white = (0, 0, 0), (255, 255, 255)
    red, green, blue = (0, 0, 255), (0, 255, 0), (255, 0,0)
    orange, purple = (0, 128, 255), (153, 0, 153)     
    num_of_frames = 5
    avg_distances = np.ones(num_of_frames, dtype=np.int)    

    #Instantiate mouse object 
    m = PyMouse()

    #Create Kalman Filter 
    kf = movingKF()
        
    #Depth camera used
    capture_depth = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
    
    while True:
        t = time.clock()
        #print "Time: ", t 
        #Retrieve depth, color, and point cloud images
        capture_depth.grab()        
        success_depth, depth_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_DEPTH_MAP)
        success_color, color_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_BGR_IMAGE)
        success_pcl, pcl_img = capture_depth.retrieve(
            channel=cv2.cv.CV_CAP_OPENNI_POINT_CLOUD_MAP)

        #Flip images horizontally over y-axis
        depth_img = cv2.flip(depth_img, 1)
        color_img = cv2.flip(color_img, 1)
        pcl_img = cv2.flip(pcl_img,1) 
                
        #Manipulate and reshape images
        depth_img[depth_img==0] = 2**16 - 1
        converted_img = np.array(depth_img, dtype='float32')

        small_depth_img = np.zeros(depth_img_shape)
        small_color_img = np.zeros(small_img_shape)
        small_pcl_img = np.zeros(small_img_shape)

        small_depth_img = converted_img[15:465, 40:640] 
        small_color_img = color_img[15:465, 40:640]
        small_pcl_img = pcl_img[15:465, 40:640]
        trigger_box = small_depth_img[25:50, 525:550]
  
        """ Begin program by placing hand over orange box. """
        if not begin: 
            cv2.rectangle(small_color_img, pt1, pt2, orange, thickness=-1) #start box
            cv2.imshow("small_depth_img", 10 * (small_depth_img/(2**16-1)))
            cv2.imshow("small_color_img", small_color_img)            
            if ii < 10:
                baseline += np.sum(trigger_box)
                if ii == 9:
                    avg = baseline / 10
                    percent = avg * coverage
                    influence = avg + percent
                ii += 1            
            else:
                totalsum = np.sum(trigger_box)
                if totalsum > influence:
                    TIME_ELAPSED = t
                    begin = True
        else:
            """ Start Tracking """
            #Region of Interest (ROI)
            depth_roi = np.zeros(depth_roi_shape)
            pcl_roi = np.zeros(pcl_roi_shape)
            if initial_roi:
                dif_x, dif_y = depth_img_shape[0], zero
                depth_roi = small_depth_img[0:220, 400:600]         
                pcl_roi = small_pcl_img[0:220, 400:600]         
            else:                
                depth_roi, pcl_roi, dif_x, dif_y = update_roi(small_depth_img, small_pcl_img, 
                                                              depth_roi_shape, pcl_roi_shape)

            #Establish thresholded image based on a max threshold of 85mm 
            min_c, max_c, min_loc_c, max_loc_c = cv2.minMaxLoc(depth_roi)
            thresh_max = min_c + thresh    
            ret, thresh_img = cv2.threshold(depth_roi, thresh_max, thresh_max, 
                                            cv2.THRESH_BINARY_INV) 
            thresh8 = np.array(thresh_img, dtype='uint8')

            #Find Hand ROI 
            hand_list = hand.find_blobs(thresh8, min_size, max_size)
            #print "Blobs: ", len(hand_list)
            if(len(hand_list) > 0):
                hd = hand_list[0]
                hand_detected = True                            
            else:
                hand_detected = False
                print "No hand object detected.\n"
                       
            #Extract information from detected hand object
            while hand_detected:
                if initial_roi:
                    x_vertex, y_vertex = upper_right_corner
                    cx, cy = hd.centroid()
                    CENTROID = (cx + 400, cy)               
                else:
                    cx, cy = hd.centroid()                 
                    xf, yf = (cx + dif_x, cy + dif_y)   
                    CENTROID = xf, yf               
                x_vertex, y_vertex = hd.draw_ellipse(initial_roi, depth_roi, small_color_img, 
                                                     dif_x, dif_y, x_vertex, y_vertex, black)    

                #Set initial Kalman Filter state as (x, y, x_vel, y_vel)                
                if initial_roi:
                    kf.x[0] = x_vertex    
                    kf.x[1] = y_vertex
                    kf.x[2] = 0
                    kf.x[3] = 0                          
                """ Dynamic Kalman Measurements """
                z = (x_vertex, y_vertex)                
                #Prediction of new points using KF 
                kf.predict()                    
                #Corrections
                kf.correct(z)  
                #Display vertices and velocity vector
                coords = (int(kf.x[0]), int(kf.x[1]))
                vel_coords = (int(kf.x[0]) + int(5*(kf.x[2])), int(kf.x[1]) + int(5*(kf.x[3])))
                cv2.circle(small_color_img, coords, 5, orange, thickness=-1)
                cv2.line(small_color_img, coords, vel_coords, green, thickness=2)
                 
                """ Mouse Control """                
                #Mouse movement via hand                 
                vertex_speed = move_mouse(kf.x, m, small_color_img)  
                #Clicking                                 
                avg_depth = int(hd.mean_depth(thresh_img, x_vertex, y_vertex, zero))  
                avg_distances = avg_distances[1:]
                avg_distances = np.append(avg_distances, avg_depth)
                #print "Avg_distances: ", avg_distances
                t_gap = t - TIME_ELAPSED 
                if t_gap > time_check: 
                    pos_x, pos_y = m.position()  
                    #print "MOUSE POSITION: ", pos
                    suc_click = is_mouse_click(avg_distances, vertex_speed)              
                    if suc_click:
                        #m.click(pos_x, pos_y, 1)
                        print "Click"
                        jj = jj - 1
                    if jj < 5:
                        cv2.circle(small_color_img, z, 10, green, thickness=-1)
                        jj -= 1                        
                        if jj == 0:
                            jj = 5                
                initial_roi = False                                    
                hand_detected = False
                suc_click = False
            #Display desired images 
            cv2.imshow("small_color_img", small_color_img)
            #cv2.imshow("small_depth_img", 10 * (small_depth_img / (2**16-1)))
            #cv2.imshow("small_pcl_img", small_pcl_img)
            #cv2.imshow("thresh_img", thresh_img)

        #Wait only 1 ms before repeating loop.
        c = cv2.waitKey(1)

def move_mouse(kf_x, m, img):
    """
    move_mouse -- Moves mouse according to centroid of a Hand object.    

    Parameters:
       kf_x - tuple containing x, y, x_vel, and y_vel
       m - pymouse object 
       img - image of window to obtain dimensions from
    
    Returns:
       Speed of the vertex
    """ 
    exponent = 1.6
    x, y, x_vel, y_vel = (int(kf_x[0]), int(kf_x[1]), kf_x[2], kf_x[3])
    mx, my = m.position()
    win_height, win_width, channel = img.shape
    x_screen, y_screen = m.screen_size()
    min_x, max_x = 0, x_screen
    min_y, max_y = 0, y_screen   

    #Calculations
    speed = np.sqrt(x_vel**2 + y_vel**2)  
    power = math.pow(speed, exponent) 
    ratio = speed / power
    theta = math.atan2(y_vel, x_vel)        
    x_comp = power * math.cos(theta)    
    y_comp = power * math.sin(theta)        
    xf, yf = mx + x_comp, my + y_comp

    if xf < min_x:   
        xf = min_x
    elif xf > max_x: 
        xf = max_x
    elif yf < min_y: 
        yf = min_y
    elif yf > max_y: 
        yf = max_y
    m.move(xf, yf)
    return speed

def movingKF():
    """ Build and return a Kalman filter object for tracking
    a 2-D object moving at a fixed velocity """
    Phi =  np.array([[1., 0, 1, 0], # State transition matrix.
                     [0., 1, 0, 1],
                     [0., 0, 1, 0],
                     [0., 0, 0, 1]])
    C_v =  np.array([[.01, 0, 0, 0],  # Covariance of state transition noise.
                     [0., .01, 0, 0],
                     [0., 0, .01, 0],
                     [0., 0, 0, .01]])
    Lambda = np.array([[1., 0, 0, 0], 
                       [0., 1, 0, 0]])
    C_w =  np.array([[.1, 0],  # Covariance of sensor noise.
                     [0., .1]])
    x_0 = np.array([.1, .1, 0, 0]) # initial state estimate.
    P_0 =  np.eye(4) * 1. # initial Covariance estimate. 

    return kalman.KalmanFilter(Phi, C_v, Lambda, C_w, x_0, P_0)
    
def offset_x_dist(img, offset):
    diffs = np.zeros(img.shape)
    diffs[:, offset:, :] = img[:, offset:, :] - img[:, :-offset, :]
    diff_sqrd = diffs**2
    dist = np.sqrt(np.sum(diff_sqrd, axis=2))
    return dist * 1000 # in millimeters 

def offset_y_dist(img, offset):
    diffs = np.zeros(img.shape)
    diffs[offset:, :, :] = img[offset:, :, :] - img[:-offset, :, :]
    diff_sqrd = diffs**2
    dist = np.sqrt(np.sum(diff_sqrd, axis=2))
    return dist * 1000 

def update_roi(small_depth_img, small_pcl_img, depth_roi_shape, pcl_roi_shape):
    global CENTROID 
    x, y = CENTROID 
    w_min, h_min = 100, 110
    up = y
    down = 450 - y
    left = x
    right = 600 - x
    complete = False
    up_flag, down_flag, left_flag, right_flag = False, False, False, False
    diff_up, diff_down = abs(110 - up), abs(110 - down)  
    diff_left, diff_right = abs(100 - left), abs(100 - right) 
    
    while not complete:
        #Check corners, then sides
        if up <= 110 and not up_flag:
            p1_y = 0; p2_y = 220
            if left > 100 and right > 100:
                p2_x = x - 100; p1_x = x + 100
                left_flag = True; right_flag = True
            up_flag = True; down_flag = True
        elif right <= 100 and not right_flag:
            p1_x = 600; p2_x = 400        
            if up > 110 and down > 110:
                p1_y = y - 110; p2_y = y + 110
                up_flag = True; down_flag = True
            right_flag = True; left_flag = True         
        elif down <= 110 and not down_flag:
            p2_y = 450; p1_y = 230
            if left > 100 and right > 100:
                p2_x = x - 100; p1_x = x + 100
                left_flag = True; right_flag = True
            down_flag = True; up_flag = True
        elif left <= 100 and not left_flag:
            p2_x = 0; p1_x = 200
            if up > 110 and down > 110:
                p1_y = y - 110; p2_y = y + 110
                up_flag = True; down_flag = True
            left_flag = True; right_flag = True
        #Check center 
        elif up > 110 and not up_flag:
            p1_y = diff_up
            up_flag = True              
        elif right > 100 and not right_flag:         
            p1_x = diff_left + 200
            right_flag = True       
        elif down > 110 and not down_flag:
            p2_y = diff_up + 220
            down_flag = True       
        elif left > 100 and not left_flag:
            p2_x = diff_left
            left_flag = True                
        elif up_flag and down_flag and left_flag and right_flag:
            complete = True
    #Shift ROI appropriately  
    depth_roi = np.zeros(depth_roi_shape)
    pcl_roi = np.zeros(pcl_roi_shape)
    depth_roi = small_depth_img[p1_y:p2_y, p2_x:p1_x] 
    pcl_roi = small_pcl_img[p1_y:p2_y, p2_x:p1_x]      
    return depth_roi, pcl_roi, p2_x, p1_y

def is_mouse_click(dist, vertex_speed):
    """
    is_mouse_click -- Check if sequence of avg distances matches standard
                      click wave sequence based on dynamic time warp.   

    Parameters: 
       check_distances - list of current average depth distances
    Returns:
       True if distances match standard wave sequence, false if not.     
    """
    confirm = False
    v_0 = dist[0]
    vel_diff = np.array(dist.shape, dtype=np.int) 
    #Obtain velocity values    
    for j in range(len(dist)):
        vel_ch = dist[j] - v_0
        vel_diff = np.append(vel_diff, vel_ch) 
        #print "Vel change: ", vel_ch

    #Check velocity values
    for k in range (len(vel_diff)):
        value = abs(vel_diff[k])        
        if value > 60 and vertex_speed < 8:
            confirm = True
    #print "Vel_diff: ", vel_diff
    return confirm

if __name__ == "__main__":
    main()
