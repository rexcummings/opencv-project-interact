#!/usr/bin/python
"""
Hand tracking using depth sensor. Demonstration of the hand module.  

Author: Rex Cummings, Nathan Sprague
Version: 1.6, August 21, 2014

Note: Mouse Integration with hand tracking achieved through
      utilities from PyUserInput, specifically PyMouse.  
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

def main():
    global CENTROID, VERTEX

    #Variable Declarations  
    i = 0
    baseline = 0
    thresh = 85
    depth_img_shape = (400, 450)
    small_img_shape = (400, 450, 3)
    pcl_roi_shape = (220, 200, 3)
    depth_roi_shape = (220, 200)
    pt1 = (525, 25)
    pt2 = (550, 50)
    black = (0, 0, 0)
    white = (255, 255, 255)
    red = (0, 0, 255)    
    blue = (255, 0,0)
    green = (0, 255, 0)
    orange = (0, 128, 255)    
    begin = False
    initial_roi = True
    hand_detected = False
    avg_distances = np.ones(26, dtype=np.int)    

    #Instantiate mouse object 
    m = PyMouse()

    #Create Kalman Filter 
    kf = movingKF()
        
    #Depth camera used
    capture_depth = cv2.VideoCapture(cv2.cv.CV_CAP_OPENNI)
    
    while True:
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
                
        #Manipulate and crop images
        depth_img[depth_img==0] = 2**16 - 1
        converted_img = np.array(depth_img, dtype='float32')

        small_depth_img = np.zeros(depth_img_shape)
        small_color_img = np.zeros(small_img_shape)
        small_pcl_img = np.zeros(small_img_shape)

        small_depth_img = converted_img[15:465, 40:640] 
        small_color_img = color_img[15:465, 40:640]
        small_pcl_img = pcl_img[15:465, 40:640]
        trigger_box = small_depth_img[25:50, 525:550]

        #Trigger program
        if begin == False: 
            cv2.rectangle(small_color_img, pt1, pt2, orange, thickness=-1)
            cv2.imshow("small_depth_img", 10 * (small_depth_img/(2**16-1)))
            cv2.imshow("small_color_img", small_color_img)            
            if i < 10:
                baseline += np.sum(trigger_box)
                if i == 9:
                    avg = baseline / 10
                    percent = avg * 0.075
                    influence = avg + percent
                i += 1            
            else:
                totalsum = np.sum(trigger_box)
                if totalsum > influence:
                    begin = True
        else:
            """ Start Tracking """
            t = time.clock()
            print "Time: ", t            

            #Region of Interest (ROI)
            depth_roi = np.zeros(depth_roi_shape)
            pcl_roi = np.zeros(pcl_roi_shape)
            if initial_roi == True:
                dif_x, dif_y = 400, 0
                depth_roi = small_depth_img[0:220, 400:600]         
                pcl_roi = small_pcl_img[0:220, 400:600]         
            else:                
                depth_roi, pcl_roi, dif_x, dif_y = update_roi(small_depth_img, small_pcl_img, depth_roi_shape, pcl_roi_shape)

            #Establish thresholded image based on a max threshold of 85mm 
            min_c, max_c, min_loc_c, max_loc_c = cv2.minMaxLoc(depth_roi)
            thresh_max = min_c + thresh    
            ret, thresh_img = cv2.threshold(depth_roi, thresh_max, thresh_max, cv2.THRESH_BINARY_INV) 
            thresh8 = np.array(thresh_img, dtype='uint8')

            #Find Hand ROI 
            hand_list = hand.find_blobs(thresh8, 550, 20000)
            print "Blobs: ", len(hand_list)
            if(len(hand_list) > 0):
                hd = hand_list[0]
                hand_detected = True                            
            else:
                hand_detected = False
                print "No hand object detected.\n"
                       
            #Extract information from detected hand object
            while hand_detected:
                if initial_roi:
                    x_vertex, y_vertex = 550, 100
                    cx, cy = hd.centroid()
                    CENTROID = (cx + 400, cy)               
                else:
                    cx, cy = hd.centroid()                 
                    xf, yf = (cx + dif_x, cy + dif_y)   
                    CENTROID = xf, yf               
                x_vertex, y_vertex = hd.draw_ellipse(initial_roi, depth_roi, small_color_img, dif_x, dif_y, x_vertex, y_vertex, [0,0,0])    

                #Set Kalman Filter                
                if initial_roi == True:
                    kf.x[0] = x_vertex
                    kf.x[1] = y_vertex      
                    print "Initial: ", kf.x
                   
                """ Dynamic Kalman Measurements """
                z = (x_vertex, y_vertex)
                
                #Prediction of new points using KF 
                kf.predict()                    
                #Corrections
                kf.correct(z)  
                
                coords = (int(kf.x[0]), int(kf.x[1]))
                vel_coords = (int(kf.x[0]) + int(5*(kf.x[2])), int(kf.x[1]) + int(5*(kf.x[3])))
                cv2.circle(small_color_img, coords, 5, orange, thickness=-1)
                cv2.line(small_color_img, coords, vel_coords, green, thickness=2)
                move_mouse(kf.x, m, small_color_img) 
                initial_roi = False                                    
                hand_detected = False
            #Display image versions
            #cv2.imshow("small_depth_img", 10 * (small_depth_img / (2**16-1)))
            cv2.imshow("small_color_img", small_color_img)
            #cv2.imshow("small_pcl_img", small_pcl_img)
            #cv2.imshow("thresh_img", thresh_img)
            #cv2.imshow("depth_roi", 10 * (depth_roi / (2**16-1)))
            #cv2.imshow("pcl_roi", pcl_roi)  

        #Wait only 1 ms before repeating loop.
        c = cv2.waitKey(1)

def move_mouse(kf_x, m, img):
    """
    move_mouse -- Moves mouse according to centroid of a Hand object.    

    Parameters:
       kf_x - tuple containing x, y, x_vel, and y_vel
       m - pymouse object 
       img - image of window to obtain dimensions from
    """ 
    exponent, thresh = 1.5, 1
    check_x, check_y, complete = False, False, False       
    x, y, x_vel, y_vel = (int(kf_x[0]), int(kf_x[1]), kf_x[2], kf_x[3])
    x_vel, y_vel = ((x_vel), (y_vel))
    print "MM x, y, xvel, yvel: ", x, y, x_vel, y_vel
    mx, my = m.position()
    x_scaled, y_scaled = 0,0
    print "before mx, my: ", mx, my
    win_bar = 25   
    win_height, win_width, channel = img.shape
    x_screen, y_screen = m.screen_size()
    x_screen, y_screen = float(x_screen), float(y_screen)
    min_x, max_x, max_x_roi = 1, x_screen, win_width
    min_y, max_y, max_y_roi = 1, y_screen, win_height
    x_ratio, y_ratio = (x_screen / float(win_width), (y_screen / float(win_height)))

       

    while complete == False:     
        if check_x == True and check_y == True:
            complete = True 
        elif x_vel == 0 and y_vel == 0:
            check_x, check_y = True, True
        elif x_vel == 0 and not check_x:
            check_x = True
        elif y_vel == 0 and not check_y:
            check_y = True
        elif x_vel > 0 and check_x == False:    
            x_scaled = x + math.pow(x_vel, exponent)
            check_x = True
        elif y_vel > 0 and check_y == False:
            y_scaled = y + math.pow(y_vel, exponent)
            check_y = True       
        elif x_vel < 0 and check_x == False: 
            x_vel = abs(x_vel)
            check_x = True
            x_scaled = -(x + math.pow(x_vel, exponent))
        elif y_vel < 0 and check_y == False: 
            y_vel = abs(y_vel)
            check_y = True
            y_scaled = -(y + math.pow(y_vel, exponent))   
    print "x_scaled: ", x_scaled
    print "y_scaled: ", y_scaled
    x_scaled, y_scaled = (abs(x_scaled), abs(y_scaled))  
        
        
        
    
    """    
    if x_scaled < min_x:       
        x_scaled = min_x
    elif x_scaled > max_x_roi: 
        x_scaled = max_x_roi
    elif y_scaled < min_y:     
        y_scaled = min_y
    elif y_scaled > max_y_roi: 
        y_scaled = max_y_roi

    print "x_scaled: ", x_scaled
    print "y_scaled: ", y_scaled
    """
    
      
    xf = abs(int(x_scaled * x_ratio)) 
    yf = abs(int(y_scaled * y_ratio))     
    print "xf: ", xf
    print "yf: ", yf  
    """    
    if xf < min_x:   
        xf = min_x
    elif xf > max_x: 
        xf = max_x
    elif yf < min_y: 
        yf = min_y
    elif yf > max_y: 
        yf = max_y
    """      
    m.move(xf, yf) 
    print "Mouse Pos: ", mx, my 
    print

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
    up = y
    down = 450 - y
    left = x
    right = 600 - x
    complete = False
    up_flag, down_flag, left_flag, right_flag = False, False, False, False
    diff_up, diff_down = abs(110 - up), abs(110 - down)  
    diff_left, diff_right = abs(100 - left), abs(100 - right) 
    
    while complete == False:
        #Check corners, then sides
        if up <= 110 and up_flag == False:
            p1_y = 0; p2_y = 220
            if left > 100 and right > 100:
                p2_x = x - 100; p1_x = x + 100
                left_flag = True; right_flag = True
            up_flag = True; down_flag = True
        elif right <= 100 and right_flag == False:
            p1_x = 600; p2_x = 400        
            if up > 110 and down > 110:
                p1_y = y - 110; p2_y = y + 110
                up_flag = True; down_flag = True
            right_flag = True; left_flag = True         
        elif down <= 110 and down_flag == False:
            p2_y = 450; p1_y = 230
            if left > 100 and right > 100:
                p2_x = x - 100; p1_x = x + 100
                left_flag = True; right_flag = True
            down_flag = True; up_flag = True
        elif left <= 100 and left_flag == False:
            p2_x = 0; p1_x = 200
            if up > 110 and down > 110:
                p1_y = y - 110; p2_y = y + 110
                up_flag = True; down_flag = True
            left_flag = True; right_flag = True
        #Check center 
        elif up > 110 and up_flag == False:
            p1_y = diff_up
            up_flag = True              
        elif right > 100 and right_flag == False:         
            p1_x = diff_left + 200
            right_flag = True       
        elif down > 110 and down_flag == False:
            p2_y = diff_up + 220
            down_flag = True       
        elif left > 100 and left_flag == False:
            p2_x = diff_left
            left_flag = True                
        elif up_flag == True and down_flag == True and left_flag == True and right_flag == True:
            complete = True
    #Shift ROI appropriately  
    depth_roi = np.zeros(depth_roi_shape)
    pcl_roi = np.zeros(pcl_roi_shape)
    depth_roi = small_depth_img[p1_y:p2_y, p2_x:p1_x] 
    pcl_roi = small_pcl_img[p1_y:p2_y, p2_x:p1_x]      
    return depth_roi, pcl_roi, p2_x, p1_y

def update_sa_img(sur_area_img, depth_roi, thresh8):
    over_break_pt = False
    sa_img = np.zeros(sur_area_img.shape, dtype=sur_area_img.dtype)
       
    while over_break_pt == False:
        min_d, max_d, min_loc, max_loc = cv2.minMaxLoc(depth_roi)
        x, y = min_loc
        loc = y, x
        a_val = sur_area_img.item(loc)
        d_val = depth_roi.item(loc)
        on_hand = thresh8.item(loc)  
        if on_hand > 0 and d_val != 0:
            if a_val > 1.1 or a_val < 10.0:
                sa_img.itemset(loc, a_val)
                depth_roi.itemset(loc, 0)
                sa_total = np.sum(sa_img)
                if sa_total > 10000:
                    over_break_pt = True
        else:
            depth_roi.itemset(loc, 0)       
    print "finished update"
    #return sa_img

def click_mouse(color_img, centroid, success_click, m):
    """
    click_mouse -- Click the mouse if the registered hand motion equates 
                   to a successful clicking gestures.  

    Parameters:
       success_click - true if gesture equals proper mouse clicking gesture, false otherwise  
       m - pymouse object  
    """
    left_click = 1
    middle_click = 2
    right_click = 3
    mx, my = m.position()  
    mpos = mx, my
    if success_click == True:
        cv2.circle(color_img, mpos, 10, (0, 255, 0), thickness=-1) 
        #m.click(mx, my, left_click)
        #Display confirm green circle for .2 seconds
        #c = cv2.waitKey(200)

    else:
        print "No mouse click registered. \n"    

def mouse_glide_to(x, y):
    """Smooth glides mouse from current position to point x,y with specific timing and speed"""
    #x and y are determined by previous centroid position 
    x1, y1 = m.position
    o_x, o_y, o_z = pcl_roi[y1, x1] 
    n_x, n_y, n_z = pcl_roi[y2, x2] 
    velocity = abs(sqrt((n_x-o_x)**2 + (n_y-o_y)**2 + (n_z-o_z)**2))    
    smooth_glide_mouse(x1,y1, x, y, velocity)
 
def smooth_glide_mouse(x1,y1,x2,y2, t, velocity):
    """Smoothly glides mouse from x1,y1, to x2,y2 in time t using intervals amount of intervals"""
    distance_x = x2-x1
    distance_y = y2-y1
    #need to debug and find proper intervals
    #use velocity to gauge distance traveled    
    for n in range(0, intervals+1):
        m.move(x1 + n * (distance_x/intervals), y1 + n * (distance_y/intervals))
        time.sleep(t*1.0/intervals)    

def is_mouse_click(check_distances):
    """
    is_mouse_click -- Check if sequence of avg distances matches standard
                      click wave sequence based on dynamic time warp.   

    Parameters: 
       check_distances - list of current average depth distances
    Returns:
       True if distances match standard wave sequence, false if not.     
    """
    confirm = False
    std_sequence = [198, 198, 199, 200, 201, 202,196, 179, 109, 58, 5, 3, 0, 6, 7, 12, 21, 29, 47, 81, 97, 152, 175, 193, 195, 195]

    wave_diff = mlpy.dtw_std(check_distances, std_sequence, dist_only=True)
    #print "wav diff: ", wave_diff
    if wave_diff < 1000:
        confirm = True
    return confirm

if __name__ == "__main__":
    main()
