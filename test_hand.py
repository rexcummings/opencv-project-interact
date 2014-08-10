#!/usr/bin/python
"""
Hand tracking using depth sensor. Demonstration of the hand module.  

Author: Rex Cummings, Nathan Sprague
Version: 1.6, August 10, 2014

Note: Mouse Integration with hand tracking achieved through
      utilities from PyUserInput, specifically PyMouse.  
""" 
#Custom Import Statements
import hand 

#Import Statements
import cv
import cv2
import numpy as np  
import time
import mlpy
from pymouse import PyMouse
from pymouse import PyMouseEvent

#Global Variable Declarations
CENTROID = None

def main():
    global CENTROID

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
    orange = (0, 128, 255)    
    begin = False
    initial_roi = True
    hand_detected = False
    avg_distances = np.ones(26, dtype=np.int)    

    #Instantiate mouse object 
    m = PyMouse()
        
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

        #Trigger Program
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
            #Start main program
            t = time.clock()
            print "Time: ", t
            
            #Region of Interest (ROI)
            if initial_roi == True:
                depth_roi = np.zeros(depth_roi_shape)
                depth_roi = small_depth_img[0:220, 400:600]         
                pcl_roi = np.zeros(pcl_roi_shape)
                pcl_roi = small_pcl_img[0:220, 400:600]         
            else:
                depth_roi = np.zeros(depth_roi_shape)
                pcl_roi = np.zeros(pcl_roi_shape)
                depth_roi, pcl_roi = update_roi(small_depth_img, small_pcl_img, depth_roi_shape, pcl_roi_shape)

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
            while hand_detected == True:
                """
                cloud_roi = hd.reconstruct_img(thresh8, pcl_roi)
                cv2.imshow("cloud roi", 10 * cloud_roi) 
                cloud = cv2.GaussianBlur(cloud_roi, (5, 5), 0)
                sur_area_img = offset_x_dist(cloud, 1) * offset_y_dist(cloud, 1)
                
                print "Total SA: ", np.sum(sur_area_img) 

                update_sa_img(sur_area_img, depth_roi, thresh8)
                """
                if initial_roi == True:
                    CENTROID = hd.centroid()
                else:
                    CENTROID = hd.centroid()                        
                hd.contour_outline(color_img, blue) 
                cv2.circle(color_img, CENTROID, 10, red, thickness=-1) 
                """
                avg_depth = int(hd.mean_depth(thresh_img))  
                avg_distances = avg_distances[1:]
                avg_distances = np.append(avg_distances, avg_depth)
                if t > 2.00:
                    min_avg = np.amin(avg_distances)
                    check_distances = avg_distances - min_avg          
                    #Check if distances match standard click sequence   
                    success_click = is_mouse_click(check_distances)
                    """
                #Control mouse with hand 
                pos = m.position()  
                print 'MOUSE POSITION: ' + repr(pos)
                """
                cx, cy = centroid
                move_mouse(cx, cy, m, capture_depth)               
                click_mouse(color_img, centroid, success_click, m)
                """          
                initial_roi = False                                    
                hand_detected = False
            #Display image versions
            cv2.imshow("small_depth_img", 10 * (small_depth_img / (2**16-1)))
            cv2.imshow("small_color_img", small_color_img)
            cv2.imshow("small_pcl_img", small_pcl_img)
            cv2.imshow("thresh_img", thresh_img)
            cv2.imshow("depth_roi", 10 * (depth_roi / (2**16-1)))
            cv2.imshow("pcl_roi", pcl_roi)  

        #Wait only 1 ms before repeating loop.
        c = cv2.waitKey(1)
    
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
    diff_up, diff_down = 110 - up, 110 - down  
    diff_left, diff_right = 100 - left, 100 - right 
    while complete == False:
        if up <= 110 and up_flag == False:
            p1_y = 0
            up_flag = True
        elif up > 110 and up_flag == False:
            p1_y = y - (110 + diff_down)
            up_flag = True
        elif right <= 100 and right_flag == False:
            p1_x = 600        
            right_flag = True      
        elif right > 100 and right_flag == False:         
            p1_x = x + diff_left + 100
            right_flag = True        
        elif down <= 110 and down_flag == False:
            p2_y = 450
            down_flag = True
        elif down > 110 and down_flag == False:
            p2_y = y + diff_up + 110
            down_flag = True
        elif left <= 100 and left_flag == False:
            p2_x = 0
            left_flag = True
        elif left > 100 and left_flag == False:
            p2_x = x - (100 + diff_right)
            left_flag = True
        elif up_flag == True and down_flag == True and left_flag == True and right_flag == True:
            complete = True
    #Shift ROI appropriately  
    depth_roi = np.zeros(depth_roi_shape)
    pcl_roi = np.zeros(pcl_roi_shape)
    depth_roi = small_depth_img[p1_y:p2_y, p2_x:p1_x] 
    pcl_roi = small_pcl_img[p1_y:p2_y, p2_x:p1_x]        
    return depth_roi, pcl_roi

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
    
def move_mouse(cx, cy, m, capture_depth):
    """
    move_mouse -- Moves mouse according to centroid of a Hand object 
                  and based on resolution ratios.    

    Parameters:
       xm - x-axis position of mouse
       ym - y-axis position of mouse 
       m - pymouse object 
       capture_depth - depth camera to obtain dimensions from
    """
    x_screen, y_screen = m.screen_size()
        
    cam_height = capture_depth.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    cam_width = capture_depth.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    #print "cam h: ", cam_height
    #print "cam w: ", cam_width
    xf = cx * (x_screen / cam_height) 
    yf = cy * (y_screen / cam_width)
    m.move(xf, yf)

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
