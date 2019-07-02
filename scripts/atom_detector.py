#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError


from neural_network.srv import * 
from neural_network.msg import * 


import os
import sys
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import numpy as np
import cv2 as cv
import yaml
import time

import tensorflow as tf

from threading import Lock


bridge = CvBridge()
dirname = os.path.dirname(__file__)






#0 = left_side
#1 = right_side
side = 0




PATTERN_WIDTH = 32
PATTERN_HEIGHT = 32

colors = [
        (128, 128, 128),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (128, 255, 255)
        ]

map_points = np.float32([
    [0.550, 0.400, 0],
    [2.450, 0.400, 0],
    [2.450, 2.000, 0],
    [0.550, 2.000, 0],
    ])

atom_points_both_sides = (
# left_side
np.float32([
    [2.875, 0.000, 0.050],
    [2.775, 0.000, 0.050],
    [2.675, 0.000, 0.050],

    [2.500, 0.457, 0.050],
    [2.400, 0.457, 0.050],
    [2.300, 0.457, 0.050],
    [2.200, 0.457, 0.050],
    [2.100, 0.457, 0.050],
    [2.000, 0.457, 0.050],

    [2.500, 0.950, 0.010],
    [2.500, 1.250, 0.010],
    [2.500, 1.550, 0.010]
]),

# right_side
np.float32([
    [0.125, 0.000, 0.050],
    [0.225, 0.000, 0.050],
    [0.325, 0.000, 0.050],

    [0.500, 0.457, 0.050],
    [0.600, 0.457, 0.050],
    [0.700, 0.457, 0.050],
    [0.800, 0.457, 0.050],
    [0.900, 0.457, 0.050],
    [1.000, 0.457, 0.050],

    [0.500, 0.950, 0.010],
    [0.500, 1.250, 0.010],
    [0.500, 1.550, 0.010]
])
)

check_points_static_map = atom_points_both_sides[side]







def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    mu_g = np.average(nimg[1])
    nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
    nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
    return  nimg.transpose(1, 2, 0).astype(np.uint8)

# 
#     _________
#    /         \
#    |  R.I.P  |
#    |         |
#    |  CLAHA  |
#    |         |
#    |  2019-  |
#    |  2019   |
#   _|_________|_
#  |_____________|
#  

def Sobel_1_color(gray):
    ddepth = cv.CV_16S
    mix = 0.5

    grad_x = cv.Sobel(gray, ddepth = ddepth, dx = 1, dy = 0)
    grad_y = cv.Sobel(gray, ddepth = ddepth, dx = 0, dy = 1)
    grad_x_abs = cv.convertScaleAbs(grad_x)
    grad_y_abs = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(grad_x_abs, mix, grad_y_abs, mix, 0)

    return grad

def Sobel_3_colors(img):
    b,g,r = cv.split(img)
    mix = 0.7

    grad_r = Sobel_1_color(r)
    grad_g = Sobel_1_color(g)
    grad_b = Sobel_1_color(b)

    grad = cv.addWeighted(grad_r, mix, grad_g, mix, 0)
    grad = cv.addWeighted(grad, 2 * mix, grad_b, mix, 0)

    return grad

def cut_atom(img, pt):
    pt = np.int32(pt)
    return img[pt[1] - PATTERN_HEIGHT // 2: pt[1] + PATTERN_HEIGHT // 2, pt[0] - PATTERN_WIDTH // 2: pt[0] + PATTERN_WIDTH // 2]

def cut_atoms(img, points):
    imgs = []
    for pt in points:
        atom_img = cut_atom(img, pt)
        imgs.append(atom_img)
    return imgs


def mark_atoms(img_display, points, types):
    for pt, tp in zip(points, types):
        pt = np.int32(pt)
        cv.circle(img_display, tuple(pt), 3, colors[tp], 3)


def chaoz_find_circles(img_buf, chaoz_vertices):
    CHAOZ_WIDTH = 512
    CHAOZ_HEIGHT = 256
    CHAOZ_ATOM_MIN_RADIUS = 8
    CHAOZ_ATOM_MAX_RADIUS = 12
    chaoz_out = np.float32([[0,0], [CHAOZ_WIDTH, 0], [CHAOZ_WIDTH, CHAOZ_HEIGHT], [0, CHAOZ_HEIGHT]])
    chaoz_vertices = np.float32(chaoz_vertices)
    
    M = cv.getPerspectiveTransform(chaoz_vertices, chaoz_out)
    chaoz = cv.warpPerspective(img_buf, M, (CHAOZ_WIDTH, CHAOZ_HEIGHT))

    chaoz = cv.blur(chaoz, (3, 3))
    chaoz_s = Sobel_3_colors(chaoz)

    cv.imshow("Sobel", chaoz_s)

    circles = cv.HoughCircles(chaoz_s ,cv.HOUGH_GRADIENT,1,minDist=16,
                                param1=55,param2=17,minRadius=CHAOZ_ATOM_MIN_RADIUS,maxRadius=CHAOZ_ATOM_MAX_RADIUS)

    if circles is None:
        circles = np.zeros((1,1,2), np.float32)
    circles = circles[:,:,0:2]
    for i in circles[0]:
        i = tuple(np.uint16(i))
        R = (CHAOZ_ATOM_MIN_RADIUS + CHAOZ_ATOM_MAX_RADIUS) // 2
        cv.circle(chaoz,i,R,(128,0,255),2)
    
    cv.imshow("CHAOZ", chaoz)

    return cv.perspectiveTransform(circles, np.linalg.inv(M))[0]



yamlstr = open(os.path.join(dirname, "../params/camera_calibration.yml"), "rt").read()
calibr_params = yaml.load(yamlstr)
camera_matrix = calibr_params["camera_matrix"]["data"]
camera_matrix = np.float32(camera_matrix)
camera_matrix = np.reshape(camera_matrix, (3,3))

distCoeffs = calibr_params["distortion_coefficients"]["data"]
distCoeffs = np.float32(distCoeffs)
distCoeffs = np.reshape(distCoeffs, (5,))

h, w = 960, 1280
newmat, roi = cv.getOptimalNewCameraMatrix(camera_matrix,distCoeffs,(w,h),0.3,(w,h),1)

def correct_distortion(img):
    #h,  w = img.shape[:2]
    
    return cv.undistort(img, camera_matrix, distCoeffs, None, newmat)


def from_map_to_cam(points):
    CAM_TO_MAP_MAT_1 = np.vstack([CAM_TO_MAP_MAT, [0,0,0,1]])
    CAM_TO_MAP_MAT_1 = np.linalg.inv(CAM_TO_MAP_MAT_1)
    CAM_TO_MAP_MAT_1 = CAM_TO_MAP_MAT_1[0:3,:]

    points = np.float32(points).T
    points = np.vstack([points, np.ones((1, points.shape[1]))])
    points_cam = np.dot(CAM_TO_MAP_MAT_1, points)

    points_draw = cv.projectPoints(points_cam.T, (0,0,0), (0,0,0), camera_matrix, None)
    points_draw = points_draw[0][:,0,:]
    
    #points_cam /= points_cam[2]
    #points_cam = np.dot(newmat, points_cam)
    #points_draw = points_cam.T[:,0:2]
    #points_draw = np.float32(points_draw)

    return points_draw


def request_config():
    global CAM_TO_MAP_MAT
    global chaoz_vertices
    rospy.wait_for_service('ocam/config')
    getConfig = rospy.ServiceProxy('ocam/config', config)
    resp = getConfig()
    CAM_TO_MAP_MAT = yaml.load(resp.config)["trasnform"]["data"]
    CAM_TO_MAP_MAT = np.float32(CAM_TO_MAP_MAT)
    CAM_TO_MAP_MAT = np.reshape(CAM_TO_MAP_MAT, (3, 4))

    chaoz_vertices = from_map_to_cam(map_points)


def net_check_atoms(model, imgs):

    #blob = cv.dnn.blobFromImages(imgs, 1.0 / 255.0, (PATTERN_WIDTH, PATTERN_HEIGHT))
    #model.setInput(blob)
    #predictions = model.forward()
  
    imgs = np.float32(imgs) / 255.0
    predictions = model.predict(imgs)
    
    types = np.argmax(predictions, axis=1)
    return types




################################################


#model = cv.dnn.readNetFromTensorflow(os.path.join(dirname, "../params/puck_model.pb"))
model = tf.keras.models.load_model(os.path.join(dirname, "../params/puck_model.h5"))
model_lock = Lock()

request_config()
check_points_static = from_map_to_cam(check_points_static_map)

#all_img_buf = np.zeros((960, 1280, 3), dtype=np.uint8)

def preprocess(img):
    img = grey_world(img)
    img = correct_distortion(img)
    return img


def detect(all_img):
    #cv.copyTo(all_img, None, all_img_buf)
 
    static_atom_img_refs = cut_atoms(all_img, check_points_static)
    static_atom_types = net_check_atoms(model, static_atom_img_refs)

    chaoz_atom_points = chaoz_find_circles(all_img, chaoz_vertices)
    chaoz_atom_imgs = cut_atoms(all_img, chaoz_atom_points)
    chaoz_atom_types = net_check_atoms(model, chaoz_atom_imgs)

    return static_atom_types, chaoz_atom_points, chaoz_atom_types


################################################

   



def from_numpy_to_multiarray_2D(nparray):
    nparray = np.reshape(nparray, (nparray.shape[0], 2))
    dims = [
            std_msgs.msg.MultiArrayDimension("point", nparray.shape[0], nparray.shape[0] * nparray.shape[1]),
            std_msgs.msg.MultiArrayDimension("coord", nparray.shape[1], nparray.shape[1])
            ]
    layout = std_msgs.msg.MultiArrayLayout(dims, 0)
    array = std_msgs.msg.Float32MultiArray(layout, np.reshape(nparray, (nparray.shape[0] * nparray.shape[1], )))
    return array


def publish_data(static_atom_types, chaoz_atom_points, chaoz_atom_types):
    red_wall = []
    red_floor = []
    green_wall = []
    green_floor = []
    blue_wall = []
    blue_floor = []
   
    for point, typ in zip(check_points_static_map, static_atom_types):
        if (typ == 1):
            red_wall.append(point)
        elif (typ == 2):
            green_wall.append(point)
        elif (typ == 3):
            blue_wall.append(point)

    for point, typ in zip(chaoz_atom_points, chaoz_atom_types):
        if (typ == 1):
            red_floor.append(point)
        elif (typ == 2):
            green_floor.append(point)
        elif (typ == 3):
            blue_floor.append(point)

    red_wall = np.float32(red_wall) 
    red_floor = np.float32(red_floor) 
    green_wall = np.float32(green_wall) 
    green_floor = np.float32(green_floor) 
    blue_wall = np.float32(blue_wall) 
    blue_floor = np.float32(blue_floor) 

    red_wall = from_numpy_to_multiarray_2D(red_wall)
    red_floor = from_numpy_to_multiarray_2D(red_floor)
    green_wall = from_numpy_to_multiarray_2D(green_wall)
    green_floor = from_numpy_to_multiarray_2D(green_floor)
    blue_wall = from_numpy_to_multiarray_2D(blue_wall)
    blue_floor = from_numpy_to_multiarray_2D(blue_floor)

    pub.publish(red_wall, red_floor, green_wall, green_floor, blue_wall, blue_floor)
        
  
ocam_image = None
ocam_lock = Lock()

def callback(data):
    global ocam_image
    global ocam_lock

    ocam_lock.acquire()
    ocam_image = bridge.imgmsg_to_cv2(data, "bgr8")
    ocam_lock.release()

def process_and_send():
    if ocam_image is None:
        return

    start_time = time.clock()
    ######

    ocam_lock.acquire()
    all_image = preprocess(ocam_image)
    ocam_lock.release()

    static_atom_types, chaoz_atom_points, chaoz_atom_types = detect(all_image)

    ######
    end_time = time.clock()
    print(str(end_time - start_time) + "ms")

    img_display = all_image.copy()
    mark_atoms(img_display, check_points_static, static_atom_types)
    mark_atoms(img_display, chaoz_atom_points, chaoz_atom_types)

    cv.polylines(img_display, [np.int32(chaoz_vertices)], True, (0,255,0), 1)

    cv.imshow("All Image", cv.resize(img_display, (640, 480)))
    cv.waitKey(1)

    publish_data(static_atom_types, chaoz_atom_points, chaoz_atom_types)





def listener():
    global pub
    
    rospy.init_node('atom_detector', anonymous=True)
    sub = rospy.Subscriber("/ocam/image_raw", Image, callback)
    pub = rospy.Publisher('/atom_detect/atoms', atoms, queue_size=10)    

    rate = rospy.Rate(0.5) # 1hz
    while not rospy.is_shutdown():
        process_and_send()
        rate.sleep()

if __name__ == '__main__':
    listener()
