import cv2
import numpy as np
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd
from scipy.spatial.distance import cdist

def image_detect_blobs(img, kernal_v=6, show_fig=True) -> tuple:
    blur = cv2.GaussianBlur(img, (5, 5), 2)
    h, w = img.shape[:2]
    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernal_v, kernal_v))
    gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    lowerb = np.array([0, 0, 0])
    upperb = np.array([15, 15, 15])
    binary = cv2.inRange(gradient, lowerb, upperb)
    for row in range(h):
        if binary[row, 0] == 255:
            cv2.floodFill(binary, None, (0, row), 0)
        if binary[row, w-1] == 255:
            cv2.floodFill(binary, None, (w-1, row), 0)

    for col in range(w):
        if binary[0, col] == 255:
            cv2.floodFill(binary, None, (col, 0), 0)
        if binary[h-1, col] == 255:
            cv2.floodFill(binary, None, (col, h-1), 0)

    foreground = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    background = cv2.dilate(foreground, kernel, iterations=3)
    unknown = cv2.subtract(background, foreground)

    # Convert the image to grayscale
    gray = background

    background[background==255] = 5
    background[background==0] = 255
    background[background==5] = 0

    # Apply a threshold to the image to
    # separate the objects from the background
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find the contours of the objects in the image
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours and calculate the area of each object

    # figure=plt.figure()
    # fig, ax = plt.subplots()
    ellipses=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Draw a bounding box around each
        # object and display the area on the image
        x, y, w, h = cv2.boundingRect(cnt)
        xy, width_height, angle = cv2.fitEllipse(cnt)
        ellipses.append([xy[0], xy[1], width_height[0], width_height[1], angle])
        if show_fig:
            cv2.ellipse(img, (int(xy[0]), int(xy[1])), (int(width_height[0]/2), int(width_height[1]/2)), angle, 0, 360, (0, 255, 255), 2)
            cv2.ellipse(background, (int(xy[0]), int(xy[1])), (int(width_height[0]/2), int(width_height[1]/2)), angle, 0, 360, (100, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(img, str(area), (x, y+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.rectangle(background, (x, y), (x+w, y+h), (0, 0, 0), 2)
            cv2.putText(background, str(area), (x, y+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 144, 100), 2)
    if show_fig:
        # plt.subplot(1,2,1)
        # plt.imshow(img)
        # plt.subplot(1,2,2)
        # plt.imshow(background)
        plt.imshow(img)
    return ellipses


def image_detect_edges(img, show_fig=False, extention=None) -> tuple:

    # Display original image
   
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur= cv2.GaussianBlur(img_gray, (5,5), 0) 


    # ret, thresh = cv2.threshold(img_blur0, 12.5, 255, cv2.THRESH_BINARY)
    # img_blur = cv2.dilate(thresh, None, iterations = 1)
    
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=140, threshold2=160) # Canny Edge Detection


    # Display Canny Edge Detection Image

    return edges


def image_select_points(img, edges=False,x_nm=None, y_nm=None, len_nm=None, pixel=256, dist_thres=1, light_limit=[50, 200], absolute_pos_nm=False):
    '''selct points from detected edges for large image (such as 250 nm) and select points from dectected blobs for small image (such as 10 nm)''' 
    if (x_nm is None) or (y_nm is None):
        x_nm, y_nm = env.createc_controller.get_offset_nm()
    if len_nm is None:
        len_nm = env.createc_controller.im_size_nm

    plt.imshow(img, extent=[x_nm-len_nm/2, x_nm+len_nm/2, y_nm+len_nm, y_nm])
    
    if edges:
        edges_img=image_detect_edges(img)
        detect_mols=np.where(edges_img>0)
    else:
        detect_mols=np.where(img>light_limit[1])

    
    data_mols ={'x': detect_mols[1], 'y': detect_mols[0]}
    data_mols=pd.DataFrame(data_mols)
    data_mols=data_mols.drop_duplicates(ignore_index=True)

    selected_points=[]
    selected_points.append([data_mols.x[0], data_mols.y[0]])
    plt.scatter(x_nm-len_nm/2+len_nm/pixel*data_mols.x[0], y_nm+len_nm/pixel*data_mols.y[0])
    for i in tqdm(range(len(data_mols))):
        selected_points_array=np.array(selected_points)
        if np.sqrt((data_mols.x[i]-selected_points_array[:, 0])**2+(data_mols.y[i]-selected_points_array[:, 1])**2).min()*len_nm/pixel>dist_thres:
            selected_points.append([data_mols.x[i], data_mols.y[i]])
            plt.scatter(x_nm-len_nm/2+len_nm/pixel*data_mols.x[i], y_nm+len_nm/pixel*data_mols.y[i])


    plt.xlim(x_nm-len_nm/2, x_nm+len_nm/2)
    plt.ylim(y_nm+len_nm, y_nm)

    if absolute_pos_nm:
        selected_points_nm=[[selected_points[i][0]*len_nm/pixel+x_nm-len_nm/2, selected_points[i][1]*len_nm/pixel+y_nm] for i in range(len(selected_points))]
        return selected_points_nm
    else:
        return selected_points
    

def detect_indiv_mols(points, indiv_thres=2.5):
    '''select indivisal molecules from selected points (absolute position in points))'''
    diss_points=[]
    dist=cdist(np.array(points), np.array(points))
    for i in range(len(points)):
        minval = np.min(dist[i][np.nonzero(dist[i])])
        if minval>indiv_thres:
            diss_points.append(points[i])
    return diss_points

def detect_indiv_mol_center(img, limit_light=180, x_nm=None, y_nm=None, len_nm=None, pixel=128, absolute_pos_nm=False):
    center_x_nm=np.where(img>limit_light)[1].mean()
    center_y_nm=np.where(img>limit_light)[0].mean()
    if absolute_pos_nm:
        center_x_nm=x_nm-len_nm/2+len_nm/pixel*np.where(img>limit_light)[1].mean()
        center_y_nm=y_nm+len_nm/pixel*np.where(img>limit_light)[0].mean()
    return center_x_nm, center_y_nm
        