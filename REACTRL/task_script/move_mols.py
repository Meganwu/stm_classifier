
import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy.spatial.distance import cdist
from .collect_images import save_scan_image, tip_form_region


def seek_tip_move_pos(ref_x=0.0, ref_y=0.0, step_radius_nm=0.5, move_radius_nm=5.0,lat_mani_limit=[0, 5, 0, 5], time_limit=50, plot_move_points=True):
    """seek tip move position """

    for i in range(time_limit):
        tip_start_x=ref_x+step_radius_nm*(np.random.rand()*2-1)
        tip_start_y=ref_y+step_radius_nm*(np.random.rand()*2-1)
        tip_end_x=tip_start_x+move_radius_nm*(np.random.rand()*2-1)
        tip_end_y=tip_start_y+move_radius_nm*(np.random.rand()*2-1)

        if (tip_start_x-lat_mani_limit[0])*(lat_mani_limit[1]-tip_start_x)>0 and (tip_start_y-lat_mani_limit[2])*(lat_mani_limit[3]-tip_start_y)>0 and (tip_end_x-lat_mani_limit[0])*(lat_mani_limit[1]-tip_end_x)>0 and (tip_end_y-lat_mani_limit[2])*(lat_mani_limit[3]-tip_end_y)>0:
            break
    if i>=time_limit:
        print('fail not find tip start and tip end point within 50 times')

    if plot_move_points:
        plt.scatter([tip_start_x], [tip_start_y], s=1, c='r')
        plt.scatter([tip_end_x], [tip_end_y], s=1, c='b')
        plt.plot([tip_start_x, tip_end_x], [tip_start_y, tip_end_y], c='g', linewidth=1)
        plt.text(tip_start_x, tip_start_y, i, color='purple', fontsize=10)
    return tip_start_x, tip_start_y, tip_end_x, tip_end_y

def lat_mani_one_mol(
    env,
    episodes: int = 100,  # number of episodes for manipulating one molecule
    x_nm: float = 0.0, # the offset_x_nm of the scanning image with the molecule
    y_nm: float = 0.0, # the offset_y_nm of the scanning  image with the molecule
    scan_len_nm: int = 10, # the length of the scanning image with the molecule
    pixel: int = 256, # the pixel of the scanning image with the molecule
    step_radius_nm: float = 0.5,  # select tip_start_pos within a circle with radius of step_radius_nm around the center of the molecule
    move_radius_nm: int = 5, # select tip_end_pos within a circle with radius of move_radius_nm around the tip_start_pos
    total_output_folder: str = 'all_output',
    task_folder_prefix: str = 'latmani',
    save_scan_img_before_prefix: str = 'scan_img_before',
    save_scan_img_after_prefix: str = 'scan_img_after',
    save_scan_data_before_prefix: str = 'scan_data_before',
    save_scan_data_after_prefix: str = 'scan_data_after',
    save_lat_data_prefix: str = 'lat_data',
    save_lat_tip_pos_prefix: str = 'lat_tip_pos',
    time_limit: int = 50, # the maximum times for seeking tip_start_pos and tip_end_pos
    plot_move_points: bool = True,
    tip_form_ref_x: float = 280,
    tip_form_ref_y: float = -160,
    tip_form_z_range: list = [20, 35],
    tip_form_len_nm: float = 150,
    tip_form_dist_thres: float = 1,
    scan_default_z: float = 20,
    tip_form_check_points: list = None
) -> None:
    
    current_time=datetime.datetime.now()    
    task_folder='%s/%s_%s_%s_%s_%s' % (total_output_folder, task_folder_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_img_before='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_before_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_img_after='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_after_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_scan_data_before='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_before_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_scan_data_after='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_after_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_lat_data='%s/%s_%s_%s_%s_%s' % (task_folder, save_lat_data_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_lat_tip_pos='%s/%s_%s_%s_%s_%s' % (task_folder, save_lat_tip_pos_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
   
    # check if these folders exist
    if not os.path.exists(total_output_folder):
        os.mkdir(total_output_folder)
    if not os.path.exists(task_folder):
        os.mkdir(task_folder)
    if not os.path.exists(save_img_before):
        os.mkdir(save_img_before)
    if not os.path.exists(save_img_after):
        os.mkdir(save_img_after)
    if not os.path.exists(save_scan_data_before):
        os.mkdir(save_scan_data_before)
    if not os.path.exists(save_scan_data_after):
        os.mkdir(save_scan_data_after)
    if not os.path.exists(save_lat_data):
        os.mkdir(save_lat_data)
    if not os.path.exists(save_lat_tip_pos):
        os.mkdir(save_lat_tip_pos)
    

    for i in range(episodes):
        # scan image before manipulation
        print('Epoch %s scan a image before moving now....' % i)
        scan_data_before=save_scan_image(env, x_nm=x_nm, y_nm=y_nm, scan_len_nm=scan_len_nm, save_img_folder=save_img_before, save_data_folder=save_scan_data_before, img_name='%s' % (i), data_name='before_%s' % (i))
        # analyze image and detect molecules
        img=cv2.imread('%s/image_forward_%s.png' % (save_img_before, i))
        detect_mol=image_select_points(img, edges=False, dist_limit=1, x_nm=x_nm, y_nm=y_nm, len_nm=scan_len_nm, pixel=pixel, dist_thres=1, absolute_pos_nm=True)

        if len(detect_mol)>1:    # consider the center of two molecules as the reference point when somethimes it detects two points of one molecule
            x=(detect_mol[0][0]+detect_mol[1][0])/2
            y=(detect_mol[0][1]+detect_mol[1][1])/2
        else:
            x=detect_mol[0][0]
            y=detect_mol[0][1]

        # select tip_start_pos and tip_end_pos
        tip_start_x, tip_start_y, tip_end_x, tip_end_y=seek_tip_move_pos(ref_x=x, ref_y=y, step_radius_nm=step_radius_nm, move_radius_nm=move_radius_nm,lat_mani_limit=[x_nm-scan_len_nm/2, x_nm+scan_len_nm/2, y_nm, y_nm+scan_len_nm], time_limit=time_limit, plot_move_points=plot_move_points)
        lat_tip_pos=[x,y,tip_start_x, tip_start_y, tip_end_x, tip_end_y]
        with open(save_lat_tip_pos+'/lat_tip_pos_%s.pkl' % i, "wb") as fp:  
            pickle.dump(lat_tip_pos, fp) 

        # lateral manipulation
        print('Epoch %s move a molecule now....' % i)
        data_move=env.createc_controller.lat_manipulation(tip_start_x, tip_start_y, tip_end_x, tip_end_y, 35, 500, np.array([x_nm, y_nm]), scan_len_nm)
        with open(save_lat_data+'/lat_data_%s.pkl' % i, "wb") as fp:  
            pickle.dump(data_move, fp)

        # scan image after manipulation
        print('Epoch %s scan a image after moving now....' % i)
        scan_data_after=save_scan_image(env, x_nm=x_nm, y_nm=y_nm, scan_len_nm=scan_len_nm, save_img_folder=save_img_after, save_data_folder=save_scan_data_after, img_name='%s' % (i), data_name='after_%s' % (i))

        # tip forming
        print('Epoch %s form tip now....' % i)
        tip_form_check_points=tip_form_region(tip_form_ref_x=tip_form_ref_x, tip_form_ref_y=tip_form_ref_y, tip_form_z_range=tip_form_z_range, tip_form_len_nm=tip_form_len_nm, tip_form_dist_thres=tip_form_dist_thres, scan_default_z=scan_default_z, tip_form_check_points=tip_form_check_points)

    save_scan_image(env, x_nm=x_nm, y_nm=y_nm, scan_len_nm=scan_len_nm, save_img=False, save_data=False)

