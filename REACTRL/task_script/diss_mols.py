import os
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy.spatial.distance import cdist
from .collect_images import save_scan_image, tip_form_region
from REACTRL.env_modules.image_module_ellipse import image_detect_blobs, image_detect_edges, image_select_points, detect_indiv_mols, detect_indiv_mol_center


def vert_mani_mols(
        env,
        diss_area: list = [0, 300, 0, 300],  # the left, right, bottom and up of the region for dissociating molecules
        scan_len_nm_large: int = 10,  # the length of the scanning image
        pixel_large: int = 256,  # the pixel of the scanning image
        scan_len_nm_small: int = 3.5,  # the length of the scanning image with individual molecule
        pixel_small: int = 256,  # the pixel of the scanning image with individual molecule
        diss_radius_nm: float = 1.0,  # select tip_diss_pos within a circle with radius of diss_radius_nm around the center of the molecule
        diss_z_nm: float = 20,  # the z_nm of tip_diss_pos
        diss_mvoltage: float = 3500,  # the mvoltage for dissociation
        diss_pcurrent: float = 1000,  # the pcurrent for dissociation
        diss_maxtime: float = 8,  # the maxtime for dissociation
        # time_limit: int = 50,  # the maximum times for seeking tip_diss_pos 
        total_output_folder: str = 'all_output',
        task_folder_prefix: str = 'vertmani',
        save_scan_img_large_prefix: str = 'scan_img_large',
        save_scan_img_before_prefix: str = 'scan_img_before',
        save_scan_img_after_prefix: str = 'scan_img_after',
        save_scan_data_large_prefix: str = 'scan_data_large',
        save_scan_data_before_prefix: str = 'scan_data_before',
        save_scan_data_after_prefix: str = 'scan_data_after',
        save_vert_data_prefix: str = 'vert_data',
        save_vert_tip_pos_prefix: str = 'vert_tip_pos',
        plot_diss_points: bool = True,
        tip_form_ref_x: float = 280,
        tip_form_ref_y: float = -160,
        tip_form_z_range: list = [20, 35],
        tip_form_len_nm: float = 150,
        tip_form_dist_thres: float = 1,
        scan_default_z: float = 20,
        tip_form_check_points: list = None
) -> None:
    
    current_time = datetime.datetime.now()
    task_folder='%s/%s_%s_%s_%s_%s' % (total_output_folder, task_folder_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_img_large='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_large_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_img_before='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_before_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_img_after='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_img_after_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_data_large='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_large_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_data_before='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_before_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_data_after='%s/%s_%s_%s_%s_%s' % (task_folder, save_scan_data_after_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_vert_data='%s/%s_%s_%s_%s_%s' % (task_folder, save_vert_data_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)
    save_vert_tip_pos='%s/%s_%s_%s_%s_%s' % (task_folder, save_vert_tip_pos_prefix, current_time.month, current_time.day, current_time.hour, current_time.minute)

    # check if these folders exist

    if not os.path.exists(task_folder):
        os.mkdir(task_folder)
    if not os.path.exists(save_img_large):
        os.mkdir(save_img_large)
    if not os.path.exists(save_img_before):
        os.mkdir(save_img_before)
    if not os.path.exists(save_img_after):
        os.mkdir(save_img_after)
    if not os.path.exists(save_data_large):
        os.mkdir(save_data_large)
    if not os.path.exists(save_data_before):
        os.mkdir(save_data_before)
    if not os.path.exists(save_data_after):
        os.mkdir(save_data_after)
    if not os.path.exists(save_vert_data):
        os.mkdir(save_vert_data)
    if not os.path.exists(save_vert_tip_pos):
        os.mkdir(save_vert_tip_pos)

    # seperate the approach area into small scanning regions
    num_x=(diss_area[1]-diss_area[0])/scan_len_nm_large
    num_y=(diss_area[3]-diss_area[2])/scan_len_nm_large
    num_xy=int(np.min([num_x, num_y]))
    
    for i in range(num_xy):
        for j in range(num_xy):
            img_large_x_nm=diss_area[0]+scan_len_nm_large/2+i*scan_len_nm_large
            img_large_y_nm=diss_area[2]+j*scan_len_nm_large

            # tip form before all
            print('start tip forming now')
            tip_form_check_points=tip_form_region(env, tip_form_ref_x=tip_form_ref_x, tip_form_ref_y=tip_form_ref_y, tip_form_z_range=tip_form_z_range, tip_form_len_nm=tip_form_len_nm, tip_form_dist_thres=tip_form_dist_thres, scan_default_z=scan_default_z, tip_form_check_points=tip_form_check_points)

            # scan image before manipulation
            save_scan_image(env, x_nm=img_large_x_nm, y_nm=img_large_y_nm, pixel=pixel_large, scan_len_nm=scan_len_nm_large, save_img_folder=save_img_large, save_data_folder=save_data_large, img_name='%s_%s' % (i, j), data_name='%s_%s' % (i, j))
            img_large=cv2.imread('%s/img_forward_%s_%s.png' % (save_img_large, i, j))
            detect_mols_rough=image_select_points(img_large, edges=False, x_nm=img_large_x_nm, y_nm=img_large_y_nm, len_nm=scan_len_nm_large, pixel=pixel_large, dist_thres=1, absolute_pos_nm=True)

            if len(detect_mols_rough)>1:
                detect_one_mol=detect_indiv_mols(detect_mols_rough)
            elif len(detect_mols_rough)==1:
                detect_one_mol=detect_mols_rough
            else:
                detect_one_mol=[]
            if len(detect_one_mol)>0:
                for k in range(len(detect_one_mol)):
                    scan_mol_x_nm=detect_one_mol[k][0]
                    scan_mol_y_nm=detect_one_mol[k][1]-scan_len_nm_small/2  # scan the molecule from the center of the molecule
                    save_scan_image(env, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s_%s' % (i, j, k), data_name='%s_%s_%s' % (i, j, k))
                    img_small=cv2.imread('%s/img_forward_%s_%s_%s.png' % (save_img_before, i, j, k))
                    detect_mols_pos=image_select_points(img_small, edges=False, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, dist_thres=1, absolute_pos_nm=True)
                    if len(detect_mols_pos)==1:
                        mol_center_x_nm, mol_center_y_nm = detect_indiv_mol_center(img_small, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=128, absolute_pos_nm=True)
                        if mol_center_x_nm>scan_mol_x_nm-scan_len_nm_small/2+scan_len_nm_small/3 and mol_center_x_nm<scan_mol_x_nm+scan_len_nm_small/2-scan_len_nm_small/3 and mol_center_y_nm>scan_mol_y_nm+scan_len_nm_small/3 and mol_center_y_nm<scan_mol_y_nm+scan_len_nm_small*2/3:
                            pass
                        else:
                            scan_mol_x_nm=mol_center_x_nm
                            scan_mol_y_nm=mol_center_y_nm-scan_len_nm_small/2 
                            save_scan_image(env, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s_%s' % (i, j, k), data_name='%s_%s_%s' % (i, j, k))
                            img_small=cv2.imread('%s/img_forward_%s_%s_%s.png' % (save_img_before, i, j, k))
                            mol_center_x_nm, mol_center_y_nm = detect_indiv_mol_center(img_small, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=128, absolute_pos_nm=True)
                        
                        for diss_i in range(diss_maxtime):
                            print('Epoch %s_%s_%s_%s dissociation now' % (i, j, k, diss_i))
                            diss_x_nm=mol_center_x_nm+diss_radius_nm*(np.random.rand()*2-1)
                            diss_y_nm=mol_center_y_nm+diss_radius_nm*(np.random.rand()*2-1)
                            diss_mvoltage=diss_mvoltage+300*(np.random.rand()*2-1)
                            vert_data=env.createc_controller.diss_manipulation(diss_x_nm, diss_y_nm, diss_z_nm, diss_mvoltage, diss_pcurrent, np.array([scan_mol_x_nm, scan_mol_y_nm]), scan_len_nm_small)
                            vert_tip_pos=[i,j,k, diss_i, mol_center_x_nm, mol_center_y_nm, diss_x_nm, diss_y_nm, diss_z_nm, diss_mvoltage, diss_pcurrent]
                            with open(save_vert_data+'/vert_data_%s_%s_%s_%s.pkl' % (i, j, k, diss_i), "wb") as fp:  
                                pickle.dump(vert_data, fp)
                            with open(save_vert_tip_pos+'/vert_tip_pos_%s_%s_%s_%s.pkl' % (i, j, k, diss_i), "wb") as fp:
                                pickle.dump(vert_tip_pos, fp)

                            # tip forming
                            print('start tip forming now')
                            tip_form_check_points=tip_form_region(env, tip_form_ref_x=tip_form_ref_x, tip_form_ref_y=tip_form_ref_y, tip_form_z_range=tip_form_z_range, tip_form_len_nm=tip_form_len_nm, tip_form_dist_thres=tip_form_dist_thres, scan_default_z=scan_default_z, tip_form_check_points=tip_form_check_points)


                            # scan image after manipulation
                            print('Epoch %s_%s_%s_%s scan a image after dissociating now....' % (i, j, k, diss_i))
                            save_scan_image(env, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_after, save_data_folder=save_data_after, img_name='%s_%s_%s_%s' % (i, j, k, diss_i), data_name='%s_%s_%s_%s' % (i, j, k, diss_i))
                            img_small=cv2.imread('%s/img_forward_%s_%s_%s_%s.png' % (save_img_after, i, j, k, diss_i))
                            detect_mols_pos=image_select_points(img_small, edges=False, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, dist_thres=1, absolute_pos_nm=True)
                            if len(detect_mols_pos)>1:
                                break



                            




