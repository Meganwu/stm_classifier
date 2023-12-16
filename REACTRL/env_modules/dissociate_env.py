from .Env_new import RealExpEnv
from .createc_control import Createc_Controller
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .assign_and_anchor import assignment, align_design, align_deisgn_stitching, get_atom_and_anchor
from .img_attrib import mol_property

from .image_module_ellipse import image_detect_blobs

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
from .get_atom_coordinate import get_atom_coordinate_nm
import findiff
from .atom_jump_detection import AtomJumpDetector_conv
import os
from matplotlib import pyplot as plt, patches
import cv2
import datetime
from .img_conv import ResNet18, NormalNet, ResNet
from .net_framework import img_classifier
import torch


from collections import namedtuple
dissociate_data = namedtuple('dissociate_data',['time','x','y','current','dI_dV','topography'])

import pickle


class DissociateEnv:
        def __init__(self,
                diss_radius_nm=0.8, # nm
                goal_nm=2,  
                max_diss_z_nm=40.0,
                min_diss_z_nm=20,  
                max_mvoltage=3500,  
                max_pcurrent=1000, #  current to voltage ratio
                max_ellipse_width=3, # nm
                max_ellipse_height=3, # nm
                max_ellipse_angle=180, # degree
                max_ellipse_area=4.0, # nm^2
                template=None, # template for seeking molecules
                current_jump=4,
                im_size_nm=None,
                offset_nm=None,
                manip_limit_nm=None,
                pixel=128,
                template_max_y=3,
                scan_mV=1000,
                max_len=5,
                load_weight=None,
                mol_absolute_nm = None,
                mol_relative_nm = None,
                pull_back_mV = None,
                pull_back_pA = None,
                random_scan_rate = 0.5,
                correct_drift = False,
                bottom = True,
                cellsize = 10, # nm
                max_radius = 150, # nm
                forbid_radius = 35, # nm
                check_simi_forbid_radius = 30, # nm
                check_similarity = None,
                scan_ref_x_center = 0.0, # reference points x for seeking molecules
                scan_ref_y_center = 0.0, # reference points y for seeking molecules
                move_upper_limit = 400, # limited movement range for seeking molecules
                approach_limit=[-314, 100, 20,300], # limited range for seeking molecules
                mol_candidates = None, # mol_candidates for seeking molecules
                real_mols = None, # real_mols for seeking molecules
                fake_mols = None, # fake_mols for seeking molecules
                tipform_section = None, # tipform_section for seeking molecules # TODO list
                ):

                # dissociation parameters
                
                self.diss_radius_nm = diss_radius_nm
                self.max_diss_z_nm = max_diss_z_nm
                self.min_diss_z_nm = min_diss_z_nm
                self.max_mvoltage = max_mvoltage
                self.max_pcurrent = max_pcurrent


                # state paramters
                self.mol_absolute_nm = mol_absolute_nm
                self.mol_relative_nm = mol_relative_nm

                self.max_ellipse_width = max_ellipse_width
                self.max_ellipse_height = max_ellipse_height
                self.max_ellipse_angle = max_ellipse_angle
                self.max_ellipse_area = max_ellipse_area



                args = im_size_nm, offset_nm, pixel, scan_mV
                self.createc_controller = Createc_Controller(*args)
                self.offset_nm = offset_nm
                self.len_nm = im_size_nm
                self.goal_nm = goal_nm
                self.pixel = pixel


                self.template = template
                
                self.current_jump = current_jump
                self.manip_limit_nm = manip_limit_nm
                if self.manip_limit_nm is not None:
                        print('manipulation limit:', self.manip_limit_nm)
                        self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])

          
                self.default_reward = -1
                self.default_reward_done = 1
                self.max_len = max_len
                self.correct_drift = correct_drift
                self.template_max_y = template_max_y

                self.lattice_constant = 0.288
                self.precision_lim = self.lattice_constant*np.sqrt(3)/3
                self.bottom = bottom
                kwargs = {'data_len': 2048, 'load_weight': load_weight}
                self.atom_move_detector = AtomJumpDetector_conv(**kwargs)
                self.random_scan_rate = random_scan_rate
                self.accuracy, self.true_positive, self.true_negative = [], [], []
                if pull_back_mV is None:
                        self.pull_back_mV = 10
                else:
                        self.pull_back_mV = pull_back_mV

                if pull_back_pA is None:
                        self.pull_back_pA = 57000
                else:
                        self.pull_back_pA = pull_back_pA




                # find next scanning images
                self.check_similarity = check_similarity
                self.mol_candidates = mol_candidates
                self.cellsize = cellsize                          # the size of the scanning image
                self.max_radius = max_radius
                self.forbid_radius = forbid_radius
                self.limit_forbid_radius = forbid_radius*3
                self.check_simi_forbid_radius = check_simi_forbid_radius  # check whether new position for scanning images is too close to previous scanning regions
                self.num_cell = int(self.max_radius/self.cellsize)
                self.real_mols = real_mols
                self.fake_mols = fake_mols
                self.scan_ref_x_center = scan_ref_x_center
                self.scan_ref_y_center = scan_ref_y_center
                self.move_upper_limit = move_upper_limit  # the maximum distance for seeking molecules from reference points.
                self.approach_limit = approach_limit

                # image classifier
                self.img_cnn=ResNet18(1, 2)
                self.img_classifier=img_classifier(self.img_cnn, device='cpu')
                self.img_classifier.model_load('REACTRL/params/img_classifier_best.pth')
                
        

        def reset(self, 
                scan_mol_x_nm=None, 
                scan_mol_y_nm=None, 
                pixel_small=128, 
                scan_len_nm_large=20,
                scan_len_nm_small=3.5,
                scan_speed=200,
                update_conv_net=True, 
                all_file_label='example', 
                find_mol=False, 
                new_scan_region=True,
                check_similarity=None, 
                candidate_mols=None, 
                current_time=None, 
                total_output_folder = 'all_output',
                task_folder_prefix = 'vertmani',
                save_scan_img_large_prefix = 'scan_img_large',
                save_scan_img_before_prefix = 'scan_img_before',
                save_scan_img_after_prefix = 'scan_img_after',
                save_scan_data_large_prefix = 'scan_data_large',
                save_scan_data_before_prefix = 'scan_data_before',
                save_scan_data_after_prefix = 'scan_data_after',
                save_vert_data_prefix = 'vert_data',
                save_vert_tip_pos_prefix = 'vert_tip_pos',
                  ):
                """
                Reset the environment

                Parameters
                ----------
                update_conv_net: bool
                        whether to update the parameters of the AtomJumpDetector_conv CNN

                Returns
                -------
                self.state: array_like
                info: dict
                """

        #TODO  build atom_diss_detector.currents_val

                # if (len(self.atom_move_detector.currents_val)>self.atom_move_detector.batch_size) and update_conv_net:
                #         accuracy, true_positive, true_negative = self.atom_move_detector.eval()
                #         self.accuracy.append(accuracy)
                #         self.true_positive.append(true_positive)
                #         self.true_negative.append(true_negative)
                #         self.atom_move_detector.train()

                if current_time is None:
                        current_time = datetime.datetime.now()
                        current_time = '%s_%s_%s_%s' % (current_time.month, current_time.day, current_time.hour, current_time.minute) 


                task_folder='%s/%s_%s' % (total_output_folder, task_folder_prefix, current_time)
                save_img_large='%s/%s_%s' % (task_folder, save_scan_img_large_prefix, current_time)
                save_img_before='%s/%s_%s' % (task_folder, save_scan_img_before_prefix, current_time)
                save_img_after='%s/%s_%s' % (task_folder, save_scan_img_after_prefix, current_time)
                save_data_large='%s/%s_%s' % (task_folder, save_scan_data_large_prefix, current_time)
                save_data_before='%s/%s_%s' % (task_folder, save_scan_data_before_prefix, current_time)
                save_data_after='%s/%s_%s' % (task_folder, save_scan_data_after_prefix, current_time)
                self.get_state_current_time=current_time

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




                if scan_mol_x_nm is None or scan_mol_y_nm is None:
                        scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()[0], self.createc_controller.get_offset_nm()[1]

                # scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=scan_speed, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name=all_file_label, data_name=all_file_label)
                # img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, all_file_label), cv2.IMREAD_GRAYSCALE)
                # img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                # img_prop_small.center_points_from_contour()
                # try:
                #         img_prop_small.contour_property()
                #         mol_center_x_nm=img_prop_small.ellipse_x
                #         mol_center_y_nm=img_prop_small.ellipse_y 
                #         # self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                #         self.state_nm=[img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle, img_prop_small.area]
                #         self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                #         self.img_info={'scan_data': scan_data, 'mol_property':img_prop_small}
                #         info = {'scan_data_before': scan_data, 'mol_property':img_prop_small, 'mol_x_nm_before': mol_center_x_nm, 'mol_y_nm_before': mol_center_y_nm, 'mol_width_before': img_prop_small.ellipse_width, 'mol_height_before': img_prop_small.ellipse_height, 'mol_angle_before': img_prop_small.ellipse_angle}
                #         find_mol=True
                # except:
                #         find_mol=False
                #         print('no molecule found, start to scan a new region')


                if not find_mol:                      
                        mol_center_x_nm, mol_center_y_nm, check_similarity, candidate_mols=self.get_state(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, scan_len_nm_large=scan_len_nm_large, new_scan_region=new_scan_region, check_similarity=check_similarity, candidate_mols=candidate_mols,  current_time=current_time, img_name=all_file_label, data_name=all_file_label)
                        scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()

                        
                scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=scan_speed, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name=all_file_label, data_name=all_file_label)
                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, all_file_label), cv2.IMREAD_GRAYSCALE)
                img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                img_prop_small.center_points_from_contour()

                img_prop_small.contour_property()
                mol_center_x_nm=img_prop_small.ellipse_x
                mol_center_y_nm=img_prop_small.ellipse_y 
                # self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                self.state_nm=[img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle, img_prop_small.area]
                self.state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]
                self.img_info={'scan_data': scan_data, 'mol_property':img_prop_small}
                info = {'scan_data_before': scan_data, 'mol_property':img_prop_small, 'mol_x_nm_before': mol_center_x_nm, 'mol_y_nm_before': mol_center_y_nm, 'mol_width_before': img_prop_small.ellipse_width, 'mol_height_before': img_prop_small.ellipse_height, 'mol_angle_before': img_prop_small.ellipse_angle, 'mol_area_before': img_prop_small.area}

                
                self.mol_center_x_nm=mol_center_x_nm
                self.mol_center_y_nm=mol_center_y_nm

                self.info=info
   

                return self.state, info, check_similarity, candidate_mols

        def step(self, 
                action,
                current_time=None,
                all_file_label='example_step',
                pixel_small=128,
                scan_len_nm_small=3.5,
                check_similarity=None,
                candidate_mols=None,
                total_output_folder = 'all_output',
                task_folder_prefix = 'vertmani',
                save_scan_img_large_prefix = 'scan_img_large',
                save_scan_img_before_prefix = 'scan_img_before',
                save_scan_img_after_prefix = 'scan_img_after',
                save_scan_data_large_prefix = 'scan_data_large',
                save_scan_data_before_prefix = 'scan_data_before',
                save_scan_data_after_prefix = 'scan_data_after',
                save_vert_data_prefix = 'vert_data',
                save_vert_tip_pos_prefix = 'vert_tip_pos',
                 ):
                """
                Take a large STM scan and update the atoms and designs after a RL episode  

                Parameters
                ----------
                succeed: bool
                        if the RL episode was successful
                
                new_atom_position: array_like
                        the new position of the manipulated atom

                Returns
                -------
                self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.anchor_chosen: array_like
                        the positions of the atom, design, target, and anchor to be used in the RL episode 
                
                self.paths: array_like
                        the planned path between atom and design
                
                offset_nm: array_like
                        offset value to use for the STM scan

                len_nm: float
                        image size for the STM scan 
                
                done:bool 
                """

                if current_time is None:
                        current_time = datetime.datetime.now()
                        current_time = '%s_%s_%s_%s' % (current_time.month, current_time.day, current_time.hour, current_time.minute) 

                task_folder='%s/%s_%s' % (total_output_folder, task_folder_prefix, current_time)
                save_img_large='%s/%s_%s' % (task_folder, save_scan_img_large_prefix, current_time)
                save_img_before='%s/%s_%s' % (task_folder, save_scan_img_before_prefix, current_time)
                save_img_after='%s/%s_%s' % (task_folder, save_scan_img_after_prefix, current_time)
                save_data_large='%s/%s_%s' % (task_folder, save_scan_data_large_prefix, current_time)
                save_data_before='%s/%s_%s' % (task_folder, save_scan_data_before_prefix, current_time)
                save_data_after='%s/%s_%s' % (task_folder, save_scan_data_after_prefix, current_time)
                save_vert_data='%s/%s_%s' % (task_folder, save_vert_data_prefix, current_time)
                save_vert_tip_pos='%s/%s_%s' % (task_folder, save_vert_tip_pos_prefix, current_time)

                self.get_state_current_time=current_time


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

                
                # scan image before dissociation manipulation
                mol_center_x_nm=self.mol_center_x_nm
                mol_center_y_nm=self.mol_center_y_nm
        
                rets = self.action_to_diss_input(action)
                diss_x_nm, diss_y_nm, diss_mvoltage, diss_pcurrent = rets
                args = diss_x_nm, diss_y_nm, diss_mvoltage, diss_pcurrent
                vert_data = self.step_dissociate(*args)

                self.diss_topography=np.array(vert_data.topography)
                diff_topography=np.abs(self.diss_topography[0:512].sum()-self.diss_topography[512:].sum())

                print('Epoch %s dissociation diff_topography: %.2f ' % (all_file_label, diff_topography))




                # scan image after dissociation manipulation
                print('Epoch %s scan a image after dissociating now....' % all_file_label)

                scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()
                scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=200, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_after, save_data_folder=save_data_after, img_name=all_file_label, data_name=all_file_label)
                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_after, all_file_label), cv2.IMREAD_GRAYSCALE)
                img_cnn_detect_mol=self.detect_mol_cnn(img_small)
                img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                img_prop_small.center_points_from_contour()
                
                find_mol=False
                new_mol=False

                if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>10 and  len(img_prop_small.detect_mols_from_contours)>0:
                        img_prop_small.contour_property()
                        if img_prop_small.area<3.0 and img_prop_small.area>0.5:
                                mol_center_x_nm=img_prop_small.ellipse_x
                                mol_center_y_nm=img_prop_small.ellipse_y
                                
                                if  mol_center_x_nm>scan_mol_x_nm-scan_len_nm_small/8 and mol_center_x_nm<scan_mol_x_nm+scan_len_nm_small/8 and mol_center_y_nm>scan_mol_y_nm+scan_len_nm_small*3/8 and mol_center_y_nm<scan_mol_y_nm+scan_len_nm_small*5/8: # and img_cnn_detect_mol:
                                        scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()
                                        find_mol=True
                                
                                else:
                                        print('adjust mol pos')
                                        scan_mol_x_nm=mol_center_x_nm
                                        scan_mol_y_nm=mol_center_y_nm-scan_len_nm_small/2 
                                        scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=200, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_after, save_data_folder=save_data_after, img_name='%s_adjust' % all_file_label, data_name='%s_adjust' % all_file_label)
                                        img_small=cv2.imread('%s/img_forward_%s_adjust.png' % (save_img_after, all_file_label), cv2.IMREAD_GRAYSCALE)
                                        img_cnn_detect_mol=self.detect_mol_cnn(img_small)    # position is important
                                        scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()
                                        img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                                        img_prop_small.center_points_from_contour()
                                        img_prop_small.contour_property()
                                        
                                        if img_prop_small.area<3.0 and img_prop_small.area>0.8: # and img_cnn_detect_mol:
                                        
                                                mol_center_x_nm=img_prop_small.ellipse_x
                                                mol_center_y_nm=img_prop_small.ellipse_y                                     
                                                scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()
                                                find_mol=True

                                self.mol_center_x_nm=mol_center_x_nm
                                self.mol_center_y_nm=mol_center_y_nm
                                next_state_nm=[img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle, img_prop_small.area]
                                self.next_state=[img_prop_small.ellipse_width/self.max_ellipse_width, img_prop_small.ellipse_height/self.max_ellipse_height, img_prop_small.ellipse_angle/self.max_ellipse_angle, img_prop_small.area/self.max_ellipse_area]

                if not find_mol:
                        next_state_nm=[0.0, 0.0, 0.0, 0.0]
                        self.next_state=[0.0, 0.0, 0.0, 0.0]

                done_diss=self.old_detect_diss(self.diss_topography)
                vert_tip_pos=[all_file_label, mol_center_x_nm, mol_center_y_nm, diss_x_nm, diss_y_nm, diss_mvoltage, diss_pcurrent, done_diss]
                print('Epoch %s dissociation diff_topography: %.2f diff_area %.2f' % (all_file_label, self.diff_topography, self.diff_area))
                with open(save_vert_data+'/vert_data_%s.pkl' % all_file_label, "wb") as fp:  
                        pickle.dump(vert_data, fp)
                with open(save_vert_tip_pos+'/vert_tip_pos_%s.pkl' % all_file_label, "wb") as fp:
                        pickle.dump(vert_tip_pos, fp)                


                reward=self.compute_reward()   #np.array(vert_data.topography))  

                self.info |= {'vert_data': vert_data, 'state': self.state, 'next_state': self.next_state, 'diss_x_nm': diss_x_nm, 'diss_y_nm': diss_y_nm, 'diss_mvoltage': diss_mvoltage, 'diss_pcurrent': diss_pcurrent, 'done_diss': done_diss,'scan_data_after': scan_data, 'mol_x_nm_after': mol_center_x_nm, 'mol_y_nm_after': mol_center_y_nm}
                                        
                
                if not find_mol or done_diss:
                        mol_center_x_nm, mol_center_y_nm, check_similarity, candidate_mols=self.get_state(x_nm=self.createc_controller.get_offset_nm()[0], y_nm=self.createc_controller.get_offset_nm()[1], scan_len_nm_large=20, new_scan_region=True, check_similarity=check_similarity, candidate_mols=candidate_mols,  current_time=current_time, img_name=all_file_label, data_name=all_file_label)
                        scan_mol_x_nm, scan_mol_y_nm = self.createc_controller.get_offset_nm()
                        find_mol=True
                        new_mol=True
                        # scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=200, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name=all_file_label, data_name=all_file_label)
                        # img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, all_file_label), cv2.IMREAD_GRAYSCALE)
                        # img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                        # img_prop_small.center_points_from_contour()
                        # img_prop_small.contour_property()
                        # mol_center_x_nm=img_prop_small.ellipse_x
                        # mol_center_y_nm=img_prop_small.ellipse_y 


                return self.next_state, reward, done_diss, self.info, new_mol

        def measure_fragment(self, img: np.array)->tuple:   
                """
                Measure the fragment after dissociation

                Parameters
                ----------
                img: array_like
                        the STM image after dissociation

                Returns
                -------
                center_x, center_y, length, width, angle: float
                        the center position and size of the fragment
                """

                ell_shape=image_detect_blobs(img, kernal_v=8)
                return ell_shape 


    

        def compute_reward(self)->float:
                """
                Calculate the reward after dissociation

                Parameters
                ----------
                img_forward: array_like
                        the STM image before dissociation

                img_forward_next: array_like
                        the STM image after dissociation

                Returns
                -------
                reward: float
                        the reward for the RL agent
                """
                topography=self.diss_topography
                self.old_detect_diss(topography)
                if self.old_detect_diss(topography):
                        reward = 1.0

                elif self.diff_topography<1.5 or self.diff_area>1.5 or self.diff_area<0.10:
                        reward = -1.0
                else:
                        reward = np.min([self.diff_topography, 2.5])/2.5
        
                return reward
        
        def random_action_generator(self):
                """
                Generate random actions for dissociation

                Returns
                -------
                action: array_like
                        the random action for the RL agent
                """
                action = list(np.random.random(2)*2-1)+list(np.random.random(2))
                return action
                
        
        def action_to_diss_input(self, action):
                """
                Convert the action to the input for the dissociation

                Parameters
                ----------
                action: array_like 4D
                        the action from the RL agent

                Returns
                -------
                diss_x_nm, diss_y_nm, x_end_nm, y_end_nm, mvoltage, pcurrent, diss_z_nm
                        the input for the dissociation
                """
                diss_x_nm = action[0]*self.diss_radius_nm+self.mol_center_x_nm
                diss_y_nm = action[1]*self.diss_radius_nm+self.mol_center_y_nm
                # diss_z_nm = np.clip(action[2], a_min = None, a_max=1)*(self.max_diss_z_nm-self.min_diss_z_nm)/2+np.mean(self.min_diss_z_nm+self.max_diss_z_nm)
                mvoltage = np.clip(action[2], a_min = None, a_max=1)*self.max_mvoltage
                pcurrent = np.clip(action[3], a_min = None, a_max=1)*self.max_pcurrent
                return diss_x_nm, diss_y_nm, mvoltage, pcurrent



    
        def step_dissociate(self, diss_x_nm, diss_y_nm, mvoltage, pcurrent):
                """
                Execute the action in Createc

                Parameters
                ----------
                diss_x_nm, diss_y_nm, diss_z_nm: float
                        the position of the tip dissociation in nm
                mvoltage: float
                        bias voltage in mV
                pcurrent: float
                        current setpoint in pA

                Return
                ------
                current: array_like
                        manipulation current trace
                d: float
                        tip movement distance
                """

                offset_nm = self.createc_controller.get_offset_nm()
                len_nm = self.createc_controller.get_len_nm()

                x_kwargs = {'a_min':offset_nm[0]-len_nm/2, 'a_max':offset_nm[0]+len_nm/2}
                y_kwargs = {'a_min':offset_nm[1], 'a_max':offset_nm[1]+len_nm}

                diss_x_nm = np.clip(diss_x_nm, **x_kwargs)
                diss_y_nm = np.clip(diss_y_nm, **y_kwargs) 

                pos = diss_x_nm, diss_y_nm
                params = mvoltage, pcurrent, offset_nm, len_nm
                vert_data = self.createc_controller.diss_manipulation(*pos, *params)

                return vert_data
        
        def old_detect_diss(self, topography, thres=2.0, thres_area_diff=0.10, thres_area_diff_2=1.5):
                """
                Estimate if atom has dissociated based on the difference of topography before and after the highest bias voltage

                Parameters
                ----------
                topography: array_like (1D)
                        

                Returns
                -------
                bool
                        whether the atom has likely dissociated
                """
                diff_area = np.abs(self.state[3]-self.next_state[3])
                diff_topography=np.abs(topography[0:512].sum()-topography[512:].sum())
                self.diff_topography=diff_topography
                self.diff_area=diff_area

                if diff_topography>thres and diff_area*self.max_ellipse_area>thres_area_diff and diff_area*self.max_ellipse_area<thres_area_diff_2:
                        return True
                else:
                        print('Different prediction shows no dissociation')
                        return False
                

        def detect_mol_cnn(self, img):
                """
                Estimate if atom has dissociated based on the CNN model

                Parameters
                ----------
                img: array_like
                        STM image after dissociation

                Returns
                -------
                bool
                        whether the atom has likely dissociated
                """
                img=cv2.resize(img, (128,128), interpolation=cv2.INTER_AREA)
                img=torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
                pred_label=self.img_classifier.model_predict(img)
                if np.argmax(pred_label.detach().numpy())==0:
                        print('img classifier evaluates the molecule is original molecule')
                        return True
                else:
                        print('img classifier evaluates the molecule is not original molecule')
                        return False






        
        def old_detect_current_jump(self, current):
                """
                Estimate if molecule has dissociated based on the gradient of the manipulation current trace

                Parameters
                ----------
                current: array_like
                        manipulation current trace

                Return
                ------
                bool
                whether the molecule has likely dissociated
                """
                if current is not None:
                        diff = findiff.FinDiff(0,1,acc=6)(current)[3:-3]
                        return np.sum(np.abs(diff)>self.current_jump*np.std(current)) > 2
                else:
                        return False
        
        def detect_current_jump_cnn(self, current):
                """
                Estimate if atom has moved based on AtomJumpDetector_conv and the gradient of the manipulation current trace

                Parameters
                ----------
                current: array_like
                        manipulation current trace

                Returns
                -------
                bool
                        whether the molecule has likely dissociated
                """
                if current is not None:
                        success, prediction = self.atom_diss_detector.predict(current)
                        old_prediction = self.old_detect_current_jump(current)
                        print('CNN prediction:',prediction,'M1 prediction:', old_prediction)
                        if success:
                                print('cnn thinks there is molecule dissociation')
                                return True
                        elif old_prediction and (np.random.random()>(self.random_scan_rate-0.3)):
                                return True
                        elif (np.random.random()>(self.random_scan_rate-0.2)) and (prediction>0.35):
                                print('Random scan')
                                return True
                        elif np.random.random()>self.random_scan_rate:
                                print('Random scan')
                                return True
                        else:
                                print('CNN and old prediction both say no dissociation')
                                return False
                else:
                        print('CNN and old prediction both say no dissociation')                    
                        return False

        def out_of_range(self, pos_nm, mani_limit_nm):
                """
                Check if the atom is out of the manipulation limit

                Parameters
                ----------
                pos: array_like
                        the position of the molcule in STM coordinates in nm

                mani_limit: array_like
                        [left, right, up, down] limit in STM coordinates in nm

                Returns
                -------
                bool
                        whether the atom is out of the manipulation limit
                """
                out = np.any((pos_nm-mani_limit_nm[[0,2]])*(pos_nm - mani_limit_nm[[1,3]])>0, axis=-1)
                return out       

        def pull_atom_back(self):
                """
                Pull atom to the center of self.manip_limit_nm with self.pull_back_mV, self.pull_back_pA
                """
                print('pulling atom back to center')
                current = self.pull_back_pA
                pos0 = self.atom_absolute_nm[0], self.atom_absolute_nm[1]
                pos1x = np.mean(self.manip_limit_nm[:2])+2*np.random.random()-1
                pos1y = np.mean(self.manip_limit_nm[2:])+2*np.random.random()-1
                params = self.pull_back_mV, current, self.offset_nm, self.len_nm
                self.createc_controller.lat_manipulation(*pos0, pos1x, pos1y, *params)        

        def atoms_detection(self, img):
                """
                Detect atoms 

                Returns
                -------
                bool
                        whether there are atoms
                """
                no_atom = (len(blob_detection(img)[0]) > 0)
                return no_atom 
        
        def debris_detection(self, topography, debris_thres = 6e-9):
                """
                Detect debris based on topography from scandata(1,4)

                Returns
                -------
                bool
                        whether there are debris
                """

                topography=np.array(topography).flatten()

                no_debris =  ((np.max(topography) - np.min(topography)) < debris_thres) 
                return no_debris

        def crash_detection(self, topography, crash_thres = 1.5e-9):
                """
                Detect crash based on topography from scandata(1,4)

                Returns 
                -------
                bool
                        whether there is crash
                """
                topography=np.array(topography).flatten()

                no_crash = (np.max(topography) - np.min(topography)) < crash_thres

                return no_crash    


        
        def adjust_mol_pos(self, mol_pos, scan_len_nm_small=3.5, pixel_small=128, scan_speed=200, save_img_folder=None, save_data_folder=None, img_name=None, data_name=None):
                """
                Adjust the molecular position based on the image and put it in the center of the image

                Parameters
                ----------
                mol_pos: list example [1.0, 2.0]
                        the molecular position

                Returns
                -------
                molecular postions
                        whether there are molecules
                """
                scan_mol_x_nm=mol_pos[0]
                scan_mol_y_nm=mol_pos[1]-scan_len_nm_small/2
                scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=scan_speed, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name), cv2.IMREAD_GRAYSCALE)
                img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small)
                img_prop_small.center_points_from_contour()
                if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>10 and  len(img_prop_small.detect_mols_from_contours)>0:
                        img_prop_small.contour_property()
                        mol_center_x_nm, mol_center_y_nm =img_prop_small.ellipse_x, img_prop_small.ellipse_y
                        if mol_center_x_nm>scan_mol_x_nm-scan_len_nm_small/8 and mol_center_x_nm<scan_mol_x_nm+scan_len_nm_small/8 and mol_center_y_nm>scan_mol_y_nm+scan_len_nm_small*3/8 and mol_center_y_nm<scan_mol_y_nm+scan_len_nm_small*5/8:
                                pass
                        else:
                                scan_mol_x_nm=mol_center_x_nm
                                scan_mol_y_nm=mol_center_y_nm-scan_len_nm_small/2 
                                scan_data=self.save_scan_image(x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_speed=scan_speed, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                                # img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, img_name))
                                # img_prop_small=mol_property(img_small, offset_x_nm=scan_mol_x_nm, offset_y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
                                # img_prop_small.center_points_from_contours()    
                                # img_prop_small.contours_property()
                                # mol_center_x_nm, mol_center_y_nm =img_prop_small.ellipse_x, img_prop_small.ellipse_y
                        # return scan_data, mol_center_x_nm, mol_center_y_nm
                return scan_data


        def save_scan_image(
                self,
                x_nm: float = 0,
                y_nm: float = 0,
                pixel: int = 128,
                scan_mvoltage: float = 1000.0,
                scan_speed: float = 500,
                scan_len_nm: float = 10,
                save_img_folder: str = None,
                save_data_folder: str = None,
                img_name: str = None,
                data_name: str = None,
                save_img: bool = True,
                save_data: bool = True
                ) -> tuple:
                """scanning a image and save

                Args:
                        env (_type_): createc environment.
                        x_nm (_type_, optional): the offset_x_nm of scanning region.
                        y_nm (_type_, optional): the offset_y_nm of scanning region.
                        scan_len_nm (_type_, optional): the length of scanning region.
                        save_img_folder (_type_, optional): the folder for saving images.
                        filename (_type_, optional): the filename for saving images.

                Returns:
                        _type_: _description_
                """        
                self.createc_controller.offset_nm = np.array([x_nm, y_nm])
                self.createc_controller.im_size_nm= scan_len_nm
                self.createc_controller.pixel = pixel
                # self.createc_controller.stm.setparam('Num.X', pixel)
                # self.createc_controller.stm.setparam('Num.Y', pixel)
                self.createc_controller.stm.setparam('BiasVolt.[mV]', scan_mvoltage)
                    #TODO add line to set pixel
                scan_data=self.createc_controller.scan_image(speed=scan_speed)
                if save_img:
                        plt.imsave('%s/img_forward_%s.png' % (save_img_folder, img_name), scan_data[0])
                        # plt.imsave('%s/img_backward_%s.png' % (save_img_folder, img_name), scan_data[1])
                        # plt.imsave('%s/img_for_back_%s.png' % (save_img_folder, img_name), (scan_data[0]+scan_data[1])/2)
                if save_data:
                        with open(os.path.join(save_data_folder, 'scan_data_%s.pkl' % (data_name)), "wb") as fp:
                           pickle.dump(scan_data, fp)
                return scan_data
        

        def tip_form_region(
                self,
                tip_form_ref_x: float = -260,
                tip_form_ref_y: float = 0,
                tip_form_z_range: list = [20, 35],
                tip_form_len_nm: float = 100,
                tip_form_dist_thres: float = 1,
                scan_default_z: float = 20,
                tip_form_check_points: list = None
                ) -> list:

                """forming tip at a specific region
                Usage example: tip_form_region(tip_form_ref_x=280, tip_form_ref_y=-160, tip_form_z_range=[20, 35], tip_form_len_nm=150, tip_form_dist_thres=1, scan_default_z=20, tip_form_check_points=None)
                Args:
                        tip_form_ref_x (float, optional): the offset_x_nm of region for forming tips. 
                        tip_form_ref_y (float, optional): the offset_y_nm of region for forming tips. 
                        tip_form_z_range (list, optional): the range of z_nm for forming tips. 
                        tip_form_len_nm (float, optional): the length of region for forming tips.
                        tip_form_dist_thres (float, optional): the distance threshold for select a tip form position which should be away from previous tip forming points.
                        scan_default_z (float, optional): the default z_nm for scanning after tip forming.
                        tip_form_check_points (list, optional): the list of previous tip forming points. 

                
                Returns:
                        _type_: _description_
                """    
                print('start tip forming now')
                tip_form_x=tip_form_ref_x-tip_form_len_nm/2+tip_form_len_nm*np.random.rand()
                tip_form_y=tip_form_ref_y+tip_form_len_nm*np.random.rand()
                if tip_form_check_points is None:
                        tip_form_check_points=[]
                        tip_form_check_points.append([tip_form_ref_x, tip_form_ref_y])
                tip_form_check_points_array=np.array(tip_form_check_points)
                while np.sqrt((tip_form_check_points_array[:, 0]-tip_form_x)**2+(tip_form_check_points_array[:, 1]-tip_form_y)**2).min()<tip_form_dist_thres:
                        tip_form_x=tip_form_ref_x-tip_form_len_nm/2+tip_form_len_nm*np.random.rand()
                        tip_form_y=tip_form_ref_y+tip_form_len_nm*np.random.rand()
                upper_limit=tip_form_z_range[1]
                lower_limit=tip_form_z_range[0]
                tip_form_z=lower_limit+np.random.rand()*(upper_limit-lower_limit)
                self.createc_controller.tip_form(tip_form_z, tip_form_x, tip_form_y, np.array([tip_form_ref_x, tip_form_ref_y]), tip_form_len_nm)  
                tip_form_check_points.append([tip_form_x, tip_form_y])
                self.createc_controller.set_Z_approach(scan_default_z)
                return tip_form_check_points

        def get_state(self,
                x_nm: float = None,
                y_nm: float = None,
                scan_len_nm_large: int = 20,
                pixel_large: int = 128,
                scan_len_nm_small: float = 3.5,
                pixel_small: int = 128,
                new_scan_region: bool = False,
                check_similarity: list = None,
                candidate_mols: list = None,               # existing_mols is a list of possible individual mols based on rought detection
                real_mols: list = None,                # checked_mols is a list of mols that have been checked
                fake_mols: list = None,                    # mol_fake is a list of mols that are not real mols
                max_seek_time: int = 3,
                total_output_folder: str = 'all_output',
                current_time: str = None,
                task_folder_prefix: str = 'get_new_region',
                save_scan_img_large_prefix: str = 'scan_img_large',
                save_scan_img_before_prefix: str = 'scan_img_before',
                save_scan_img_after_prefix: str = 'scan_img_after',
                save_scan_data_large_prefix: str = 'scan_data_large',
                save_scan_data_before_prefix: str = 'scan_data_before',
                save_scan_data_after_prefix: str = 'scan_data_after',
                save_vert_data_prefix: str = 'vert_data',
                save_vert_tip_pos_prefix: str = 'vert_tip_pos',
                img_name: str = None,
                data_name: str = None,
                ):
        
                """
                Get the state of the environment

                Returns
                -------
                self.state: array_like
                """
                if current_time is None:
                        current_time = datetime.datetime.now()
                        current_time = '%s_%s_%s_%s' % (current_time.month, current_time.day, current_time.hour, current_time.minute) 


                task_folder='%s/%s_%s' % (total_output_folder, task_folder_prefix, current_time)
                save_img_large='%s/%s_%s' % (task_folder, save_scan_img_large_prefix, current_time)
                save_img_before='%s/%s_%s' % (task_folder, save_scan_img_before_prefix, current_time)
                save_img_after='%s/%s_%s' % (task_folder, save_scan_img_after_prefix, current_time)
                save_data_large='%s/%s_%s' % (task_folder, save_scan_data_large_prefix, current_time)
                save_data_before='%s/%s_%s' % (task_folder, save_scan_data_before_prefix, current_time)
                save_data_after='%s/%s_%s' % (task_folder, save_scan_data_after_prefix, current_time)
                self.get_state_current_time=current_time


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


                if check_similarity is None or len(check_similarity)==0:
                        check_similarity = [[x_nm, y_nm]]

                if real_mols is None:
                        real_mols = []
                if fake_mols is None:
                        fake_mols = []


                if self.check_similarity is None:
                        self.check_similarity = check_similarity


                
                done = False
                max_time_limit=5
                seek_time=0

                if new_scan_region:
                        found_mol=False
                        while not found_mol:
                                self.tip_form_region()   
                                while (candidate_mols is None) or len(candidate_mols)==0:  
                                            
                                        seek_time+=1
                                        print('start scanning region: seek_time', seek_time, self.createc_controller.get_offset_nm())
                
                                        if img_name is None:
                                                self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixel=pixel_large, scan_len_nm=scan_len_nm_large, scan_speed=1000, save_img_folder=save_img_large, save_data_folder=save_data_large, img_name=str(seek_time), data_name=str(seek_time))
                                                img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_large, str(seek_time)), cv2.IMREAD_GRAYSCALE)
                                        else:
                                                self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixel=pixel_large, scan_len_nm=scan_len_nm_large, scan_speed=1000, save_img_folder=save_img_large, save_data_folder=save_data_large, img_name=img_name+'_'+str(seek_time), data_name=data_name+'_'+str(seek_time))
                                                img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_large, img_name+'_'+str(seek_time)), cv2.IMREAD_GRAYSCALE)
                                        img_prop_large=mol_property(img_large, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_large)
                                        # select mol from points
                                        detect_mols_1=img_prop_large.detect_mol_from_points()
                                        # detect_mols_1=img_prop_large.detect_mol_from_points()

                                        # select mol from contours
                                        img_prop_large.center_points_from_contour()
                                        detect_mols_2=img_prop_large.detect_mols_center_from_contours

                                        # detect_mols=[i for i in detect_mols_1 if i in detect_mols_2]
                                        
                                        if len(detect_mols_1)>0 and len(detect_mols_2)>0:
                                                detect_mols_1_array=np.array(detect_mols_1)
                                                detect_mols_2_array=np.array(detect_mols_2)
                                                candidate_mols=[i for i in detect_mols_2 if np.sqrt((detect_mols_1_array[:, 0]-i[0])**2+(detect_mols_1_array[:, 1]-i[1])**2).min()<1.2]
                                        else:
                                                candidate_mols=[]

                                        if len(candidate_mols)>0:
                                                print('candidate_mols', candidate_mols)
                                        
                                        else:
                                                forbid_radius=20
                                                print('start finding new region')
                                                candidate_mols=None 
                                                if seek_time>max_seek_time:
                                                        x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, forbid_radius=forbid_radius*2, check_similarity=check_similarity)
                                                else:
                                                        x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, check_similarity=check_similarity)
                                                check_similarity.append([x_nm, y_nm])
                                                print('looking for new region x_nm, y_nm', x_nm, y_nm)

                                candidate_mols_num=len(candidate_mols)

                                for i in range(candidate_mols_num):
                                        print('mol num:', i)
                                        
                                        mol_pos=candidate_mols[0]
                                        candidate_mols.remove(mol_pos)
      
                                                
                                        if img_name is None:
                                                self.adjust_mol_pos(mol_pos, scan_len_nm_small=scan_len_nm_small, pixel_small=128, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s' % (seek_time, i), data_name='%s_%s' % (seek_time, i))                
                                                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, '%s_%s' % (seek_time, i)), cv2.IMREAD_GRAYSCALE)
                                        else:
                                                self.adjust_mol_pos(mol_pos, scan_len_nm_small=scan_len_nm_small, pixel_small=128, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name='%s_%s_%s' % (img_name, seek_time, i), data_name='%s_%s_%s' % (img_name, seek_time, i))              

                                                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, '%s_%s_%s' % (img_name, seek_time, i)), cv2.IMREAD_GRAYSCALE)

                                        img_cnn_detect_mol=self.detect_mol_cnn(img_small)
                                        scan_offset_nm=self.createc_controller.get_offset_nm()
                                        img_prop_small=mol_property(img_small, offset_x_nm=scan_offset_nm[0], offset_y_nm=scan_offset_nm[1], len_nm=scan_len_nm_small)
                                        img_prop_small.center_points_from_contour()
                                        

  
        
                                        

                                        if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>10 and  len(img_prop_small.detect_mols_from_contours)>0 and img_cnn_detect_mol:
                                                img_prop_small.contour_property()
                                                self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                                                real_mols.append(mol_pos)  
                                                found_mol=True
                                                break

                                        else:
                                                fake_mols.append(mol_pos)
                                                found_mol=False
                                                print('No Molecule detected')


                                        


                else:
                        self.save_scan_image(x_nm=x_nm, y_nm=y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, scan_speed=200, save_img_folder=save_img_before, save_data_folder=save_data_before, img_name=img_name, data_name=data_name)
                        img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, img_name), cv2.IMREAD_GRAYSCALE)

                        img_prop_small=mol_property(img_small, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_small)
                        img_prop_small.center_points_from_contour()
                        img_prop_small.contour_property()
                        img_cnn_detect_mol=self.detect_mol_cnn(img_small)

                        try:
                                if len(img_prop_small.contours)>0 and len(img_prop_small.contours_max)>5 and len(img_prop_small.detect_mols_from_contours)>0:
                                        self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_width, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                        except:
                                print('No Molecule detected')
    
                return img_prop_small.ellipse_x, img_prop_small.ellipse_y, check_similarity, candidate_mols

                                
                
                # if found_mol:
                #         return self.state, scan_offset_nm, scan_len_nm_small, found_mol, check_similarity, candidate_mols, real_mols, fake_mols
                # else:
                #         return None, scan_offset_nm, scan_len_nm_small, found_mol, check_similarity, candidate_mols, real_mols, fake_mols
                   




        def set_new_tippos(self, tip_x, tip_y, im_size_nm=None):
                """
                Set the tip position
                """
                self.createc_controller.set_xy_nm([tip_x, tip_y])
                self.createc_controller.offset_nm= np.array([tip_x, tip_y])
                if im_size_nm is not None:
                        self.createc_controller.im_size_nm = im_size_nm  
                # img=self.createc_controller.scan_image()
                # self.createc_controller.stm.quicksave()
                # return img       
        
        def reset_max_radius_cellsize(self, cellsize: float=10, max_radius: float=300) -> None:
                """
                Reset the max_radius and cellsize
                """
                self.max_radius = max_radius
                self.cellsize = cellsize
                self.num_cell = int(self.max_radius/self.cellsize)

        def get_next_region(self, tip_x, tip_y, scan_ref_x_center=None, scan_ref_y_center=None, forbid_radius=20, check_simi_forbid_radius=25, move_upper_limit=400, approach_limit=[-300, 300, -300, 300], spiralreg=1.0, mn=100, detect_similarity=True, check_similarity=None):
                """
                Get the next good closest tip position
                """
                if scan_ref_x_center is None:
                        scan_ref_x_center=self.scan_ref_x_center
                if scan_ref_y_center is None:
                        scan_ref_y_center=self.scan_ref_y_center
                if forbid_radius is None:
                        forbid_radius=self.forbid_radius
                if check_simi_forbid_radius is None:
                        check_simi_forbid_radius=self.check_simi_forbid_radius
                if approach_limit is None:
                        approach_limit=self.approach_limit
                if move_upper_limit is None:
                        move_upper_limit=self.move_upper_limit

                move_limit=move_upper_limit/float(self.cellsize)
                found=False
                self.get_approach_area()
                self.forbidden_area(forbid_radius=forbid_radius)
                for i in range(-self.num_cell, self.num_cell+1):
                        for j in range(-self.num_cell,self.num_cell+1):
                                # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='orange',s=1)
                                if self.mask[i+self.num_cell,j+self.num_cell] == True:
                                        
                                # plt.gca().add_patch(patches.Rectangle((x+i*self.cellsize-4,  y+j*self.cellsize), 8, 8, fill=False, edgecolor='grey', lw=2))
                                        # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='g',s=1)
                                        continue

                                dist_euclidian = np.sqrt(float((i*self.cellsize)**2)+ float((j*self.cellsize)**2)) #Euclidian distance
                                if (dist_euclidian>self.move_upper_limit):
                                        # plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='blue')
                                        continue

                                if tip_x+i*self.cellsize<approach_limit[0] or tip_x+i*self.cellsize>approach_limit[1] or tip_y+j*self.cellsize<approach_limit[2] or tip_y+j*self.cellsize>approach_limit[3]:
                                # new_x_all.append(ref_x+i*radius*1.5)
                                # new_y_all.append(ref_y+j*radius*1.5)
                                        plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='blue', s=1)
                                        continue

                                # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='yellow')

                                #dist_manhattan = abs(i*self.cellsize)+abs(j*self.cellsize) #Manhattan distance

                                dist_manhattan = max(abs(i*self.cellsize), abs(j*self.cellsize))  #Manhattan distance
                                                
                                dist=(spiralreg*dist_euclidian+dist_manhattan)

                                if detect_similarity:
                                        if check_similarity is None:
                                                check_similarity = [[tip_x, tip_y]]
                                        # print('check_similarity:', check_similarity)
                                        check_similarity_array=np.array(check_similarity)-np.array([tip_x+i*self.cellsize, tip_y+j*self.cellsize])
                                        # print(check_similarity_array)
                                        simi_points_dist=np.array([np.sqrt(check_similarity_array[k][0]**2+check_similarity_array[k][1]**2) for k in range(len(check_similarity_array))]).min()

                                else:
                                        simi_points_dist=1000000

                                # print('ssss', similarity_dist, sim_forbiden_radius)
                                # print('dist', dist, move_limit)
              
                                if simi_points_dist > check_simi_forbid_radius:
                                        if dist < move_limit  or (not found):
                                                found=True
                                                move_limit=dist
                                                tip_x_move=i
                                                tip_y_move=j
                                                plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='y')
                                        elif dist==move_limit:
                                                if np.sqrt(float((tip_x+i*self.cellsize-scan_ref_x_center)**2)+float((tip_y+j*self.cellsize-scan_ref_y_center)**2))<np.sqrt(float((tip_x+tip_x_move*self.cellsize-scan_ref_x_center)**2)+ float((tip_y+tip_y_move*self.cellsize-scan_ref_y_center)**2)):
                                                        found=True
                                                        move_limit=dist
                                                        tip_x_move=i
                                                        tip_y_move=j
                                                        plt.scatter(tip_x+i*self.cellsize,tip_y+j*self.cellsize,color='y')
                                # print(i, j, move_limit, dist_manhattan, dist_euclidian, dist)

                try:
                        plt.scatter(tip_x+tip_x_move*self.cellsize,tip_y+tip_y_move*self.cellsize,color='r')            
                        return tip_x+tip_x_move*self.cellsize, tip_y+tip_y_move*self.cellsize
                except:
                        print('No good cloest point found, return the previous tip position')
                        return tip_x+10, tip_y+10
                


        
        def get_approach_area(self):
                """
                Get the approach area
                """
                print("starting new approach area...")
                self.mask = np.zeros((2*self.num_cell+1,2*self.num_cell+1),dtype=np.bool_)

                
        def forbidden_area(self, forbid_radius: float = 100) -> tuple:
                """
                Check if the coordinates x, y is in the forbidden area

                Parameters
                ----------
                forbiden_r: float
                forbidden area radius in nm

                Return
                ------
                mask: array_like
                whether the coordinates is in the forbidden area
                """
                for i in range(-self.num_cell, self.num_cell+1):
                        for j in range(-self.num_cell, self.num_cell+1):
                                if self.mask[i+self.num_cell, j+self.num_cell] == True:
                                        continue
                                dist=np.sqrt((i)**2+(j)**2) #Euclidian distance
                                max_dist=forbid_radius/self.cellsize
                                if dist<max_dist:
                                        self.mask[i+self.num_cell, j+self.num_cell] = True
                np.save("mask.npy", self.mask)