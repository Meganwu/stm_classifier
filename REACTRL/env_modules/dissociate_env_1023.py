from .Env_new import RealExpEnv
from .createc_control import Createc_Controller
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .assign_and_anchor import assignment, align_design, align_deisgn_stitching, get_atom_and_anchor

from .image_module_ellipse import image_detect_blobs

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
from .get_atom_coordinate import get_atom_coordinate_nm
import findiff
from .atom_jump_detection import AtomJumpDetector_conv
import os
from matplotlib import pyplot as plt, patches


from collections import namedtuple
dissociate_data = namedtuple('dissociate_data',['time','x','y','current','dI_dV','topography'])


class DissociateEnv:
        def __init__(self,
                diss_radius_nm,
                goal_nm,
                max_z_nm,
                max_mvolt,
                max_pcurrent_to_mvolt_ratio,
                template,
                current_jump,
                im_size_nm,
                offset_nm,
                manip_limit_nm,
                pixel,
                template_max_y,
                scan_mV,
                max_len,
                load_weight,
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
                scan_ref_x_center = None, # reference points x for seeking molecules
                scan_ref_y_center = None, # reference points y for seeking molecules
                move_upper_limit = 400, # limited movement range for seeking molecules
                approach_limit=[-180, 180, -180, 180], # limited range for seeking molecules
                mol_candidates = None, # mol_candidates for seeking molecules
                real_mols = None, # real_mols for seeking molecules
                fake_mols = None, # fake_mols for seeking molecules
                tipform_section = None, # tipform_section for seeking molecules # TODO list
                ):
                
                self.diss_radius_nm = diss_radius_nm
                self.goal_nm = goal_nm
                self.max_z_nm = max_z_nm
                self.max_mvolt = max_mvolt
                self.max_pcurrent_to_mvolt_ratio = max_pcurrent_to_mvolt_ratio
                self.pixel = pixel


                self.template = template
                args = im_size_nm, offset_nm, pixel, scan_mV
                self.createc_controller = Createc_Controller(*args)
                self.current_jump = current_jump
                self.manip_limit_nm = manip_limit_nm
                if self.manip_limit_nm is not None:
                        print('manipulation limit:', self.manip_limit_nm)
                        self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])
                self.offset_nm = offset_nm
                self.len_nm = im_size_nm

                self.default_reward = -1
                self.default_reward_done = 1
                self.max_len = max_len
                self.correct_drift = correct_drift
                self.atom_absolute_nm = None
                self.atom_relative_nm = None
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


                self.mol_candidates = mol_candidates
                self.real_mols = real_mols
                self.fake_mols = fake_mols

                # find next scanning images

                self.cellsize = cellsize                          # the size of the scanning image
                self.max_radius = max_radius
                self.forbid_radius = forbid_radius
                self.limit_forbid_radius = forbid_radius*3
                self.check_simi_forbid_radius = check_simi_forbid_radius  # check whether new position for scanning images is too close to previous scanning regions
                self.num_cell = int(self.max_radius/self.cellsize)
                self.check_similarity = check_similarity
                self.scan_ref_x_center = scan_ref_x_center
                self.scan_ref_y_center = scan_ref_y_center
                self.move_upper_limit = move_upper_limit  # the maximum distance for seeking molecules from reference points.
                self.approach_limit = approach_limit
        

        def reset(self, updata_conv_net=True):
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
                self.len = 0

        #TODO  build atom_diss_detector.currents_val

                if (len(self.atom_move_detector.currents_val)>self.atom_move_detector.batch_size) and update_conv_net:
                        accuracy, true_positive, true_negative = self.atom_move_detector.eval()
                        self.accuracy.append(accuracy)
                        self.true_positive.append(true_positive)
                        self.true_negative.append(true_negative)
                        self.atom_move_detector.train()

                if (self.atom_absolute_nm is None) or (self.atom_relative_nm is None):
                        self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()

                if self.out_of_range(self.atom_absolute_nm, self.inner_limit_nm):
                        print('Warning: atom is out of limit')
                        self.pull_atom_back()
                        self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()


                #goal_nm is set between 0.28 - 2 nm (Cu)
                goal_nm = self.lattice_constant + np.random.random()*(self.goal_nm - self.lattice_constant)
                print('goal_nm:',goal_nm)

                self.atom_start_absolute_nm, self.atom_start_relative_nm = self.atom_absolute_nm, self.atom_relative_nm
                # self.destination_relative_nm, self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_relative_nm, self.atom_start_absolute_nm, goal_nm)

                # self.state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))
                # self.dist_destination = goal_nm
                # img_forward, img_backward, offset_nm, len_nm = env.createc_controller.scan_image()

                img = self.get_state(self.offset_nm[0], self.offset_nm[1])

                ell_x, ell_y, ell_len, ell_wid = self.measure_fragment(img)   # analyze forward or backward images
                self.state = np.array([ell_x, ell_y, ell_len, ell_wid])

                info = {'start_absolute_nm':self.atom_start_absolute_nm, 'start_relative_nm':self.atom_start_relative_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                        'goal_relative_nm':self.destination_relative_nm, 'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b,
                        'start_relative_nm_f':self.atom_relative_nm_f, 'start_relative_nm_b':self.atom_relative_nm_b}
                return self.state, info

        def step(self, action):
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

                rets = self.action_to_diss_input(action)
                x_start_nm, y_start_nm, z_nm, mvoltage, pcurrent = rets
                args = x_start_nm, y_start_nm, z_nm, mvoltage, pcurrent
                diss_data = self.step_dissociate(*args)

                info={'diss_data': diss_data, 'mol_x_nm': self.mol_absolute_nm[0], 'mol_y_nm': self.mol_absolute_nm[1], 'x_start_nm': x_start_nm, 'y_start_nm': y_start_nm, 'z_nm': z_nm, 'mvoltage': mvoltage, 'pcurrent': pcurrent}
                topography = diss_data.topography

                done=False
                self.len+=1
                done = self.len==self.max_len
                if not done:
                        diss=self.old_detect_diss(topography)
                if done or diss:
                        save_scan_image(env, x_nm=x_start_nm, y_nm=y_start_nm, pixel=self.pixel, scan_len_nm=self.len_nm, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
             
                        if np.abs(img_forward_next - img_forward)>1e-6 and self.measure_fragment(img_forward_next)!=self.measure_fragment(img_forward):  # if the image is obviously different from the previous one
                                done=True
        # if no changes in the image or slight changes but no breakage of covalent bonds
                next_state=self.measure_fragment(img_forward_next)  # return ell_x, ell_y, ell_len, ell_wid
                # reward=self.compute_reward(self.state, next_state)  # or reward=self.compute_reward(self.image_forward, image_forward_next)
                reward=self.compute_reward(topography)

                info  |= {'dist_destination':self.dist_destination,
                        'atom_absolute_nm':self.atom_absolute_nm, 'atom_relative_nm':self.atom_relative_nm, 'atom_absolute_nm_f':self.atom_absolute_nm_f,
                        'atom_relative_nm_f' : self.atom_relative_nm_f, 'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_relative_nm_b':self.atom_relative_nm_b,
                        'img_info':self.img_info}   #### not very sure about the img_info
                
                self.state=next_state


                return next_state, reward, done, info

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


    

        def compute_reward(self, topography: np.array)->float:
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
                if self.old_detect_diss(topography):
                        reward = 1.0
                else:
                        reward = 0.0
                pass

        def action_to_diss_input(self, action):
                """
                Convert the action to the input for the dissociation

                Parameters
                ----------
                action: array_like 7D
                        the action from the RL agent

                Returns
                -------
                x_start_nm, y_start_nm, x_end_nm, y_end_nm, z_nm, mvolt, pcurrent: float
                        the input for the dissociation
                """
                x_start_nm = action[0]*self.diss_radius_nm+self.mol_absolute_nm[0]
                y_start_nm = action[1]*self.diss_radius_nm+self.mol_absolute_nm[1]
                z_nm = np.clip(action[2], a_min = None, a_max=1)*(self.max_z_nm-self.min_z_nm)/2+np.mean(self.min_z_nm+self.max_z_nm)
                mvoltage = np.clip(action[3], a_min = None, a_max=1)*self.max_mvolt
                pcurrent = np.clip(action[4], a_min = None, a_max=1)*self.max_pcurrent_to_mvolt_ratio
                return x_start_nm, y_start_nm, z_nm, mvoltage, pcurrent



    
        def step_dissociate(self, x_start_nm, y_start_nm, z_nm, mvoltage, pcurrent):
                """
                Execute the action in Createc

                Parameters
                ----------
                x_start_nm, y_start_nm: float
                        start position of the tip dissociation in nm
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



                x_start_nm = x_start_nm + self.atom_absolute_nm[0]
                y_start_nm = y_start_nm + self.atom_absolute_nm[1]

                x_kwargs = {'a_min':self.manip_limit_nm[0], 'a_max':self.manip_limit_nm[1]}
                y_kwargs = {'a_min':self.manip_limit_nm[2], 'a_max':self.manip_limit_nm[3]}

                x_start_nm = np.clip(x_start_nm, **x_kwargs)
                y_start_nm = np.clip(y_start_nm, **y_kwargs) 

                pos = x_start_nm, y_start_nm, z_nm
                params = mvoltage, pcurrent, self.offset_nm, self.len_nm


                data = self.createc_controller.diss_manipulation(*pos, *params)


                return data
        
        def old_detect_diss(self, topography, thres=2.0):
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
                diff_topography=topography[0:512].sum()-topography[512:].sum()

                if diff_topography>thres:
                        return True
                else:
                        print('Different prediction shows no dissociation')
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

        def detect_mols(self, img_large, img_large_x_nm, img_large_y_nm, scan_len_nm_large=10, pixel_large=128):
                """
                Detect molecules based on the image

                Parameters
                ----------
                img: array_like
                        the STM image

                Returns
                -------
                molecular postions
                        whether there are molecules
                """
                detect_mols_rough=image_select_points(img_large, edges=False, x_nm=img_large_x_nm, y_nm=img_large_y_nm, len_nm=scan_len_nm_large, pixel=pixel_large, dist_thres=1, absolute_pos_nm=True)
                if len(detect_mols_rough)>1:
                        detect_one_mol=detect_indiv_mols(detect_mols_rough)
                elif len(detect_mols_rough)==1:
                        detect_one_mol=detect_mols_rough
                else:
                        detect_one_mol=[]
                return detect_one_mol 
        
        def adjust_mol_pos(self, mol_pos, scan_len_nm_small=3.5, pixel_small=128, save_img_folder=None, save_data_folder=None, img_name=None, data_name=None):
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
                scan_data=save_scan_image(env, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                detect_mols_pos=image_select_points(img_small, edges=False, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, dist_thres=1, absolute_pos_nm=True)
                if len(detect_mols_pos)==1:
                        mol_center_x_nm, mol_center_y_nm = detect_indiv_mol_center(img_small, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
                        if mol_center_x_nm>scan_mol_x_nm-scan_len_nm_small/2+scan_len_nm_small/4 and mol_center_x_nm<scan_mol_x_nm+scan_len_nm_small/2-scan_len_nm_small/4 and mol_center_y_nm>scan_mol_y_nm-scan_len_nm_small/4 and mol_center_y_nm<scan_mol_y_nm+scan_len_nm_small*3/4:
                                pass
                        else:
                                scan_mol_x_nm=mol_center_x_nm
                                scan_mol_y_nm=mol_center_y_nm-scan_len_nm_small/2 
                                scan_data=save_scan_image(env, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                                img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_before, img_name))
                                mol_center_x_nm, mol_center_y_nm = detect_indiv_mol_center(img_small, x_nm=scan_mol_x_nm, y_nm=scan_mol_y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
                        return [scan_data, mol_center_x_nm, mol_center_y_nm]
                else:
                        return None

        # def find_one_mol(self, mol_candidates=None, real_mols=None, fake_mols=None):
        #         if mol_candidates is None:
        #                 mol_candidates=self.mol_candidates
        #         if real_mols is None:
        #                 real_mols=self.real_mols
        #         if fake_mols is None:
        #                 fake_mols=self.fake_mols
        #         for i in range(len(candidate_mols)):
        #                 mol_pos=candidate_mols[i]
        #                 scan_mol_data=self.adjust_mol_pos(mol_pos, scan_len_nm_small=3.5, pixel_small=128, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
        #                 img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
        #                 img_prop_small=mol_property(img_small, offset_x_nm=mol_pos[0], offset_y_nm=mol_pos[1], len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
        #                 img_prop_small.center_points_from_contours()
        #                 one_mol=img_prop_small.detect_mols_from_contours
        #                 if len(one_mol)==1:
        #                         real_mols.append(one_mol)
        #                         candidate_mols.remove(mol_pos)
        #                         break
        #                 else:
        #                         candidate_mols.remove(mol_pos)
        #                         fake_mols.append(mol_pos)


        def get_state(self,
                x_nm: float = None,
                y_nm: float = None,
                scan_len_nm_large: int = 10,
                pixel_large: int = 128,
                scan_len_nm_small: float = 3.5,
                pixel_small: int = 128,
                new_scan_region: bool = False,
                check_similarity: Any | None = None,
                candidate_mols: Any | None = None               # existing_mols is a list of possible individual mols based on rought detection
                real_mols: Any | None = None,                # checked_mols is a list of mols that have been checked
                fake_mols: Any | None = None,                    # mol_fake is a list of mols that are not real mols
                ) -> None:
                """
                Get the state of the environment

                Returns
                -------
                self.state: array_like
                """

                if check_similarity is None:
                        check_similarity = [[x_nm, y_nm]]

                if self.check_similarity is None:
                        self.check_similarity = check_similarity


                
                done = False
                max_time_limit=5
                seek_time=0

                if new_scan_region:
                        found_mol=False
                        while not found_mol:
                                while candidate_mols is None:
                                        x_nm, y_nm=self.get_next_region(x_nm, y_nm, detect_similarity=True, check_similarity=check_similarity)
                                        check_similarity.append([x_nm, y_nm])
                                        seek_time+=1
                                        save_scan_image(env, x_nm=x_nm, y_nm=y_nm, pixel=pixel_large, scan_len_nm=scan_len_nm_large, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                                        img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                                        img_prop_large=mol_property(img_large, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_large, pixel=pixel_large, absolute_pos_nm=True)
                                        detect_mols=img_prop_large.detect_mol_from_points()
                                        if len(detect_mols)>0:
                                                candidate_mols=detect_mols

                                for i in range(len(candidate_mols)):
                                        mol_pos=candidate_mols[i]
                                        scan_mol_data=self.adjust_mol_pos(mol_pos, scan_len_nm_small=3.5, pixel_small=128, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                                        img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                                        img_prop_small=mol_property(img_small, offset_x_nm=mol_pos[0], offset_y_nm=mol_pos[1], len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
                                        img_prop_small.center_points_from_contours()
                                        img_prop_small.contours_property()

                                        if len(img_prop_small.contours)>0 and len(mg_prop_small.contours_max)>5:
                                                self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_weight, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                                                checked_mols.append(one_mol)
                                                candidate_mols.remove(mol_pos)
                                                break
                                        else:
                                                print('No Molecule detected')
                                                candidate_mols.remove(mol_pos)
                                                fake_mols.append(mol_pos)


                else:
                        save_scan_image(env, x_nm=x_nm, y_nm=y_nm, pixel=pixel_small, scan_len_nm=scan_len_nm_small, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                        img_small=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                        img_prop_small=mol_property(img_small, offset_x_nm=x_nm, offset_y_nm=y_nm, len_nm=scan_len_nm_small, pixel=pixel_small, absolute_pos_nm=True)
                        img_prop_small.contours_property()
                        try:
                                if len(img_prop_small.contours)>0 and len(mg_prop_small.contours_max)>5:
                                        self.state=[img_prop_small.ellipse_x, img_prop_small.ellipse_y, img_prop_small.ellipse_weight, img_prop_small.ellipse_height, img_prop_small.ellipse_angle]
                        except:
                                print('No Molecule detected')


                

                       






       

        def get_state_old(self, 
                      tip_x, 
                      tip_y, 
                      scan_len_nm_large=10, 
                      pixel_large=256, 
                      pixel_small=128,
                      forbid_radius=35, 
                      new_tippos=False, 
                      check_similarity=None):
                """
                Get the state of the environment

                Returns
                -------
                self.state: array_like
                """
                if not new_tippos:
                        self.set_new_tippos(tip_x, tip_y)

                if check_similarity is None:
                        check_similarity = [[tip_x, tip_y]]

                if self.check_similarity is None:
                        self.check_similarity = check_similarity
                else:
                        self.check_similarity.append([tip_x, tip_y])

                no_atom = True
                no_debris = False
                no_crash = False
                epoch = 0

                while no_atom or not no_debris or not no_crash:
                        plt.text(tip_x, tip_y+5, epoch)
                        plt.gca().add_patch(patches.Circle((tip_x, tip_y), forbid_radius, fill=False, color='orange'))
                        save_scan_image(env, x_nm=tip_x, y_nm=tip_y, pixel=pixel_large, scan_len_nm=scan_len_nm_large, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                        img_large=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                        detect_one_mols=self.detect_mols(img_large, img_large_x_nm, img_large_y_nm, scan_len_nm_large=scan_len_nm_large, pixel_large=pixel_large)
                        if len(detect_one_mols)>0:
                                for i in range(len(detect_one_mols)):
                                        scan_data=self.adjust_mol_pos(detect_one_mols[i], scan_len_nm_small=3.5, pixel_small=128, save_img_folder=save_img_folder, save_data_folder=save_data_folder, img_name=img_name, data_name=data_name)
                                        if scan_data is not None:
                                                no_atom=False
                                                break
                        if scan_data is not None:
                                mol_pos=scan_data[1:]
                                img=cv2.imread('%s/img_forward_%s.png' % (save_img_folder, img_name))
                                topography=scan_data[0].img_forward.flatten()
                                no_debris = self.debris_detection(topography)
                                no_crash = self.crash_detection(topography)
                                if not no_atom and no_debris and no_crash:
                                        pass
                                else:
                                        tip_x, tip_y=self.get_nextgoodcloest(tip_x, tip_y, forbid_radius=forbid_radius, check_similarity=check_similarity)
                                        if tip_x is None:
                                                tip_x, tip_y=check_similarity[-1][0], check_similarity[-1][1]
                                                print('No molecule found, expand the searching radius')              
                                                forbid_radius=forbid_radius*2
                                                self.max_radius=self.max_radius*2

                                        else:
                                                self.reset_max_radius_cellsize()

                                        self.set_new_tippos(tip_x, tip_y)



                        check_similarity.append([tip_x, tip_y])
                        self.check_similarity.append([tip_x, tip_y])
                        epoch+=1
                        
                        # if tip_y > 50: # test
                        if forbid_radius > self.limit_forbid_radius:
                                print('No molecule found, expand the searching radius*3 and stop searching, need to change approach region')
                                break

                # TODO list image processing

                # ell_x, ell_y, ell_len, ell_wid = self.measure_fragment(img)   # analyze forward or backward images
                # self.state = np.array([ell_x, ell_y, ell_len, ell_wid])
                # retrun self.state  # return state


                return img, tip_x, tip_y, check_similarity   

        def set_new_tippos(self, tip_x, tip_y, im_size_nm=None):
                """
                Set the tip position
                """
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

        def get_next_region(self, tip_x, tip_y, scan_ref_x_center=None, scan_ref_y_center=None, forbid_radius=35, check_simi_forbid_radius=30, move_upper_limit=400, approach_limit=[-180, 180, -180, 180], spiralreg=1.0, mn=100, detect_similarity=True, check_similarity=None):
                """
                Get the next good closest tip position
                """
                if scan_ref_x_center is None:
                        scan_ref_x_center=self.scan_ref_x_center
                if scan_ref_x_center_y is None:
                        scan_ref_x_center_y=self.scan_ref_x_center_y
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
                                                if np.sqrt(float((tip_x+i*self.cellsize-scan_ref_x_center)**2)+ float((tip_y+j*self.cellsize-scan_ref_y_center)**2))<np.sqrt(float((tip_x+tip_x_move*self.cellsize-scan_ref_x_center)**2)+ float((tip_y+tip_y_move*self.cellsize-scan_ref_y_center)**2)):
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
                        return tip_x, tip_y
                


        
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