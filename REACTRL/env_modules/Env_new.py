from .createc_control import Createc_Controller
import numpy as np
from .get_atom_coordinate import get_atom_coordinate_nm
import findiff
from .atom_jump_detection import AtomJumpDetector_conv
import os
from matplotlib import pyplot as plt, patches

class RealExpEnv:
    """
    Environment for reinforcement learning through interaction with real-world STM
    """
    def __init__(self,
                 step_nm,
                 max_mvolt,
                 max_pcurrent_to_mvolt_ratio,
                 goal_nm,
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
                 ):

        self.step_nm = step_nm
        self.max_mvolt = max_mvolt
        self.max_pcurrent_to_mvolt_ratio = max_pcurrent_to_mvolt_ratio
        self.pixel = pixel
        self.goal_nm = goal_nm

        self.template = template
        args = im_size_nm, offset_nm, pixel, scan_mV
        self.createc_controller = Createc_Controller(*args)
        self.current_jump = current_jump
        self.manip_limit_nm = manip_limit_nm
        if self.manip_limit_nm is not None:
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

        self.cellsize = cellsize
        self.max_radius = max_radius
        self.num_cell = int(self.max_radius/self.cellsize)

    def reset(self, update_conv_net = True):
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
        #goal_nm is set between 0.28 - 2 nm
        goal_nm = self.lattice_constant + np.random.random()*(self.goal_nm - self.lattice_constant)
        print('goal_nm:',goal_nm)
        self.atom_start_absolute_nm, self.atom_start_relative_nm = self.atom_absolute_nm, self.atom_relative_nm
        self.destination_relative_nm, self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_relative_nm, self.atom_start_absolute_nm, goal_nm)

        self.state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))
        self.dist_destination = goal_nm

        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'start_relative_nm':self.atom_start_relative_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'goal_relative_nm':self.destination_relative_nm, 'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b,
                'start_relative_nm_f':self.atom_relative_nm_f, 'start_relative_nm_b':self.atom_relative_nm_b}
        return self.state, info

    def step(self, action):
        """
        Take a step in the environment with the given action

        Parameters
        ----------
        action: array_like

        Return
        ------
        next_state: np.array
        reward: float
        done: bool
        info: dict
        """
        rets = self.action_to_latman_input(action)
        x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent = rets
        args = x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent
        current_series, d = self.step_latman(*args)
        info = {'current_series':current_series, 'd': d, 'start_nm':  np.array([x_start_nm , y_start_nm]), 'end_nm':np.array([x_end_nm , y_end_nm])}
        done = False
        self.len+=1
        done = self.len == self.max_len
        if not done:
            jump = self.detect_current_jump(current_series)

        if done or jump:
            self.dist_destination, dist_start, dist_last = self.check_similarity()
            print('atom moves by: {:.3f} nm'.format(dist_start))
            oor = self.out_of_range(self.atom_absolute_nm, self.manip_limit_nm)
            in_precision_lim =  (self.dist_destination < self.precision_lim)
            too_far = (dist_start > 1.5*self.goal_nm)
            done = done or too_far or in_precision_lim or oor
            self.atom_move_detector.push(current_series, dist_last)

        next_state = np.concatenate((self.goal, (self.atom_absolute_nm -self.atom_start_absolute_nm)/self.goal_nm))
        reward = self.compute_reward(self.state, next_state)

        info |= {'dist_destination':self.dist_destination,
                'atom_absolute_nm':self.atom_absolute_nm, 'atom_relative_nm':self.atom_relative_nm, 'atom_absolute_nm_f':self.atom_absolute_nm_f,
                'atom_relative_nm_f' : self.atom_relative_nm_f, 'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_relative_nm_b':self.atom_relative_nm_b,
                'img_info':self.img_info}
        self.state = next_state
        return next_state, reward, done, info

    def calculate_potential(self, state):
        """
        Caculate the reward potential based on state

        Parameters
        ----------
        state: array_like

        Return
        ------
        -dist/self.lattice_constant: float
                reward potential
        dist: float
                the precision, i.e. the distance between the atom and the target
        """
        dist = np.linalg.norm(state[:2]*self.goal_nm - state[2:]*self.goal_nm)
        return -dist/self.lattice_constant, dist

    def compute_reward(self, state, next_state):
        """
        Caculate the reward based on state and next state

        Parameters
        ----------
        state, next_state: array_like

        Return
        ------
        reward: float
        """
        old_potential, _ = self.calculate_potential(state)
        new_potential, dist = self.calculate_potential(next_state)
        #print('old potential:', old_potential, 'new potential:', new_potential)
        reward = self.default_reward_done*(dist<self.precision_lim) + self.default_reward*(dist>self.precision_lim) + new_potential - old_potential
        return reward

    def scan_atom(self):
        """
        Take a STM scan and extract the atom position

        Return
        ------
        self.atom_absolute_nm: array_like
                atom position in STM coordinates (nm)
        self.atom_relative_nm: array_like
                atom position relative to the template position in STM coordinates (nm)
        """
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm}

        args = img_forward, offset_nm, len_nm, self.template, self.template_max_y, self.bottom
        atom_absolute_nm_f, atom_relative_nm_f, template_nm_f, template_wh_f  = get_atom_coordinate_nm(*args)

        args = img_backward, offset_nm, len_nm, self.template, self.template_max_y, self.bottom
        atom_absolute_nm_b, atom_relative_nm_b, template_nm_b, template_wh_b  = get_atom_coordinate_nm(*args)

        self.atom_absolute_nm_f = atom_absolute_nm_f
        self.atom_relative_nm_f = atom_relative_nm_f
        self.atom_absolute_nm_b = atom_absolute_nm_b
        self.atom_relative_nm_b = atom_relative_nm_b

        self.atom_absolute_nm, self.atom_relative_nm, template_nm, self.template_wh = 0.5*(atom_absolute_nm_f+atom_absolute_nm_b), 0.5*(atom_relative_nm_f+atom_relative_nm_b), 0.5*(template_nm_f+template_nm_b), 0.5*(template_wh_b+template_wh_f)

        if self.out_of_range(self.atom_absolute_nm, self.manip_limit_nm):
            print('Warning: atom is out of limit')
        if self.correct_drift:
            try:
                template_drift = template_nm - self.template_nm
                max_drift_nm = 0.5
                if (np.linalg.norm(template_drift)>max_drift_nm):
                    print('Move offset_nm from:{} to:{}'.format((self.createc_controller.offset_nm, self.createc_controller.offset_nm+template_drift)))
                    print('Move manip_limit_nm from:{} to:{}'.format((self.createc_controller.offset_nm, self.createc_controller.offset_nm+template_drift)))
                    self.createc_controller.offset_nm+=template_drift
                    self.manip_limit_nm += np.array((template_drift[0], template_drift[0], template_drift[1], template_drift[1]))
                    self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])
                    self.offset_nm = offset_nm
                    template_nm = self.template_nm
            except AttributeError:
                self.template_nm = template_nm
        self.template_nm = template_nm
        return self.atom_absolute_nm, self.atom_relative_nm

    def get_destination(self, atom_relative_nm, atom_absolute_nm, goal_nm):
        """
        Uniformly sample a new target that is within the self.inner_limit_nm

        Parameters
        ----------
        atom_absolute_nm: array_like
                atom position in STM coordinates (nm)
        atom_relative_nm: array_like
                atom position relative to the template position in STM coordinates (nm)
        goal_nm: array_like
                distance between the current atom position and the target position in nm

        Return
        ------
        destination_relative_nm: array_like
                target position relative to the template position in STM coordinates (nm)
        destination_absolute_nm: array_like
                target position in STM coordinates (nm)
        dr/self.goal_nm: array_like
                target position relative to the initial atom position in STM coordinates (nm)
        """
        while True:
            r = np.random.random()
            angle = 2*np.pi*r
            dr = goal_nm*np.array([np.cos(angle), np.sin(angle)])
            destination_absolute_nm = atom_absolute_nm + dr
            args = destination_absolute_nm, self.inner_limit_nm
            if not self.out_of_range(*args):
                break
        destination_relative_nm = atom_relative_nm + dr
        return destination_relative_nm, destination_absolute_nm, dr/self.goal_nm



    def old_detect_current_jump(self, current):
        """
        Estimate if atom has moved based on the gradient of the manipulation current trace

        Parameters
        ----------
        current: array_like
                manipulation current trace

        Return
        ------
        bool
            whether the atom has likely moved
        """
        if current is not None:
            diff = findiff.FinDiff(0,1,acc=6)(current)[3:-3]
            return np.sum(np.abs(diff)>self.current_jump*np.std(current)) > 2
        else:
            return False

    def detect_current_jump(self, current):
        """
        Estimate if atom has moved based on AtomJumpDetector_conv and the gradient of the manipulation current trace

        Parameters
        ----------
            current: array_like
                manipulation current trace

        Return
        ------
        bool
            whether the atom has likely moved
        """
        if current is not None:
            success, prediction = self.atom_move_detector.predict(current)
            old_prediction = self.old_detect_current_jump(current)
            print('CNN prediction:',prediction,'Old prediction:', old_prediction)
            if success:
                print('cnn thinks there is atom movement')
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
                print('CNN and old prediction both say no movement')
                return False
        else:
            print('CNN and old prediction both say no movement')
            return False

    def check_similarity(self):
        """
        Take a STM scan and calculate the distance between the atom and the target, the start position, the previous position

        Return
        ------
            dist_destination, dist_start, dist_last: float
                distance (nm) between the atom and the target, the start position, the previous position
        """
        old_atom_absolute_nm = self.atom_absolute_nm
        self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()
        dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        dist_start = np.linalg.norm(self.atom_absolute_nm - self.atom_start_absolute_nm)
        dist_last = np.linalg.norm(self.atom_absolute_nm - old_atom_absolute_nm)
        return dist_destination, dist_start, dist_last

    def out_of_range(self, nm, limit_nm):
        """
        Check if the coordinates nm is outside of the limit_nm

        Parameters
        ----------
        nm: array_like
            STM coordinates in nm
        limit_nm: array_like
            [left, right, up, down] limit in STM coordinates in nm

        Return
        ------
        bool
            whether the atom has likely moved
        """
        out = np.any((nm-limit_nm[[0,2]])*(nm - limit_nm[[1,3]])>0, axis=-1)
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


##############################################################################################################


    def tip_form_modify(self, dip_z_nm: float, voltage: float, x_nm: float, y_nm: float) -> None:
        """Perform tip forming
        Parameters
        ----------
        dip_z_nm : float
            Z approach value in A
        voltage : float
            Voltage value in V
        x_nm, y_nm : float
            STM coordinates (nm)
        """

        offset_nm=self.createc_controller.offset_nm
        len_nm=self.createc_controller.im_size_nm
        #offset_nm = self.createc_controller.get_offset_nm() 
        #len_nm = self.createc_controller.get_len_nm()
        self.createc_controller.stm.setparam('BiasVolt.[mV]',voltage)
        self.createc_controller.ramp_bias_mV(voltage)
        preamp_grain = 10**float(self.createc_controller.stm.getparam("Latmangain"))
        self.createc_controller.stm.setparam("LatmanVolt",  voltage) #(mV)
       # self.createc_controller.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)
        
        self.createc_controller.set_Z_approach(dip_z_nm)
        args = x_nm, y_nm, None, None, offset_nm, len_nm
        x_pixel, y_pixel, _, _ = self.createc_controller.nm_to_pixel(*args)
        self.createc_controller.stm.btn_tipform(x_pixel, y_pixel)
        self.createc_controller.stm.waitms(50)






# Find an atom according to the spiral trjactory
    def default_max_radius_cellsize(self, cellsize: float=10, max_radius: float=300) -> tuple:
        self.max_radius = max_radius
        self.cellsize = cellsize
        self.num_cell = int(self.max_radius/self.cellsize) #

    def approach_area(self):
        self.num_cell = int(self.max_radius/self.cellsize)

    def switch_approach_area(self):
        print("starting new approach area...")
        self.mask = np.zeros((2*self.num_cell+1,2*self.num_cell+1),dtype=np.bool_)


    #def ForbiddenArea(self, x, y, forbiden_radius: float = 0.5, cellsize: float=10, max_r: float=100, num_cell=None) -> tuple:
    def ForbiddenArea(self, forbiden_radius: float = 100) -> tuple:
        """
        Check if the coordinates x, y is in the forbidden area

        Parameters
        ----------
        x: float
            STM x coordinate in nm
        y: float
            STM y coordinate in nm
        forbiden_r: float
            forbidden area radius in nm
        cellsize: float
            Size of the gridcells used for motion planing in nm
        max_r: float
            maximum radius of the forbidden area in nm

        Return
        ------
        mask: array_like
            whether the coordinates is in the forbidden area
        """


        x=self.offset_nm[0]/self.cellsize
        y=self.offset_nm[1]/self.cellsize
        self.num_cell=int(self.max_radius/self.cellsize) #
        self.mask = np.zeros((2*self.num_cell+1, 2*self.num_cell+1), dtype=np.bool_)

        for i in range(-self.num_cell, self.num_cell+1):
            for j in range(-self.num_cell, self.num_cell+1):
                if self.mask[i+self.num_cell, j+self.num_cell] == True:
                    continue
                dist=np.sqrt((i)**2+(j)**2) #Euclidian distance
                max_dist=forbiden_radius/self.cellsize
                if dist<max_dist:
                    self.mask[i+self.num_cell, j+self.num_cell] = True
        np.save("mask.npy", self.mask)
        return self.mask

    def computeLocationIDs(self):
        leg = 0
        x = 0
        y = 0
        layer = 2
        ids = []
        cellsize_half = self.cellsize//2
        d = self.cellsize+cellsize_half
        while True:
            if leg == 0:
                x += 2
                if x == layer:
                    leg += 1
            elif leg == 1:
                y += 2
                if y == layer:
                    leg += 1
            elif leg == 2:
                x -= 2
                if -x == layer:
                    leg += 1
            elif leg == 3:
                y -= 2
                if -y == layer:
                    leg = 0
                    layer += 2

            if ((abs(x))*d+cellsize_half) > self.max_radius or ((abs(y))*d+cellsize_half) > self.max_radius:
                break
            ids.append((x,y))
        self.ids = ids
        return(ids)

    def SwitchManiArea(self):
        print("starting new approach area...")

        self.mask = np.zeros((2*self.num_cell+1,2*self.num_cell+1),dtype=np.bool)
        self.noDebresCounter=0

        if self.params['spiral'] == "simple":
            self.computeLocationIDs()

    def GetNextSpiralPoint(self, scansize=10):
         # scan image size of scanning region in nanometer
        x = None
        y = None
        for idx in self.ids:
            x, y = idx[0], idx[1]
            xid = x
            yid = y
            if self.mask[xid, yid] == True:
                continue
        return x*scansize*1.5, y*scansize*1.5

    def GetNextGoodClosest(self, x, y, initial_x=0, initial_y=0, forbiden_radius=35, sim_forbiden_radius=30, is_similarity=True, check_similarity=None, spiralreg=1.0, upper_limit_move=400, mn=100, approach_limit=[-100, 100, -100, 100]): #     # regularization factor for selecting the next region
    # large value leads to more regular path
        # Tip position in mask scale
        # x = x/float(self.cellsize)
        # y = y/float(self.cellsize)
        upper_limit_move=upper_limit_move/float(self.cellsize)     # Maximum allowed movement between to images in nm. If the required movement is larger, the approach area is changed
        move_limit = self.num_cell*100
        x_move=0
        y_move=0
        found=False

        self.switch_approach_area()
        self.ForbiddenArea(forbiden_radius=forbiden_radius)

        plt.scatter(x,y,color='b',s=50)
        
        for i in range(-self.num_cell,self.num_cell+1):
            for j in range(-self.num_cell,self.num_cell+1):
                # print('iiijjj', i, j)
                if self.mask[i+self.num_cell,j+self.num_cell] == True:
                    # plt.gca().add_patch(patches.Rectangle((x+i*self.cellsize-4,  y+j*self.cellsize), 8, 8, fill=False, edgecolor='grey', lw=2))
                    # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='g',s=1)
                    continue

                dist_euclidian = np.sqrt(float((i*self.cellsize)**2)+ float((j*self.cellsize)**2)) #Euclidian distance
                if (dist_euclidian>upper_limit_move):
                    # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='grey')
                    continue

                if x+i*self.cellsize<approach_limit[0] or x+i*self.cellsize>approach_limit[1] or y+j*self.cellsize<approach_limit[2] or y+j*self.cellsize>approach_limit[3]:
                    # new_x_all.append(ref_x+i*radius*1.5)
                    # new_y_all.append(ref_y+j*radius*1.5)
                    # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='black')
                    continue

                # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='yellow')

                #dist_manhattan = abs(i*self.cellsize-0)+abs(j*self.cellsize-0) #Manhattan distance

                dist_manhattan = max(abs(i*self.cellsize-0), abs(j*self.cellsize-0))  #Manhattan distance
                                    
                dist=(spiralreg*dist_euclidian+dist_manhattan)

                if is_similarity:
                    check_similarity_array=np.array(check_similarity)-np.array([x+i*self.cellsize, y+j*self.cellsize])
                        # print(check_similarity_array)
                    similarity_dist=np.array([np.sqrt(check_similarity_array[k][0]**2+check_similarity_array[k][1]**2) for k in range(len(check_similarity_array))]).min()

                else:
                    similarity_dist=1000000

                # print('ssss', similarity_dist, sim_forbiden_radius)
                # print('dist', dist, move_limit)
                if similarity_dist>sim_forbiden_radius:
                    if dist<move_limit or (not found):
                        found=True
                        move_limit=dist
                        x_move=i
                        y_move=j
                        # plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='r')
                    elif dist==move_limit:
                        if np.sqrt(float((x+i*self.cellsize-initial_x)**2)+ float((y+j*self.cellsize-initial_y)**2))<np.sqrt(float((x+x_move*self.cellsize-initial_x)**2)+ float((y+y_move*self.cellsize-initial_y)**2)):
                            found=True
                            move_limit=dist
                            x_move=i
                            y_move=j
                            plt.scatter(x+i*self.cellsize,y+j*self.cellsize,color='r')
                    print(i, j, move_limit, dist_manhattan, dist_euclidian, dist)
 
                   

        if not found:
            return None, None
        # Return tip position in nm


        return x+x_move*self.cellsize, y+y_move*self.cellsize
    

    def set_newtip_pos(self, tip_x, tip_y, im_size_nm=2):
        self.createc_controller.offset_nm=np.array([tip_x, tip_y])
        self.createc_controller.im_size_nm = im_size_nm
        image=self.createc_controller.scan_image()
        self.createc_controller.stm.quicksave()
        # self.createc_controller.set_xy_nm(np.array([tip_x, tip_y]))
        return image



