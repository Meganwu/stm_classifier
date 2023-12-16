from .Env_new import RealExpEnv
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design
from .image_module_ellipse import image_detect_blobs, image_detect_edges, image_select_points

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import pandas as pd


def assign_mol_design(img, x_nm=1.702, y_nm=-164.915, len_nm=20, pixel=256, grid_num=10): 
    '''based on the detected blobs, constructe target points and assign the original points to the target points'''
    
    selected_points=image_select_points(img, edges=False, dist_limit=1, x_nm=x_nm, y_nm=y_nm, len_nm=len_nm, pixel=pixel, dist_thres=1)

    # plt.show()

    # mean_x_pixel=np.mean([selected_points[i][0] for i in range(len(selected_points))])
    # mean_y_pixel=np.mean([selected_points[i][1] for i in range(len(selected_points))])
    mean_x_pixel=128
    mean_y_pixel=128
    mean_x=x_nm-len_nm/2+len_nm/pixel*mean_x_pixel
    mean_y=y_nm+len_nm/pixel*mean_y_pixel

    
    selected_points_nm=[[selected_points[i][0]*len_nm/pixel+x_nm-len_nm/2, selected_points[i][1]*len_nm/pixel+y_nm] for i in range(len(selected_points))]
    
    xv, yv = np.meshgrid(np.linspace(x_nm-len_nm/2, x_nm+len_nm/2, grid_num), np.linspace(y_nm, y_nm+len_nm, grid_num))
    xv=xv.flatten()
    yv=yv.flatten()

    target_points=[]

    plt.scatter([mean_x], [mean_y], c='g', s=100)


    target_points={'x': xv, 'y': yv}
    target_points=pd.DataFrame(target_points)
    target_points.loc[:, 'dist']=np.sqrt((target_points.x-mean_x)**2+(target_points.y-mean_y)**2)
    target_points=target_points.sort_values(by='dist', ascending=True, ignore_index=True)
    target_points=target_points.iloc[:len(selected_points), :]

    # based on the farthest points away from the designed center, assign the original points to the target points

    target_points_order=target_points.sort_values(by='dist', ascending=False, ignore_index=True)
    selected_points_nm_order=[]
    for i in range(len(target_points_order)):
        selected_points_nm_array=np.array(selected_points_nm)
        x=target_points_order.x[i]
        y=target_points_order.y[i]
        pos=np.argmin(cdist(selected_points_nm_array,np.array([x, y]).reshape(1, -1)))
        selected_points_nm.remove(selected_points_nm[pos])
        selected_points_nm_order.append([selected_points_nm_array[pos][0], selected_points_nm_array[pos][1]])
        print(pos)

    target_points_order=target_points_order[['x', 'y']].values.tolist()
    for i in range(len(selected_points_nm_order)):
    # for i in range(10):
        plt.text(selected_points_nm_order[i][0], selected_points_nm_order[i][1], str(i))
        plt.scatter(selected_points_nm_order[i][0], selected_points_nm_order[i][1])
        plt.scatter(target_points_order[i][0], target_points_order[i][1], c='b')
        plt.plot([selected_points_nm_order[i][0],target_points_order[i][0]] , [selected_points_nm_order[i][1], target_points_order[i][1]])

    plt.xlim(x_nm-len_nm/2, x_nm+len_nm/2)
    plt.ylim(y_nm+len_nm, y_nm)
    
    return selected_points_nm_order, target_points_order
    


def assignment(start, goal):
    """
    Assign start to goal with the linear_sum_assignment function and setting the cost matrix to the distance between each start-goal pair

    Parameters
    ----------
    start, goal: array_like
        start and goal positions

    Returns
    -------
    np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost: array_like
            sorted start and goal positions, and their distances

    total_cost: float
            total distances
    
    row_ind, col_ind: array_like
            Indexes of the start and goal array in sorted order
    """
    cost_matrix = cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost, row_ind, col_ind

def align_design(atoms, design):
    """
    Move design positions and assign atoms to designs to minimize total manipulation distance 

    Parameters
    ----------
    atoms, design: array_like
        atom and design positions

    Returns
    -------
    atoms_assigned, design_assigned: array_like
            sorted atom and design (moved) positions
    
    anchor: array_like
            position of the atom that will be used as the anchor
    """
    assert atoms.shape == design.shape
    c_min = np.inf
    for i in range(atoms.shape[0]):
        for j in range(design.shape[0]):
            a = atoms[i,:]
            d = design[j,:]
            design_ = design+a-d
            a_index = np.delete(np.arange(atoms.shape[0]), i)
            d_index = np.delete(np.arange(design.shape[0]), j)
            a, d, _, c, _, _ = assignment(atoms[a_index,:], design_[d_index,:])
            if (c<c_min):
                c_min = c
                atoms_assigned, design_assigned = a, d
                anchor = atoms[i,:]
    return atoms_assigned, design_assigned, anchor

def align_deisgn_stitching(all_atom_absolute_nm, design_nm, align_design_params):
    """
    Shift the designs to match the atoms based on align_design_params. 
    Assign atoms to designs to minimize total manipulation distance.
    Get the obstacle list from align_design_params

    Parameters
    ----------
    all_atom_absolute_nm, design_nm: array_like
        atom and design positions

    align_design_params: dict
        {'atom_nm', 'design_nm', 'obstacle_nm'} 

    Returns
    -------
    atoms, designs: array_like
            sorted atom and design (moved) positions
    
    anchor_atom_nm: array_like
            position of the atom that will be used as the anchor
    """
    anchor_atom_nm = align_design_params['atom_nm']
    anchor_design_nm = align_design_params['design_nm']
    obstacle_nm = align_design_params['obstacle_nm']
    assert anchor_design_nm.tolist() in design_nm.tolist()
    dist = cdist(all_atom_absolute_nm, anchor_atom_nm.reshape((-1,2)))
    anchor_atom_nm = all_atom_absolute_nm[np.argmin(dist),:]
    atoms = np.delete(all_atom_absolute_nm, np.argmin(dist), axis=0)
    dist = cdist(design_nm, anchor_design_nm.reshape((-1,2)))
    designs = np.delete(design_nm, np.argmin(dist), axis=0)
    designs += (anchor_atom_nm - anchor_design_nm)
    if obstacle_nm is not None:
        obstacle_nm[:,:2] = obstacle_nm[:,:2]+(anchor_atom_nm - anchor_design_nm)
    return atoms, designs, anchor_atom_nm, obstacle_nm

def get_atom_and_anchor(all_atom_absolute_nm, anchor_nm):
    """
    Separate the positions of the anchor and the rest of the atoms 

    Parameters
    ----------
    all_atom_absolute_nm, anchor_nm: array_like
        positions of all the atoms and the anchor

    Returns
    -------
    atoms_nm, new_anchor_nm: array_like
            positions of all the atoms (except the anchor) and the anchor
    """
    new_anchor_nm, anchor_nm, _, _, row_ind, _ = assignment(all_atom_absolute_nm, anchor_nm)
    atoms_nm = np.delete(all_atom_absolute_nm, row_ind, axis=0)
    return atoms_nm, new_anchor_nm