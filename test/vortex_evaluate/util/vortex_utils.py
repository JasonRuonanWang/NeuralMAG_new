# 2023.06.10

import numpy as np
import random

def numpy_roll(arr, shift, axis, pbc):
    """
    Re-defined numpy.roll(), including pbc judgement
    
    Arguments
    ---------
    arr      : Numpy Float(...)
               Array to be rolled
    shift    : Int
               Roll with how many steps
    axis     : Int
               Roll along which axis
    pbc      : Int or Bool
               Periodic condition for rolling; 1: pbc, 0: non-pbc
    
    Returns
    -------
    arr_roll : Numpy Float(...)
               arr after rolling
    """
    arr_roll = np.roll(arr, shift=shift, axis=axis)
    
    if not pbc:
        if axis == 0:
            if shift > 0:
                arr_roll[:shift, ...] = 0.0
            elif shift < 0:
                arr_roll[shift:, ...] = 0.0
            
        elif axis == 1:
            if shift > 0:
                arr_roll[:, :shift, ...] = 0.0
            elif shift < 0:
                arr_roll[:, shift:, ...] = 0.0
    
    return arr_roll


def get_winding(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )
    
    winding_density = (spin_xp[:,:,0] - spin_xm[:,:,0])/2 \
                      * (spin_yp[:,:,1] - spin_ym[:,:,1])/2 \
                    - (spin_xp[:,:,1] - spin_xm[:,:,1])/2 \
                      * (spin_yp[:,:,0] - spin_ym[:,:,0])/2
    winding_density[ model<=0 ] = 0
    winding_density[ model_bd ] = 0
    
    winding_density = winding_density / np.pi
    
    winding_abs = np.abs(winding_density).sum()
    winding_sum = winding_density.sum()
    
    return winding_density, winding_abs, winding_sum


def get_curl(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )

    # Get vortex core chirality (clockwise, counterclockwise)
    curl = (model > 0) * (( spin_xp[:,:,1] - spin_xm[:,:,1] )/2
                       -  ( spin_yp[:,:,0] - spin_ym[:,:,0] )/2 )

    curl[model_bd] = 0

    return curl


def analyze_winding(spin, model):
    spin_xp = np.roll(spin, shift=-1, axis=0)
    spin_xm = np.roll(spin, shift= 1 ,axis=0)
    spin_yp = np.roll(spin, shift=-1, axis=1)
    spin_ym = np.roll(spin, shift= 1, axis=1)
    
    # Ignore boundary
    model_bd = (model > 0) & \
               ( (numpy_roll(model, shift=-1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=0, pbc=False) <=0 ) |
                 (numpy_roll(model, shift=-1, axis=1, pbc=False) <=0 ) |
                 (numpy_roll(model, shift= 1, axis=1, pbc=False) <=0 ) )
    
    # Get vortex core number
    winding_density = (spin_xp[:,:,0] - spin_xm[:,:,0])/2 \
                      * (spin_yp[:,:,1] - spin_ym[:,:,1])/2 \
                    - (spin_xp[:,:,1] - spin_xm[:,:,1])/2 \
                      * (spin_yp[:,:,0] - spin_ym[:,:,0])/2
    winding_density[ model<=0 ] = 0
    winding_density[ model_bd ] = 0
    
    winding_density = winding_density / np.pi
    
    winding_abs = np.abs(winding_density).sum()
    winding_sum = winding_density.sum()
    
    vortex_cores = (winding_abs + winding_sum) /2
    antivortex_cores = (winding_abs - winding_sum) /2
    
    # Get vortex core polarity
    positive_abs = np.abs(winding_density[spin[:,:,2]>0]).sum()
    negative_abs = np.abs(winding_density[spin[:,:,2]<0]).sum()
    positive_sum = winding_density[spin[:,:,2]>0].sum()
    negative_sum = winding_density[spin[:,:,2]<0].sum()
    
    positive_vortices = (positive_abs + positive_sum) /2
    positive_antivortices = (positive_abs - positive_sum) /2
    negative_vortices = (negative_abs + negative_sum) /2
    negative_antivortices = (negative_abs - negative_sum) /2
    
    # Get vortex core chirality (clockwise, counterclockwise)
    curl = (model > 0) * (( spin_xp[:,:,1] - spin_xm[:,:,1] )/2
                       -  ( spin_yp[:,:,0] - spin_ym[:,:,0] )/2 )
    
    cw_vortices  = (winding_density * (winding_density>0))[curl<0].sum()
    ccw_vortices = (winding_density * (winding_density>0))[curl>0].sum()
    
    return np.around(vortex_cores), np.around(antivortex_cores), \
           np.around(positive_vortices), np.around(positive_antivortices), \
           np.around(negative_vortices), np.around(negative_antivortices), \
           np.around(cw_vortices),  np.around(ccw_vortices)

def get_randspin_2D(size=(1,1,1), split=2, add_perturbation=False, deg = 5, seed=None, perturbation_seed=None):
    """
    To get a random spin distribution in 2D view with optional perturbation
    
    Parameters:
    -----------
    size : tuple
        Size of the spin array (x, y, z)
    split : int
        Number of splits along x and y axes
    add_perturbation : bool
        Whether to add a perturbation to the spins
    seed : int or None
        Seed for random selection of base directions (None for no seed)
    perturbation_seed : int or None
        Seed for random perturbation direction (None for no seed)
        If None but seed is provided, will use seed+1
    
    Returns:
    --------
    numpy.ndarray
        Array of spin vectors with shape (size[0], size[1], 3)
    """
    size = tuple(size)
    split = int(split)

    # Base spin directions (normalized)
    base_spin_cases = [
        [    1.0,     0.0, 0.0],  # +x
        [   -1.0,     0.0, 0.0],  # -x
        [    0.0,     1.0, 0.0],  # +y
        [    0.0,    -1.0, 0.0],  # -y
        [ 0.7071,  0.7071, 0.0],  # +x+y
        [ 0.7071, -0.7071, 0.0],  # +x-y
        [-0.7071,  0.7071, 0.0],  # -x+y
        [-0.7071, -0.7071, 0.0]   # -x-y
    ]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    xsplit = size[0] / split  # x length of each split area
    ysplit = size[1] / split  # y length of each split area
    spin = np.empty(size + (3,))

    # Store the base directions before perturbation
    base_directions = []
    for nx in range(split):
        for ny in range(split):
            selected_case = random.choice(base_spin_cases)
            base_directions.append((nx, ny, selected_case))
    
    # Apply perturbation if requested
    if add_perturbation:
        # Set up perturbation seed (use seed+1 if not provided)
        if perturbation_seed is None and seed is not None:
            perturbation_seed = seed + 1
        
        if perturbation_seed is not None:
            random.seed(perturbation_seed)
            np.random.seed(perturbation_seed)
        
        # 5 degrees in radians
        angle = np.deg2rad(deg)
        
        for nx, ny, base_dir in base_directions:
            # Randomly choose clockwise or counter-clockwise
            direction = random.choice([-1, 1])
            actual_angle = angle * direction
            
            # Rotate the vector in xy plane
            cos_a, sin_a = np.cos(actual_angle), np.sin(actual_angle)
            x, y, z = base_dir
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            perturbed_case = [new_x, new_y, z]
            
            # Determine bounds
            xlow_bound = int(nx * xsplit)
            xhigh_bound = int((nx+1) * xsplit) if nx + 1 < split else size[0]
            ylow_bound = int(ny * ysplit)
            yhigh_bound = int((ny+1) * ysplit) if ny + 1 < split else size[1]
            
            spin[xlow_bound:xhigh_bound, ylow_bound:yhigh_bound, :] = perturbed_case
    else:
        # No perturbation - just use base directions
        for nx, ny, base_dir in base_directions:
            xlow_bound = int(nx * xsplit)
            xhigh_bound = int((nx+1) * xsplit) if nx + 1 < split else size[0]
            ylow_bound = int(ny * ysplit)
            yhigh_bound = int((ny+1) * ysplit) if ny + 1 < split else size[1]
            
            spin[xlow_bound:xhigh_bound, ylow_bound:yhigh_bound, :] = base_dir

    return spin
