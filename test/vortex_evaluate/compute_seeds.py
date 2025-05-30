# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:09:00 2024
"""

import torch
import numpy as np
import time, os
from tqdm import tqdm
import argparse
import random
import glob
import csv
from collections import defaultdict, OrderedDict

from util.plot import *
from util.vortex_utils import *

from libs.misc import Culist, create_model, create_random_mask
import libs.MAG2305 as MAG2305
def save_to_csv(results_list, file):
    fieldnames = OrderedDict([
        ('index', 'Index'),
        ('rand_seed', 'Random Seed'),
        ('vortex', 'Vortex cores'),
        ('antivtx', 'Antivtx cores'),
        ('pp_vortex', 'Positive vortex'),
        ('pp_antivtx', 'Positive antivtx'),
        ('np_vortex', 'Negative vortex'),
        ('np_antivtx', 'Negative antivtx'),
        ('cw_vortex', 'Clockwise vortex'),
        ('ccw_vortex', 'CounterCW vortex')
    ])
    
    with open(file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames.keys())
        writer.writerow({k: v for k, v in fieldnames.items()})
        
        for result in results_list:
            row = {
                'index': result['index'],
                'rand_seed': result['rand_seed'],
                'vortex': result['vortex_2305'],
                'antivtx': result['antivtx_2305'],
                'pp_vortex': result['pp_vortex_2305'],
                'pp_antivtx': result['pp_antivtx_2305'],
                'np_vortex': result['np_vortex_2305'],
                'np_antivtx': result['np_antivtx_2305'],
                'cw_vortex': result['cw_vortex_2305'],
                'ccw_vortex': result['ccw_vortex_2305']
            }
            writer.writerow(row)

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

def initialize_models(args):
    # Model shape and save model
    test_model = create_model(args.w, args.modelshape)
    if args.experiment == 'square':
        ex_type = 'sq'
    elif args.experiment == 'random_material':
        ex_type = 'mt'
    elif args.experiment == 'random_shape':
        ex_type = 'sp'
    path0 = "./seed/size{}_{}/".format(args.w, ex_type)
    os.makedirs(path0, exist_ok=True)
    np.save(path0 + 'model', test_model[:,:,0])

    #Initialize MAG2305 models.
    model0 = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), cell=(3,3,3), 
                             Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, Kvec=args.Kvec, 
                             device="cuda:" + str(args.gpu)
                             )
    print('Creating {} layer model \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    model0.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for getting demag matrix \n'.format(time_finish-time_start))

    return model0, test_model, path0


def update_spin_state(film0, Hext, args, test_model, path):
    # Do iteration
    rcd_dspin_fft = np.array([[],[]])
    rcd_windabs_fft = np.array([[],[]])
    rcd_windsum_fft = np.array([[],[]])
    fig, ax1, ax2, ax3 = plot_prepare()
    
    nplot = args.nplot
    for iters in range(args.max_iter):
        if iters == 0:
            spin_ini = np.array(film0.Spin[:,:,0].cpu())

        #MAG calculate for spin iteration
        error_fft = film0.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=0.1)
        spin_fft = film0.Spin.cpu().numpy()

        if iters % args.nsave ==0 or error_fft <=args.error_min:
            rcd_dspin_fft = np.append(rcd_dspin_fft, [[iters], [error_fft]], axis=1)

            wind_dens_fft, wind_abs_fft, wind_sum_fft = get_winding(spin_fft[:,:,0],
                                                                    test_model[:,:,0])
            rcd_windabs_fft = np.append(rcd_windabs_fft, 
                                        [[iters], [wind_abs_fft]], axis=1)
            rcd_windsum_fft = np.append(rcd_windsum_fft, 
                                        [[iters], [wind_sum_fft]], axis=1)

        if iters % nplot ==0 or error_fft <=args.error_min:
            plot_spin( spin_fft[:,:,0], ax1, 'fft - iters{}'.format(iters))
            plot_wind( wind_dens_fft, ax2, 'fft-vortices wd[{}]/[{}]'.format(round(wind_abs_fft), round(wind_sum_fft)))
            plot_error( rcd_dspin_fft, ax3)
            plot_save(path, "spin_iters{}".format(iters))
    
        if error_fft <=args.error_min or iters==args.max_iter-1:
            spin_end_fft = np.array(film0.Spin[:,:,0].cpu())
            plot_close()
            break

    return (rcd_dspin_fft, rcd_windabs_fft, rcd_windsum_fft,
            spin_ini, spin_end_fft)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--experiment',  type=str,    default="square",  help='experiment type (square, random_material, random_shape) (default: square)')

    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 2)')
    parser.add_argument('--split',      type=int,    default=4,         help='MAG model split (default: 4)')
    parser.add_argument('--modelshape', type=str,    default='square',  help='MAG model shape: square, circle, triangle')
    parser.add_argument('--deg',        type=float,  default=0.1,         help='degree of perturbation (default: 0.1)')

    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,0,0),   help='external field vector (default:(1,0,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-5,    help='min error (default: 1.0e-5)')
    parser.add_argument('--max_iter',   type=int,    default=100000,    help='max iteration number (default: 100000)')
    parser.add_argument('--nsave',      type=int,    default=10,        help='save number (default: 10)')
    parser.add_argument('--nplot',      type=int,    default=2000,      help='plot number (default: 2000)')
    parser.add_argument('--nsamples',   type=int,    default=100,       help='sample number (default: 100)')
    parser.add_argument('--test_num',   type=int,    default=10,        help='test number (default: 10)')

    args = parser.parse_args()
    
    # create two film models
    film0, test_model, path0 = initialize_models(args)

    # Random seed list
    seeds_list = list(range(10000, 110000, 100))
    stable_seeds_list = []
    pbar = tqdm(total=args.nsamples * 10)
    for i in range(args.nsamples):
        rand_seed = seeds_list[i]        
        np.random.seed(rand_seed)
        if args.experiment == 'random_material':
            args.Ms = 400 + np.random.rand()*800
            args.Ax = 3e-7 + np.random.rand()*4e-7
        if args.experiment == 'random_shape':
            mask = create_random_mask((args.w,args.w),30,fixshape=True,seed=rand_seed)
            mask = mask[:,:,:,0]
        else:
            mask = np.ones((args.w, args.w, 1), dtype=np.float32)
            
        for j in range(args.test_num):
            add_perturbation = True
            if j == 0:
                add_perturbation = False

            spin0 = get_randspin_2D(size=(args.w, args.w, args.layers), split=args.split, deg=args.deg, 
                                    add_perturbation=add_perturbation, seed=rand_seed, perturbation_seed=rand_seed+j)
            film0.SpinInit(spin0 * mask)
            print('initializing spin state \n')
            
            # External field
            Hext = args.Hext_val * np.array(args.Hext_vec)

            # Create directory
            path = path0+"split{}_deg{}_rand{}_{}/".format(args.split, args.deg, rand_seed, j)
            os.makedirs(path, exist_ok=True)
            
            #check any bad cases
            if os.path.exists(os.path.join(path, f'Spin_fft_converge.npy')):
                print('exits and skip: ',path)
                pbar.update(1)
                continue # skip this case
            print('do ',path)

            ###########################
            # Spin update calculation #
            ###########################
            print('Begin spin updating:\n')
            (rcd_dspin_fft, rcd_windabs_fft, rcd_windsum_fft,
            spin_ini, spin_end_fft
            ) = update_spin_state(film0, Hext, args, test_model, path)
            
            
            ###################
            # Data processing #
            ###################
            
            np.save(path+f'Dspin_fft_max', rcd_dspin_fft)
            np.save(path+f'Wind_fft_abs', rcd_windabs_fft)
            np.save(path+f'Wind_fft_sum', rcd_windsum_fft)
            
            np.save(path+f'Spin_initial',  spin_ini)
            np.save(path+f'Spin_fft_converge', spin_end_fft)

            pbar.update(1)
    
    pbar.close()
