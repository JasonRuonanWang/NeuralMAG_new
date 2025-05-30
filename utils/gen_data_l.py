# -*- coding: utf-8 -*-
import random
import time, os
import numpy as np
from tqdm import tqdm
import argparse
import torch

from libs.misc import Culist, initial_spin_prepare, create_random_mask, error_plot
import libs.MAG2305 as MAG2305

def prepare_model(args):
    film = MAG2305.mmModel(types='bulk', size=(args.w, args.w, args.layers), 
                           cell=(3,3,3), Ms=args.Ms, Ax=args.Ax, Ku=args.Ku, 
                           Kvec=args.Kvec, device="cuda:" + str(args.gpu))
    print('Creating {} layer models \n'.format(args.layers))

    # Initialize demag matrix
    time_start = time.time()
    film.DemagInit()
    time_finish = time.time()
    print('Time cost: {:f} s for initializing demag matrix \n'.format(time_finish-time_start))
    return film

def generate_data(args, film):
    Hext_val = np.random.randn(3) * args.Hext_val
    Hext = Hext_val * args.Hext_vec

    for seed in tqdm(range(0, args.nseeds)):
        path_format = './Dataset/data_Hd{}_Hext{}_mask/seed{}' if args.mask=='True' else './Dataset/data_Hd{}_Hext{}/seed{}'
        save_path = path_format.format(args.w, int(args.Hext_val), seed)
        os.makedirs(save_path, exist_ok=True)

        spin = initial_spin_prepare(args.w, args.layers, seed)
        if args.mask == 'True':
            mask = create_random_mask((args.w, args.w), np.random.randint(2, args.w), random.choice([True, False]))
            spin = film.SpinInit(spin * mask)
        else:
            spin = film.SpinInit(spin)

        error_list = simulate_spins(film, spin, Hext, args, save_path)
        save_simulation_data(args, save_path, error_list, Hext)

def simulate_spins(film, spin, Hext, args, save_path):
    error_list = []
    itern = 0
    error_ini = 1

    Spininit = np.reshape(spin[:,:,:,:], (args.w, args.w, args.layers*3))
    np.save(os.path.join(save_path, f'Spins_0.npy'), Spininit)
    pbar = tqdm(total=args.max_iter)
    while error_ini > args.error_min and itern < args.max_iter:
        error = film.SpinLLG_RK4(Hext=Hext, dtime=args.dtime, damping=args.damping)
        error_ini = error
        error_list.append(error)
        itern += 1
        np.save(os.path.join(save_path, f'Spins_{itern}.npy'), np.reshape(film.Spin.cpu(), (args.w, args.w, args.layers*3)))
        np.save(os.path.join(save_path, f'Hds_{itern}.npy'), np.reshape(film.Hd.cpu(), (args.w, args.w, args.layers*3)))
        pbar.update(1)
        pbar.set_description(f"num_seeds: {itern}/{args.sav_samples}, error: {error}")
    pbar.close()
    return error_list

def save_simulation_data(args, save_path, error_list, Hext):
    # 获取所有保存的文件
    spin_files = sorted([f for f in os.listdir(save_path) if f.startswith('Spins_') and f.endswith('.npy')])
    hd_files = sorted([f for f in os.listdir(save_path) if f.startswith('Hds_') and f.endswith('.npy')])

    # 随机选取所需的样本
    random_indices = sorted(random.sample(range(1, len(hd_files)-1), args.sav_samples))
    Spins_random_list = [np.load(os.path.join(save_path, spin_files[i])) for i in random_indices]
    Hds_random_list = [np.load(os.path.join(save_path, hd_files[i])) for i in random_indices]

    Spins_random_list.append(np.load(os.path.join(save_path, spin_files[-1])))
    Hds_random_list.append(np.load(os.path.join(save_path, hd_files[-1])))

    # 删除所有的文件
    for f in spin_files:
        os.remove(os.path.join(save_path, f))
    for f in hd_files:
        os.remove(os.path.join(save_path, f))
    # 保存数据
    np.save(os.path.join(save_path, 'Spins.npy'), np.stack(Spins_random_list, axis=0))
    np.save(os.path.join(save_path, 'Hds.npy'), np.stack(Hds_random_list, axis=0))

    error_plot(error_list, os.path.join(save_path, 'iterns{:.1e}_errors_{:.1e}'.format(len(error_list), error_list[-1])),
               str('[{:.2f}, {:.2f}, {:.2f}]'.format(Hext[0], Hext[1], Hext[2])))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--gpu',        type=int,    default=0,         help='GPU ID (default: 0)')
    parser.add_argument('--w',          type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--layers',     type=int,    default=2,         help='MAG model layers (default: 1)')
    
    parser.add_argument('--Ms',         type=float,  default=1000,      help='MAG model Ms (default: 1000)')
    parser.add_argument('--Ax',         type=float,  default=0.5e-6,    help='MAG model Ax (default: 0.5e-6)')
    parser.add_argument('--Ku',         type=float,  default=0.0,       help='MAG model Ku (default: 0.0)')
    parser.add_argument('--Kvec',       type=Culist, default=(0,0,1),   help='MAG model Kvec (default: (0,0,1))')
    parser.add_argument('--damping',    type=float,  default=0.1,       help='MAG model damping (default: 0.1)')
    parser.add_argument('--Hext_val',   type=float,  default=0,         help='external field value (default: 0.0)')
    parser.add_argument('--Hext_vec',   type=Culist, default=(1,1,0),   help='external field vector (default:(1,1,0))')

    parser.add_argument('--dtime',      type=float,  default=1.0e-13,   help='real time step (default: 1.0e-13)')
    parser.add_argument('--error_min',  type=float,  default=1.0e-6,    help='min error (default: 1.0e-6)')
    parser.add_argument('--max_iter',   type=int,    default=50000,     help='max iteration number (default: 50000)')
    parser.add_argument('--sav_samples',type=int,    default=500,       help='save samples (default: 500)')
    parser.add_argument('--mask',       type=str,    default='False',   help='mask (default: False)')
    parser.add_argument('--nseeds',     type=int,    default=100,       help='number of seeds (default: 100)')
    args = parser.parse_args() 

    device = torch.device("cuda:{}".format(args.gpu))
    
    #Prepare MAG model: film
    film = prepare_model(args)

    #Generate spin and Hd pairs data
    generate_data(args, film)
