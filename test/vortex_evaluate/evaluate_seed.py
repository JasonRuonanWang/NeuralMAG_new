# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:00:00 2023

#########################################
#                                       #
#  Get vortex counting from spin data   #
#                                       #
#########################################

"""
import os
import glob
import argparse
import shutil
import numpy as np

import csv
from collections import defaultdict, OrderedDict

from util.plot import *
from util.vortex_utils import *

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet speed test method: LLG_RK4')    
    parser.add_argument('--w',           type=int,    default=32,        help='MAG model size (default: 32)')
    parser.add_argument('--experiment',  type=str,    default="square",  help='experiment type (square, random_material, random_shape) (default: square)')
    parser.add_argument('--layers',      type=int,    default=2,         help='MAG model layers (default: 1)')
    parser.add_argument('--split',       type=int,    default=4,        help='MAG model split (default: 4)')
    parser.add_argument('--deg',         type=float,  default=0.1,       help='MAG model deg (default: 0.1)')
    parser.add_argument('--errorfilter', type=float,  default=1e-5,      help='MAG model errorlimit (default: 0.1)')
    args = parser.parse_args()

    #dir 
    if args.experiment == 'square':
        ex_type = 'sq'
    elif args.experiment == 'random_material':
        ex_type = 'mt'
    elif args.experiment == 'random_shape':
        ex_type = 'sp'
    path0 = "./seed/size{}_{}/".format(args.w, ex_type)
    path_save="./seed/size{}_{}/deg{}/".format(args.w, ex_type, args.deg)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    rand_seed_list = []
    for subpath in os.listdir(path0):
        if r"split" + str(args.split) + r"_deg" + str(args.deg) in subpath:
            rand_seed = subpath.split('d')[2].split('_')[0]
            rand_seed_list.append(rand_seed)
    rand_seed_list = list(set(rand_seed_list))
    rand_seed_list.sort()
    stable_seed_list = []
    max_same_sum = 0
    for rand_seed in rand_seed_list:
        all_results = []  
        baseline_result = None 
        files = [f for f in os.listdir(path0) if r"split" + str(args.split) + r"_deg" + str(args.deg) in f and str(rand_seed) in f]
        if len(files) == 0:
            print("No files found for rand_seed: {}".format(rand_seed))
            continue
        files.sort(key=lambda x: int(x.split('_')[2].split('d')[1]))
        for i,subpath in enumerate(files):
            if r"split" + str(args.split) in subpath and rand_seed in subpath:
            
                files_dir = glob.glob(os.path.join(path0 + subpath, '*'))
                matching_files1 = [f for f in files_dir if 'Spin_fft_converge' in f]
                if len(matching_files1) == 0:
                    print('\n',subpath, 'not complete')
                    continue

                spin_2305 = np.load(path0 + subpath + '/Spin_fft_converge.npy')

                bmodel = np.sum(np.abs(spin_2305)[:, :, :], axis=2)
                model = np.zeros((bmodel.shape[0], bmodel.shape[1]))
                model[np.where(bmodel > 0)] = 1
        
                (vortex_2305, antivtx_2305, 
                pp_vortex_2305, pp_antivtx_2305,
                np_vortex_2305, np_antivtx_2305,
                cw_vortex_2305, ccw_vortex_2305) = analyze_winding(spin_2305, model)
                
                result = {
                    'index': i,
                    'rand_seed': rand_seed,
                    'vortex_2305': vortex_2305,
                    'antivtx_2305': antivtx_2305,
                    'pp_vortex_2305': pp_vortex_2305,
                    'pp_antivtx_2305': pp_antivtx_2305,
                    'np_vortex_2305': np_vortex_2305,
                    'np_antivtx_2305': np_antivtx_2305,
                    'cw_vortex_2305': cw_vortex_2305,
                    'ccw_vortex_2305': ccw_vortex_2305
                }

                all_results.append(result)
                if str(rand_seed) + r'_0' in subpath:
                    baseline_result = result
        if len(all_results) < 10:
            print("Not enough results for rand_seed: {}".format(rand_seed))
            continue

        exclude_columns = {'index'}

        result_counts = defaultdict(int)
        for res in all_results:
            key = tuple(v for k, v in res.items() if k not in exclude_columns)
            result_counts[key] += 1
        
        baseline_key = tuple(v for k, v in baseline_result.items() if k not in exclude_columns)
        
        total_unique = len(result_counts)
        same_as_baseline = result_counts.get(baseline_key, 0)
        max_same_count = max(result_counts.values())  
        max_same_sum += total_unique
        if total_unique == 1:
            stable_seed_list.append(int(rand_seed))
        save_to_csv(all_results, path_save + "results_{}.csv".format(rand_seed))
        content = ''' rand_seed: {} unique_results: {}, same_as_baseline: {} max_same_count: {}''' \
            .format(rand_seed, total_unique, same_as_baseline, max_same_count)

        with open(path_save+"summary_size{}.txt".format(args.w), "a") as file:
            file.write(content)
            file.write("\n")

        print(content)
    print("Stable rand_seed: ", stable_seed_list)
    print("Stable rand_seed count: ", len(stable_seed_list))
    print("Average stable rate: ", len(stable_seed_list)/len(rand_seed_list))
    print("Average unique results per seed: ", max_same_sum/len(rand_seed_list))
    np.save(path_save + "stable_seed.npy", np.array(stable_seed_list))
