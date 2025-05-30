import os
import numpy as np
import matplotlib.pyplot as plt
plt_path = "./plot"
if not os.path.exists(plt_path):
    os.makedirs(plt_path)
size_list = [64, 128, 256, 512]
def find_Hext(x_plot, y_plot):
    y_p = min(y_plot[y_plot>0])
    y_n = max(y_plot[y_plot<0])
    y_p_index = np.where(y_plot == y_p)
    y_n_index = np.where(y_plot == y_n)

    Hext_p = x_plot[y_p_index]
    Hext_n = x_plot[y_n_index]
    Hext_0 = Hext_p*y_n/(y_n-y_p) - Hext_n*y_p/(y_n-y_p)
    return Hext_0
fig, axs = plt.subplots(8, 3, figsize=(30, 30))
# 1
shape_list = ["False","True","triangle", "hole"]
shape_title_list = ["Square", "Convex hull", "Triangle", "Hole"]
for i,shape in enumerate(shape_list):
    for j, size in enumerate(size_list):
        file_path = f"./figs_k48/shape_{shape}/size{size}_Ms1000.0_Ax5e-07_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[2*i,0].plot(size, y1_plot[100], lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i,0].plot(size, y2_plot[100], lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i,0].set_ylabel(r"$M_r$ / $M_s$")
        axs[2*i,0].set_ylim(0, 1)
        Hext_1 = find_Hext(x_plot, y1_plot)
        Hext_2 = find_Hext(x_plot, y2_plot)
        axs[2*i+1,0].plot(size, -Hext_1, lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i+1,0].plot(size, -Hext_2, lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i+1,0].set_xlabel("Size")
        axs[2*i+1,0].set_ylabel(r"$H_{ext}$ [Oe]")
        axs[2*i+1,0].set_ylim(0, 800)
    axs[2*i,0].set_title(f'{shape_title_list[i]}',fontsize=20)


#2
Ms_list = [800, 1200]
for i,Ms in enumerate(Ms_list):
    for j, size in enumerate(size_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms{Ms}.0_Ax5e-07_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[2*i,1].plot(size, y1_plot[100], lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i,1].plot(size, y2_plot[100], lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i,1].set_ylabel(r"$M_r$ / $M_s$")
        axs[2*i,1].set_ylim(0, 1)
        Hext_1 = find_Hext(x_plot, y1_plot)
        Hext_2 = find_Hext(x_plot, y2_plot)
        axs[2*i+1,1].plot(size, -Hext_1, lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i+1,1].plot(size, -Hext_2, lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i+1,1].set_xlabel("size")
        axs[2*i+1,1].set_ylabel(r"$H_{ext}$ [Oe]")
        axs[2*i+1,1].set_ylim(0, 800)
    axs[2*i,1].set_title(f'Ms= {Ms} emu/cc',fontsize=20)

#3
Ax_list = [3e-07, 7e-07]
for i,Ax in enumerate(Ax_list):
    for j, size in enumerate(size_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms1000.0_Ax{Ax}_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[2*i+4,1].plot(size, y1_plot[100], lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i+4,1].plot(size, y2_plot[100], lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i+4,1].set_ylabel(r"$M_r$ / $M_s$")
        axs[2*i+4,1].set_ylim(0, 1)
        Hext_1 = find_Hext(x_plot, y1_plot)
        Hext_2 = find_Hext(x_plot, y2_plot)
        axs[2*i+5,1].plot(size, -Hext_1, lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i+5,1].plot(size, -Hext_2, lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i+5,1].set_xlabel("size")
        axs[2*i+5,1].set_ylabel(r"$H_{ext}$ [Oe]")
        axs[2*i+5,1].set_ylim(0, 800)
    axs[2*i+4,1].set_title(f'Ax= {Ax} erg/cm',fontsize=20)


#4
Ku_list = [100000, 200000, 300000, 400000]
Ku_e_list = [1e5, 2e5, 3e5, 4e5]
for i, Ku in enumerate(Ku_list):
    for j, size in enumerate(size_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms1000.0_Ax5e-07_Ku{Ku}.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[2*i,2].plot(size, y1_plot[100], lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i,2].plot(size, y2_plot[100], lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i,2].set_ylabel(r"$M_r$ / $M_s$")
        axs[2*i,2].set_ylim(0, 1)
        Hext_1 = find_Hext(x_plot, y1_plot)
        Hext_2 = find_Hext(x_plot, y2_plot)
        axs[2*i+1,2].plot(size, -Hext_1, lw=1.5, marker='o', markersize=10, alpha=0.6, color='blue')
        axs[2*i+1,2].plot(size, -Hext_2, lw=1.5, marker='*', markersize=10, alpha=0.6, color='red')
        axs[2*i+1,2].set_xlabel("size")
        axs[2*i+1,2].set_ylabel(r"$H_{ext}$ [Oe]")  
        axs[2*i+1,2].set_ylim(0, 800)
    axs[2*i,2].set_title(f'Ku= {Ku} erg/cc',fontsize=20)
plt.tight_layout()
plt.savefig(f'./plot/M_r.svg', dpi=300)
plt.close()

