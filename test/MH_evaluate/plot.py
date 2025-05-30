import os
import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(4, 4, figsize=(10, 10))
size_list = [64, 128, 256, 512]
colors = ['blue', 'green', 'orange', 'purple']
line_styles = ['--', '-']

# 1
shape_list = ["False","True","triangle", "hole"]
shape_title_list = ["Square", "Convex hull", "Triangle", "Hole"]
for i,shape in enumerate(shape_list):
    for j, size in enumerate(size_list):
        file_path = f"./figs_k48/shape_{shape}/size{size}_Ms1000.0_Ax5e-07_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[0, i].plot(x_plot, y1_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[0], alpha=0.6)
        axs[0, i].plot(x_plot, y2_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[1], alpha=0.6)

    axs[0, i].set_title(f'{shape_title_list[i]}',fontsize=16)
    axs[0, i].set_xlabel(r"$H_{ext}$ [Oe]")
    axs[0, i].set_ylabel(r"$M_{ext}$ / $M_s$")
    axs[0, i].set_xlim(min(x_plot)*1.1, max(x_plot)*1.1)
    axs[0, i].set_ylim(-1.1, 1.1)
    axs[0, i].grid(True, axis='both', lw=0.5, ls='-.')

    line_a, = axs[0, i].plot([], [], lw=1.5, color='black', linestyle='-', label='Unet')
    line_b, = axs[0, i].plot([], [], lw=1.5, color='black', linestyle='--', label='FFT')
    axs[0, i].legend(handles=[line_a, line_b], fontsize=5)

    color_patches = [plt.Line2D([0], [0], color=color, lw=4, label=f'size {size}') for color, size in zip(colors, size_list)]
    axs[0, i].legend(handles=color_patches, fontsize=5)

#2
Ms_list = [800, 1200]
for i,size in enumerate(size_list):
    for j, Ms in enumerate(Ms_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms{Ms}.0_Ax5e-07_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[1, i].plot(x_plot, y1_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[0], alpha=0.6)
        axs[1, i].plot(x_plot, y2_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[1], alpha=0.6)

    axs[1, i].set_title(f'MH vs Ms: size {size}',fontsize=16)
    axs[1, i].set_xlabel(r"$H_{ext}$ [Oe]")
    axs[1, i].set_ylabel(r"$M_{ext}$ / $M_s$")
    axs[1, i].set_xlim(min(x_plot)*1.1, max(x_plot)*1.1)
    axs[1, i].set_ylim(-1.1, 1.1)
    axs[1, i].grid(True, axis='both', lw=0.5, ls='-.')

    line_a, = axs[1, i].plot([], [], lw=1.5, color='black', linestyle='-', label='Unet')
    line_b, = axs[1, i].plot([], [], lw=1.5, color='black', linestyle='--', label='FFT')
    axs[1, i].legend(handles=[line_a, line_b], fontsize=5)

    color_patches = [plt.Line2D([0], [0], color=color, lw=4, label=f'Ms {Ms}') for color, Ms in zip(colors, Ms_list)]
    axs[1, i].legend(handles=color_patches, fontsize=5)

#3
Ax_list = [3e-07, 7e-07]
for i,size in enumerate(size_list):
    for j, Ax in enumerate(Ax_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms1000.0_Ax{Ax}_Ku0.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[2, i].plot(x_plot, y1_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[0], alpha=0.6)
        axs[2, i].plot(x_plot, y2_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[1], alpha=0.6)

    axs[2, i].set_title(f'MH vs Ax: size {size}',fontsize=16)
    axs[2, i].set_xlabel(r"$H_{ext}$ [Oe]")
    axs[2, i].set_ylabel(r"$M_{ext}$ / $M_s$")
    axs[2, i].set_xlim(min(x_plot)*1.1, max(x_plot)*1.1)
    axs[2, i].set_ylim(-1.1, 1.1)
    axs[2, i].grid(True, axis='both', lw=0.5, ls='-.')

    line_a, = axs[2, i].plot([], [], lw=1.5, color='black', linestyle='-', label='Unet')
    line_b, = axs[2, i].plot([], [], lw=1.5, color='black', linestyle='--', label='FFT')
    axs[2, i].legend(handles=[line_a, line_b], fontsize=5)

    color_patches = [plt.Line2D([0], [0], color=color, lw=4, label=f'Ax {Ax}') for color, Ax in zip(colors, Ax_list)]
    axs[2, i].legend(handles=color_patches, fontsize=5)

#4
Ku_list = [100000, 200000, 300000, 400000]
Ku_e_list = [1e5, 2e5, 3e5, 4e5]
for i,size in enumerate(size_list):
    for j, Ku in enumerate(Ku_list):
        file_path = f"./figs_k48/shape_False/size{size}_Ms1000.0_Ax5e-07_Ku{Ku}.0_dtime2e-13_split8_seed1234_Layers2"
        x_plot = np.load(file_path + '/Hext_array.npy')
        y1_plot = np.load(file_path + '/Mext_array_mm.npy')
        y2_plot = np.load(file_path + '/Mext_array_un.npy')
        axs[3, i].plot(x_plot, y1_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[0], alpha=0.6)
        axs[3, i].plot(x_plot, y2_plot, lw=1.5, marker='o', markersize=0, color=colors[j], linestyle=line_styles[1], alpha=0.6)

    axs[3, i].set_title(f'MH vs Ku: size {size}',fontsize=16)
    axs[3, i].set_xlabel(r"$H_{ext}$ [Oe]")
    axs[3, i].set_ylabel(r"$M_{ext}$ / $M_s$")
    axs[3, i].set_xlim(min(x_plot)*1.1, max(x_plot)*1.1)
    axs[3, i].set_ylim(-1.1, 1.1)
    axs[3, i].grid(True, axis='both', lw=0.5, ls='-.')

    line_a, = axs[3, i].plot([], [], lw=1.5, color='black', linestyle='-', label='Unet')
    line_b, = axs[3, i].plot([], [], lw=1.5, color='black', linestyle='--', label='FFT')
    axs[3, i].legend(handles=[line_a, line_b], fontsize=5)

    color_patches = [plt.Line2D([0], [0], color=color, lw=4, label=f'Ku {Ku}') for color, Ku in zip(colors, Ku_e_list)]
    axs[3, i].legend(handles=color_patches, fontsize=5)

plt.tight_layout()
plt.show()
plt.savefig('./figs_k48/MH_vs_parameters.svg', dpi=300)
plt.close()

