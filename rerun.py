from uedge import *
from uedge.hdf5 import *
from uedge.rundt import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as an
import uedge_mvu.tstep as ut
import UEDGE_utils.analysis as ana
import UEDGE_utils.plot as utplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from runcase import *

setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=1.0, kyi=1.0, difni=0.5,nonuniform = True)
setBoundaryConditions(ncore=6.2e19, pcoree=2.0e6, pcorei=2.0e6, recycp=0.98)
setimpmodel(impmodel=True)
#setgaspuff()

bbb.cion=3
bbb.oldseec=0
bbb.restart=1
bbb.nusp_imp = 3
bbb.icntnunk=0
hdf5_restore("./final.hdf5") # converged solution with Li atoms and ions for bbb.isteon=0
bbb.issfon=0; bbb.ftol = 1e20
bbb.exmain()

mu.paws("Completed reading a converged solution for Li atoms and Li ions solution")
Case_t0 = ana.get_surface_heatflux_components()


current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "fngxrb_use")

# Find CSV files matching the new pattern
csv_files = sorted([
    f for f in os.listdir(data_dir)
    if f.startswith("fngxrb_use_") and f.endswith(".csv")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

if csv_files:
    last_csv = csv_files[-1]
    csv_path = os.path.join(data_dir, last_csv)
    bbb.fngxrb_use[:, 1, 0] = np.loadtxt(csv_path, delimiter=',')
    hdf5_restore("./final_iteration.hdf5") 
    bbb.issfon = 0
    bbb.ftol = 1e20
    bbb.exmain()
    Case_t5 = ana.get_surface_heatflux_components()
else:
    print("No matching CSV files found in the directory.")


y_pos = com.yyrb.reshape(-1)[1:-1]


import matplotlib.pyplot as plt
import numpy as np

def plot_li_surface_heatflux_2x2_custom(Case_t0, Case_t5, y_pos, 
                                        label0='Case t0', label5='Case t5', 
                                        target='t0_vs_t5', log=False, 
                                        axis_label_fontsize=14, legend_fontsize=11, labelsize_font=12):

    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey='row')

    # --- Top row: Surface Heat Flux Components ---
    # Left: Case_t0
    axs[0, 0].plot(y_pos, Case_t0['conv_cond'], label='Conv. + Cond.', linestyle='-', color='blue', linewidth=1.5)
    axs[0, 0].plot(y_pos, Case_t0['q_D_MW'], label='Surface recomb.', linestyle='--', color='green', linewidth=1.5)
    axs[0, 0].plot(y_pos, Case_t0['ion_ke'], label='Ion KE', linestyle='-.', color='black', linewidth=1.5)
    axs[0, 0].plot(y_pos, Case_t0['h_photons'], label='H photons', linestyle=':', color='purple', linewidth=1.5)
    axs[0, 0].plot(y_pos, Case_t0['imp_photons'], label='Imp. photons', linestyle='-', color='red', linewidth=1.5)
    axs[0, 0].plot(y_pos, Case_t0['total_flux'], 'm-', lw=2, label='Total')
    axs[0, 0].set_title(label0, fontsize=axis_label_fontsize)
    #axs[0, 0].legend(fontsize=legend_fontsize, ncol=2)
    axs[0, 0].set_ylabel(r'$q_{\perp}$ [MW/m$^2$]', fontsize=axis_label_fontsize)
    axs[0, 0].grid(True, which='both', linestyle=':', color='gray')
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].text(0.02, 0.95, '(a)', transform=axs[0, 0].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')

    # Right: Case_t5 (no legend)
    axs[0, 1].plot(y_pos, Case_t5['conv_cond'], label='Conv. + Cond.', linestyle='-', color='blue', linewidth=1.5)
    axs[0, 1].plot(y_pos, Case_t5['q_D_MW'], label='Surface recomb.', linestyle='--', color='green', linewidth=1.5)
    axs[0, 1].plot(y_pos, Case_t5['ion_ke'], label='Ion KE', linestyle='-.', color='black', linewidth=1.5)
    axs[0, 1].plot(y_pos, Case_t5['h_photons'], label='H photons', linestyle=':', color='purple', linewidth=1.5)
    axs[0, 1].plot(y_pos, Case_t5['imp_photons'],  label='Imp. photons', linestyle='-', color='red', linewidth=1.5)
    axs[0, 1].plot(y_pos, Case_t5['total_flux'], 'm-', lw=2, label='Total')
    axs[0, 1].set_title(label5, fontsize=axis_label_fontsize)
    axs[0, 1].grid(True, which='both', linestyle=':', color='gray')
    axs[0, 1].legend(fontsize=legend_fontsize, ncol=2)
    axs[0, 1].set_axisbelow(True)
    axs[0, 1].text(0.02, 0.95, '(b)', transform=axs[0, 1].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')

    # --- Bottom row: Surface Recombination Breakdown ---
    # Left: Case_t0
    axs[1, 0].plot(y_pos, Case_t0['q_D_MW'], label=r'D$^+$ $\rightarrow$ D', color='blue', linestyle='-')
    axs[1, 0].plot(y_pos, Case_t0['q_Li1_MW'], label=r'Li$^+$ $\rightarrow$ Li', color='green', linestyle='--')
    axs[1, 0].plot(y_pos, Case_t0['q_Li2_MW'], label=r'Li$^{2+}$ $\rightarrow$ Li', color='orange', linestyle='-.')
    axs[1, 0].plot(y_pos, Case_t0['q_Li3_MW'], label=r'Li$^{3+}$ $\rightarrow$ Li', color='red', linestyle=':')
    axs[1, 0].legend(fontsize=legend_fontsize)
    axs[1, 0].set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize=axis_label_fontsize)
    axs[1, 0].set_ylabel(r'q$_{recomb.}$ [MW/m$^2$]', fontsize=axis_label_fontsize)
    axs[1, 0].grid(True, which='both', linestyle=':', color='gray')
    axs[1, 0].set_axisbelow(True)
    axs[1, 0].text(0.02, 0.95, '(c)', transform=axs[1, 0].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')

    # Right: Case_t5 (no legend)
    axs[1, 1].plot(y_pos, Case_t5['q_D_MW'], color='green', linestyle='-')
    axs[1, 1].plot(y_pos, Case_t5['q_Li1_MW'], color='blue', linestyle='--')
    axs[1, 1].plot(y_pos, Case_t5['q_Li2_MW'], color='orange', linestyle='-.')
    axs[1, 1].plot(y_pos, Case_t5['q_Li3_MW'], color='red', linestyle=':')
    axs[1, 1].set_xlabel(r'r$_{div}$ - r$_{sep}$ (m)', fontsize=axis_label_fontsize)
    axs[1, 1].grid(True, which='both', linestyle=':', color='gray')
    axs[1, 1].set_axisbelow(True)
    axs[1, 1].text(0.02, 0.95, '(d)', transform=axs[1, 1].transAxes, fontsize=14, va='top', ha='left', fontweight='bold')

    # Set log/linear scale
    if log:
        for i in range(2):
            axs[0, i].set_yscale('log')
            axs[1, i].set_yscale('log')
        axs[0, 0].set_ylim([1e-3, 10])
        axs[1, 0].set_ylim([1e-5, 2])
    else:
        for i in range(2):
            axs[0, i].set_ylim([0, 10])
            axs[1, i].set_ylim([0, 2])

    for ax in axs.flat:
        ax.tick_params(axis='both', labelsize=labelsize_font)
    plt.tight_layout()
    plt.savefig(f'q_surface_{target}_2x2.png', dpi=300)
    plt.show()

# Example usage:
plot_li_surface_heatflux_2x2_custom(Case_t0, Case_t5, y_pos, label0='t = 0 s', label5='t = 5 s', log =True)