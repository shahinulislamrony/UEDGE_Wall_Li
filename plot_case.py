# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from matplotlib.collections import LineCollection
import json
import re

from matplotlib.collections import LineCollection



sxnp = np.array([
    1.65294727e-08, 1.68072047e-02, 1.57913285e-02, 1.47307496e-02,
    1.37802350e-02, 1.28710288e-02, 1.19247238e-02, 1.09516290e-02,
    1.02117906e-02, 2.14465429e-02, 1.11099457e-02, 1.26587791e-02,
    1.45917191e-02, 1.66915905e-02, 1.94826593e-02, 2.23582531e-02,
    2.53806793e-02, 2.94667140e-02, 3.39163144e-02, 3.85707117e-02,
    4.34572856e-02, 4.70735328e-02, 5.17684150e-02, 5.64336990e-02,
    5.97825750e-02, 6.08629236e-08
])

yyrb = np.array([
    -0.05544489, -0.0507309 , -0.04174753, -0.03358036, -0.02614719,
    -0.01935555, -0.01316453, -0.00755603, -0.00245243,  0.00497426,
     0.012563  ,  0.01795314,  0.02403169,  0.03088529,  0.03865124,
     0.04744072,  0.05723254,  0.06818729,  0.0804908 ,  0.09413599,
     0.10907809,  0.12501805,  0.14181528,  0.15955389,  0.17792796,
     0.18716496
])

def eval_Li_evap_at_T_Cel(temperature):
    """Calculate lithium evaporation flux at a given temperature in Celsius."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    tempK = temperature + 273.15
    if tempK <= 0:
        raise ValueError("Temperature must be above absolute zero (-273.15\u00b0C).")
    vpres1 = 760 * 10 ** (a1 + b1 / tempK)  # Vapor pressure
    sqrt_argument = xm1 * tempK
    if sqrt_argument <= 0:
        raise ValueError("Invalid value for sqrt: xm1 * tempK has non-positive values.")
    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap

def get_available_indices(folder, prefix, suffix):
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist.")
        return []
    files = os.listdir(folder)
    indices = []
    pattern = re.compile(rf"{prefix}(\d+){suffix}")
    for f in files:
        m = pattern.match(f)
        if m:
            indices.append(int(m.group(1)))
    indices.sort()
    return indices

def replace_with_linear_interpolation(arr):
    arr = pd.Series(arr)
    arr_interpolated = arr.interpolate(method='linear', limit_direction='both')
    return arr_interpolated.bfill().ffill().to_numpy()

import os
import numpy as np
import pandas as pd

def load_csv_data(filename, row=None, col=None):
    """
    Tries to load data from a CSV file robustly, using numpy or pandas.
    Returns a single value if row/col specified, or the array otherwise.
    """
    try:
        data = np.loadtxt(filename, delimiter=',')
    except Exception:
        try:
            data = pd.read_csv(filename, header=None).values
        except Exception as e:
            print(f"Could not load {filename}: {e}")
            return np.nan
    if row is not None and col is not None:
        try:
            return data[row, col]
        except Exception as e:
            print(f"Index error in {filename} at [{row},{col}]: {e}")
            return np.nan
    elif row is not None:
        try:
            return data[row]
        except Exception as e:
            print(f"Index error in {filename} at [{row}]: {e}")
            return np.nan
    else:
        return data

import os
import numpy as np
import pandas as pd


def load_data_auto(filename, row=None, col=None):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return np.nan
    try:
        if filename.endswith('.npy'):
            data = np.load(filename)
        else:
            # Force numeric, replace non-numeric with NaN
            data = pd.read_csv(filename, header=None).apply(pd.to_numeric, errors='coerce').values
    except Exception as e:
        print(f"Could not load {filename}: {e}")
        return np.nan
    try:
        if row is not None and col is not None:
            row = int(round(row))
            col = int(round(col))
            return data[row, col]
        elif row is not None:
            row = int(round(row))
            return data[row]
        else:
            return data
    except Exception as e:
        print(f"Index error in {filename} at [{row},{col}]: {e}")
        return np.nan

def process_dataset(data_path, dt, sep=8, ixmp=36, sxnp=None, eval_Li_evap_at_T_Cel=None):
    dirs = {
        "q_perp": os.path.join(data_path, 'q_perp'),
        "Tsurf_Li": os.path.join(data_path, 'Tsurf_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "C_Li_omp": os.path.join(data_path, 'C_Li_omp'),
        "n_Li3": os.path.join(data_path, 'n_Li3'),
        "n_Li2": os.path.join(data_path, 'n_Li2'),
        "n_Li1": os.path.join(data_path, 'n_Li1'),
        "n_e": os.path.join(data_path, 'n_e'),
        "T_e": os.path.join(data_path, 'T_e'),
        "Li": os.path.join(data_path, 'Gamma_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
    }
    available_indices = get_available_indices(dirs["Tsurf_Li"], "T_surfit_", ".csv")
    max_value_tsurf, max_q, evap_flux_max, max_q_Li_list = [], [], [], []
    C_Li_omp, Te, n_Li_total, ne, phi_sput, evap, ad, total, n_Li3 = [], [], [], [], [], [], [], [], []

    for i in available_indices:
        filenames = {
            "tsurf": os.path.join(dirs["Tsurf_Li"], f'T_surfit_{i}.csv'),
            "qsurf": os.path.join(dirs["q_perp"], f'q_perpit_{i}.csv'),
            "qsurf_Li": os.path.join(dirs["q_Li_surface"], f'q_Li_surface_{i}.csv'),
            "C_Li": os.path.join(dirs["C_Li_omp"], f'CLi_prof_{i}.csv'),
            "n_Li3": os.path.join(dirs["n_Li3"], f'n_Li3_{i}.csv'),
            "n_Li2": os.path.join(dirs["n_Li2"], f'n_Li2_{i}.csv'),
            "n_Li1": os.path.join(dirs["n_Li1"], f'n_Li1_{i}.csv'),
            "ne": os.path.join(dirs["n_e"], f'n_e_{i}.npy'),  
            "Te": os.path.join(dirs["T_e"], f'T_e_{i}.csv'),
            "PS": os.path.join(dirs["Li"], f'PhysSput_flux_{i}.csv'),
            "Evap": os.path.join(dirs["Li"], f'Evap_flux_{i}.csv'),
            "Ad": os.path.join(dirs["Li"], f'Adstom_flux_{i}.csv'),
            "Total": os.path.join(dirs["Li"], f'Total_Li_flux_{i}.csv')
        }
        # Fallback for ne if .npy not found
        if not os.path.exists(filenames["ne"]):
            filenames["ne"] = os.path.join(dirs["n_e"], f'n_e_{i}.csv')

        max_tsurf = max_q_i = max_q_Li_i = C_Li_i = Te_i = n_Li3_i = n_Li2_i = n_Li1_i = ne_i = phi_sput_i = evap_i = ad_i = total_i = np.nan

        max_tsurf = np.nanmax(load_data_auto(filenames["tsurf"]))
        max_q_i = np.nanmax(load_data_auto(filenames["qsurf"]))
        max_q_Li_i = np.nanmax(load_data_auto(filenames["qsurf_Li"]))
        C_Li_i = load_data_auto(filenames["C_Li"], row=sep)
        Te_i = load_data_auto(filenames["Te"], row=ixmp, col=sep)
        n_Li3_i = load_data_auto(filenames["n_Li3"], row=ixmp, col=sep)
        n_Li2_i = load_data_auto(filenames["n_Li2"], row=ixmp, col=sep)
        n_Li1_i = load_data_auto(filenames["n_Li1"], row=ixmp, col=sep)
        ne_i = load_data_auto(filenames["ne"], row=ixmp, col=sep)
        ps_arr = load_data_auto(filenames["PS"])
        evap_arr = load_data_auto(filenames["Evap"])
        ad_arr = load_data_auto(filenames["Ad"])
        total_arr = load_data_auto(filenames["Total"])

        phi_sput_i = np.sum(ps_arr * sxnp) 
        evap_i = np.sum(evap_arr * sxnp) 
        ad_i = np.sum(ad_arr * sxnp)
        total_i =   phi_sput_i +   evap_i +   ad_i
        max_value_tsurf.append(max_tsurf)

        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)
        Te.append(Te_i)
        n_Li3.append(n_Li3_i)
        n_Li_total.append(n_Li3_i + n_Li2_i + n_Li1_i)
        ne.append(ne_i)
        phi_sput.append(phi_sput_i)
        evap.append(evap_i)
        ad.append(ad_i)
        total.append(total_i)

    # Interpolate missing values
    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)
    n_Li_total = replace_with_linear_interpolation(n_Li_total)
    Te = replace_with_linear_interpolation(Te)
    ne = replace_with_linear_interpolation(ne)
    phi_sput = replace_with_linear_interpolation(phi_sput)
    evap = replace_with_linear_interpolation(evap)
    ad = replace_with_linear_interpolation(ad)
    total = replace_with_linear_interpolation(total)
    nLi3 = replace_with_linear_interpolation(n_Li3)

    evap_flux_max = []
    for max_tsurf_val in max_value_tsurf:
        if not np.isnan(max_tsurf_val) and eval_Li_evap_at_T_Cel is not None:
            try:
                evap_flux = eval_Li_evap_at_T_Cel(max_tsurf_val)
            except Exception as e:
                print(f"Error calculating evaporation flux: {e}")
                evap_flux = np.nan
        else:
            evap_flux = np.nan
        evap_flux_max.append(evap_flux)
    evap_flux_max = replace_with_linear_interpolation(evap_flux_max)
    q_surface = np.array(max_q) - 2.26e-19 * np.array(evap_flux_max)
    time_axis = dt * np.arange(1, len(max_q) + 1)

    return (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
            max_q_Li_list, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3)


parent_dir = '/global/u1/s/shahinul/NSTX_PoP/revised_code_reviweres/dt_scan'
folders = {
    "nx_P5": os.path.join(parent_dir, "dt5ms", "C_Li_omp"),
    "nx_P6": os.path.join(parent_dir, "dt10ms", "C_Li_omp"),
    "nx_P7": os.path.join(parent_dir, "dt20ms", "C_Li_omp"),
    "nx_P8": os.path.join(parent_dir, "dt30ms", "C_Li_omp"),
    "nx_P9": os.path.join(parent_dir, "dt40ms", "C_Li_omp"),
}
def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return 0
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])

file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}
nx_P5 = file_counts["nx_P5"]
nx_P6 = file_counts["nx_P6"]
nx_P7 = file_counts["nx_P7"]
nx_P8 = file_counts["nx_P8"]
nx_P9 = file_counts["nx_P9"]



datasets = [


  {
        'path': os.path.join(parent_dir, 'dt40ms'),
        'nx': nx_P9,
        'dt': 40e-3,
        'label_tsurf': 'dt: 40ms'
    },

]


colors = ['r', 'g', 'b', 'k', 'm', 'y', 'c', 'purple']


ymax = 0
for dataset in datasets:
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, total, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    ymax = max(ymax, np.max(phi_sput), np.max(evap), np.max(ad), np.max(total))

plt.figure(figsize=(10, 6))

for dataset in datasets:
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, total, _) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    plt.plot(time_axis, phi_sput, label=dataset['label_tsurf'] + ' - Physical Sputtering')
    plt.plot(time_axis, evap, label=dataset['label_tsurf'] + ' - Evaporation')
    plt.plot(time_axis, ad, label=dataset['label_tsurf'] + ' - Ad-atom', linestyle='--')

plt.xlabel('Time')
plt.ylabel('Flux')
plt.ylim([0, ymax*1.05])
plt.legend()
plt.tight_layout()
plt.savefig('li_flux.png', dpi=300)
plt.close()

# 1. Three-panel plot: q_surf, T_surf, C_Li_omp
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    ax2.plot(time_axis, max_value_tsurf, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    ax3.plot(time_axis, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 10])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 15])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.savefig('qsurf_T_surf_CLi_omp.png', dpi=300)
plt.show()

# 2. Two-panel plot: short time axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharex=True)
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis,
     max_q_Li, C_Li_omp, n_Li_total, Te, ne, phi_sput, evap, ad, total, n_Li3) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, '-', linewidth=1.5, label=dataset["label_tsurf"], color=color)
    ax2.plot(time_axis, max_value_tsurf, '-', linewidth=1.5, label=dataset["label_tsurf"], color=color)

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=14)
ax1.set_xlim([0, 1.5])
ax1.set_ylim([0, 10])
ax1.grid(True, linestyle='--', linewidth=0.5)
ax1.tick_params(axis='both', labelsize=12)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=14)
ax2.set_xlabel('t$_{sim}$ (s)', fontsize=14)
ax2.set_ylim([0, 750])
ax2.grid(True, linestyle='--', linewidth=0.5)
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(loc='best', fontsize=12, ncol=2)

plt.tight_layout()
plt.savefig('qsurf_T_surf.pdf', format='pdf', bbox_inches='tight')
plt.savefig('qsurf_T_surf.png', dpi=600, bbox_inches='tight')
plt.savefig('qsurf_T_surfdt.eps', format='eps', bbox_inches='tight')
plt.show()

# 3. Emission mechanisms
fig, ax = plt.subplots(figsize=(4, 3))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, time_axis, _, _, _, _, _, phi_sput, evap, ad, _,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    ax.plot(time_axis, evap, '-', linewidth=2, color='red', label='Evaporation' if idx == 0 else "")
    ax.plot(time_axis, phi_sput, '--', linewidth=2, color='blue', label='Phy. Sput' if idx == 0 else "")
    ax.plot(time_axis, ad, ':', linewidth=3, color='green', label='Ad-atom' if idx == 0 else "")

ax.set_xlabel('t$_{simulation}$ (s)', fontsize=16)
ax.set_ylabel('$\phi_{Li}^{Emitted}$ (atom/s)', fontsize=16)
ax.set_xlim([0, 5])
ax.set_ylim([1e15, 1e23])
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('Phi_Li_combined.png', dpi=300)
plt.savefig('Phi_Li_combined.eps', format='eps', bbox_inches='tight')
plt.show()

# 4. T_surf vs C_Li_omp
plt.figure(figsize=(5, 4))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, _, _, _, _, _, C_Li_omp, _, _, _, _, _, _, _,_) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(max_value_tsurf, C_Li_omp * 100, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')
plt.legend(fontsize=14)
plt.ylim([0, 15])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_CLi_omp.png', dpi=300)
plt.show()

# 5. T_surf vs n_Li3
plt.figure(figsize=(5, 4))
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (max_value_tsurf, _, _, _, _, _, _, n_Li_total, _, _, _, _, _, _, n_Li3) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(max_value_tsurf, n_Li3, '-', linewidth=2, label=dataset["label_tsurf"], color=color)

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=14)
plt.ylim([0, 2e18])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('T_surf_nLi_omp.png', dpi=300)
plt.show()

# 6. Total Li emission vs n_Li3
plt.figure(figsize=(4, 2.25))
all_n_Li3 = []
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, _, _, _, n_Li_total, _, _, _, _, _, total,n_Li3) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(total / 1e21, n_Li3, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    all_n_Li3.append(n_Li3)

ymax = max([np.nanmax(n) if np.iterable(n) else n for n in all_n_Li3]) * 1.05

plt.xlabel("$\phi_{Li}$ ($10^{21}$atom/s)", fontsize=18)
plt.ylabel("n$_{Li-sep}^{omp}$ (m$^{-3}$)", fontsize=18)
plt.legend(fontsize=11)
plt.ylim([0, ymax])
plt.xlim([0, 2])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('Cs_Phi_Li_nLi_omp.eps', format='eps', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.jpg', format='jpg', dpi=600)
plt.savefig('Cs_Phi_Li_nLi_omp.png', format='png', dpi=300)
plt.show()

# 7. Total Li emission vs time
plt.figure(figsize=(5, 4))

all_totals = []  # To store all y-data for auto-scaling

for idx, dataset in enumerate(datasets):
    print(f"Processing dataset: {dataset['label_tsurf']}")
    (_, _, _, _, time_axis, _, _, _, _, _, _, _, _, total, n_Li3) = process_dataset(
        data_path=dataset['path'],
        dt=dataset['dt'],
        sxnp=sxnp,
        eval_Li_evap_at_T_Cel=eval_Li_evap_at_T_Cel
    )
    color = colors[idx % len(colors)]
    plt.plot(time_axis, total, '-', linewidth=2, label=dataset["label_tsurf"], color=color)
    all_totals.append(total)

# Find global max for y-axis
ymax = max([t.max() for t in all_totals]) * 1.05

plt.xlabel("t$_{sim}$ (s)", fontsize=18)
plt.ylabel("$\phi_{Li}$ (atom/s)", fontsize=18)
plt.legend(fontsize=14)
plt.ylim([0, ymax])
plt.xlim([0, 5])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig('tsim_Phi_Li_omp.png', dpi=300)
plt.show()





