import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


vol = np.loadtxt(r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX-U_g116313\Old_data_until02072025\nsep_5.7e19_for_power_scan\nsep_5.7e19_for_power_scan\vol.txt')
print('Vol read is done and len is ', len(vol))
y = np.array([ -0.04339678, -0.03580569, -0.0287926, -0.02241815, -0.01662467,
                    -0.01138454, -0.00667084, -0.00222074, 0.00074189, 0.00253791, 0.00484185,
                    0.00742937, 0.01039286, 0.0138512, 0.01783639, 0.02244914, 0.02780647,
                    0.03406285, 0.04135004, 0.04984127, 0.05964307, 0.07102367, 0.08421488,
                    0.09925538, 0.10724673])
# Define the datasets and their parameters


def count_files_in_folder(folder_path):
    return len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))])


folders = {
    "nx_P6": r"C:\UEDGE_run_Shahinul\NSTX_U\PePi6MW_Dn0.35Chi0.5_fneut0.35_check_saveall/C_Li_omp",
    "nx_P8": r"C:\UEDGE_run_Shahinul\NSTX_U\PePi8MW_Dn0.35Chi0.5_fneut0.35_check_saveall/C_Li_omp",
    "nx_P9": r"C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb\C_Li_omp",
    "nx_P92": r"C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb_t5ms\C_Li_omp",
}


file_counts = {key: count_files_in_folder(path) for key, path in folders.items()}

nx_P92 = file_counts["nx_P92"]
nx_P9 = file_counts["nx_P9"]
nx_P8 = file_counts["nx_P8"]
nx_P6 = file_counts["nx_P6"]

datasets = [
    
    
    {
     'path': r'C:\UEDGE_run_Shahinul\NSTX_U\PePi6MW_Dn0.35Chi0.5_fneut0.35_check_saveall',
     'nx': nx_P6,
     'dt': 10e-3,
     'label_tsurf': 'P$_{in}$: 6MW'
 },
    
    {
     'path': r'C:\UEDGE_run_Shahinul\NSTX_U\PePi8MW_Dn0.35Chi0.5_fneut0.35_check_saveall',
     'nx': nx_P8,
     'dt': 10e-3,
     'label_tsurf': 'P$_{in}$: 8MW'
 },
    
 #    {
 #     'path': r'C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb',
 #     'nx': nx_P9,
 #     'dt': 10e-3,
 #     'label_tsurf': 'P$_{in}$: 9MW'
 # },
    
 #    {
 #     'path': r'C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb_t5ms',
 #     'nx': nx_P92,
 #     'dt': 10e-3,
 #     'label_tsurf': 'P$_{in}$: 9MW-2'
 # },


]


fig, ax1 = plt.subplots()

def read_csv(filepath):
    try:
        return pd.read_csv(filepath).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


for dataset in datasets:
    data_path = dataset['path'].rstrip("\\")
    nx = dataset['nx']
    dt = dataset['dt']
    label_tsurf = dataset['label_tsurf']

    max_q = []

    q_perp_dir = os.path.join(data_path, 'q_perp')
    print(f"Checking q_perp directory: {q_perp_dir}")

    for i in range(1, nx):
        filename_qsurf = os.path.join(q_perp_dir, f'q_perpit_{i}.0.csv')  # Adjust extension if needed
        print(f"Checking file: {filename_qsurf}")

        if not os.path.exists(filename_qsurf):
            print(f"File not found: {filename_qsurf}")
            max_q.append(np.nan)  # Append NaN for missing files
            continue

        try:  
            q_perp_data = pd.read_csv(filename_qsurf).values  
            max_value = np.max(q_perp_data)  
            max_q.append(max_value)
        except Exception as e:
            print(f"Error loading file {filename_qsurf}: {e}")
            max_q.append(np.nan)  # Append NaN for errors

    # Replace NaN values with the average of the previous and next values
    def replace_with_avg(arr):
        arr = np.array(arr)
        for idx in range(len(arr)):
            if np.isnan(arr[idx]):
                prev_val = arr[idx - 1] if idx > 0 else np.nan
                next_val = arr[idx + 1] if idx < len(arr) - 1 else np.nan
                arr[idx] = np.nanmean([prev_val, next_val])
        return arr

    max_q = replace_with_avg(max_q)

    # Plotting
    time_axis = np.arange(1, nx) * dt
    plt.plot(time_axis, np.divide(max_q, 1e6), marker='*', linestyle='-', label=f'({label_tsurf})')

# Finalize plot
plt.xlabel('t$_{sim}$ (s)', fontsize=20)
plt.ylabel("q$_{\perp, Odiv}^{max}$ (MW/m$^2$)", fontsize=20, color='k')
plt.ylim([0, 6])
plt.xlim([0, 5])
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.legend(loc='best', fontsize=14)
plt.grid()
plt.show()


for dataset in datasets:
    data_path = dataset['path'].rstrip("\\")
    nx = dataset['nx']
    dt = dataset['dt']
    label_tsurf = dataset['label_tsurf']


    max_Tsurf  =[]
    T_surf_dir = os.path.join(data_path, 'Tsurf_Li')
    
    for i in range(1, nx):
        filename_tsurf = os.path.join(T_surf_dir, f'T_surfit_{i}.0.csv')

          
        if not os.path.exists(filename_qsurf):
            print(f"File not found: {filename_qsurf}")
            continue

        try:  
          
            
            T_surf_data = pd.read_csv(filename_tsurf).values  
            max_value2 = np.max(T_surf_data)  
            max_Tsurf.append(max_value2)
        except Exception as e:
            print(f"Error loading file {filename_qsurf}: {e}")
    plt.plot(np.arange(1, nx) * dt, max_Tsurf, marker ='*', linestyle='-', label=f'({label_tsurf})')


max_q = np.array(max_q)
max_Tsurf = np.array(max_Tsurf)

plt.xlabel('t$_{sim}$ (s)', fontsize = 20)
plt.ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=20, color='k')
plt.ylim([0,700])
plt.xlim([0, 5])
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
# Add legends
plt.legend(loc='best', fontsize =14)
plt.grid()
plt.show()

#fig, ax1 = plt.subplots(figsize=(10, 6))
#ax2 = ax1.twinx()


colors = ['r', 'g', 'k', 'b', 'm', 'y', 'c', 'purple']  # More colors



def interpolate_data(x, y):
    """Interpolate missing data using linear interpolation."""
    series = pd.Series(y, index=x)
    interpolated = series.interpolate(method='linear', limit_direction='forward')
    return interpolated.values

# Assuming 'datasets' is a list of dictionaries, each representing a dataset
for idx, dataset in enumerate(datasets):
    data_path = dataset['path'].rstrip("\\")
    nx = dataset['nx']
    dt = dataset['dt']
    label_tsurf = dataset['label_tsurf']

    max_CLi = []
    CLi_avg = []
    max_T = []

    n_Li1_dir = os.path.join(data_path, 'n_Li1')
    n_Li2_dir = os.path.join(data_path, 'n_Li2')
    n_Li3_dir = os.path.join(data_path, 'n_Li3')
    n_e_dir = os.path.join(data_path, 'n_e')
    T_surf_dir = os.path.join(data_path, 'Tsurf_Li')

    for i in range(1, nx):
        filename_nLi1 = os.path.join(n_Li1_dir, f'n_Li1_{i}.0.csv')
        filename_nLi2 = os.path.join(n_Li2_dir, f'n_Li2_{i}.0.csv')
        filename_nLi3 = os.path.join(n_Li3_dir, f'n_Li3_{i}.0.csv')
        filename_ne = os.path.join(n_e_dir, f'n_e_{i}.0.csv.npy')
        filename_tsurf = os.path.join(T_surf_dir, f'T_surfit_{i}.0.csv')

        try:
            n_Li1_data = np.loadtxt(filename_nLi1)  # Adjust delimiter as needed
            n_Li2_data = np.loadtxt(filename_nLi2)
            n_Li3_data = np.loadtxt(filename_nLi3)
            ne_data = np.load(filename_ne)  # Removed `.values`
            xpt1 = 12
            oxpt = 92
            iy = 8
            # Ensure the shapes align for the volume and data slices
            n_Li_sep = np.sum((n_Li1_data[xpt1:oxpt, iy] + n_Li2_data[xpt1:oxpt, iy] + n_Li3_data[xpt1:oxpt, iy]) * vol[xpt1:oxpt, iy])
            C_Li = n_Li_sep / (np.sum(ne_data[xpt1:oxpt, iy] * vol[xpt1:oxpt, iy]))

            CLi_avg.append(C_Li)
            T_surf_data2 = pd.read_csv(filename_tsurf).values
            max_value2 = np.max(T_surf_data2)
            max_T.append(max_value2)

        except Exception as e:
            print(f"Error loading file {filename_tsurf}: {e}")

    # Ensure CLi_avg and max_T have the same length before plotting
    if len(CLi_avg) == len(max_T):
        # Interpolate missing data if any NaN values are present
        if np.any(np.isnan(CLi_avg)) or np.any(np.isnan(max_T)):
            valid_indices = np.where(~np.isnan(CLi_avg))[0]
            interpolated_CLi_avg = interpolate_data(valid_indices, np.array(CLi_avg)[valid_indices])

            valid_indices_T = np.where(~np.isnan(max_T))[0]
            interpolated_max_T = interpolate_data(valid_indices_T, np.array(max_T)[valid_indices_T])

            # After interpolation, update the original lists
            CLi_avg = interpolated_CLi_avg
            max_T = interpolated_max_T

        #plt.figure (figsize = (6,4))
        plt.plot(max_T, np.multiply(CLi_avg, 100), linestyle='--', linewidth = 2,
                 label=f'{label_tsurf}', color=colors[idx % len(colors)])  # Use a different color for each dataset

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=20)
plt.ylabel("C$_{Li, sep}^{average}$ (%)", fontsize=20, color='k')
plt.axhline(3, color='black', linestyle=':', linewidth=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([0, 10])
plt.grid()
plt.legend(loc='best', ncol=1, fontsize=12)
plt.show()




data_path = dataset['path'].rstrip("\\")
nx = dataset['nx']
dt = dataset['dt']
label_tsurf = dataset['label_tsurf']

CLi_omp = []
max_Tsurf_plot2 = []

C_Li_dir = os.path.join(data_path, 'C_Li_omp')
T_surf_dir = os.path.join(data_path, 'Tsurf_Li')

for i in range(1, nx):
    filename_CLi = os.path.join(C_Li_dir, f'CLi_prof{i}.0.csv')  
    filename_tsurf = os.path.join(T_surf_dir, f'T_surfit_{i}.0.csv')

    try:
        # Process C_Li data
        if os.path.exists(filename_CLi):
            C_Li_data = pd.read_csv(filename_CLi).values
            max_value4 = np.average(C_Li_data)
            CLi_omp.append(max_value4)
        else:
            print(f"File not found: {filename_CLi}")
            CLi_omp.append(np.nan)

        # Process T_surf data
        if os.path.exists(filename_tsurf):
            T_surf_data2 = pd.read_csv(filename_tsurf).values
            max_value2 = np.max(T_surf_data2)
            max_Tsurf_plot2.append(max_value2)
        else:
            print(f"File not found: {filename_tsurf}")
            max_Tsurf_plot2.append(np.nan)

    except Exception as e:
        print(f"Error processing files at step {i}: {e}")
        CLi_omp.append(np.nan)
        max_Tsurf_plot2.append(np.nan)

# Function to replace NaN with interpolation
def interpolate_missing_data(arr):
    arr = np.array(arr)
    # Identify the indices of non-NaN values
    not_nan_indices = np.where(~np.isnan(arr))[0]
    # Identify the indices of NaN values
    nan_indices = np.where(np.isnan(arr))[0]

    if len(not_nan_indices) > 1:  # If there are enough data points
        # Interpolate using linear interpolation
        interpolate_func = interp1d(not_nan_indices, arr[not_nan_indices], kind='linear', fill_value="extrapolate")
        arr[nan_indices] = interpolate_func(nan_indices)
    
    return arr

# Interpolate missing data in CLi_omp and max_Tsurf_plot2
CLi_omp = interpolate_missing_data(CLi_omp)
max_Tsurf_plot2 = interpolate_missing_data(max_Tsurf_plot2)

# Plotting
plt.plot(
    max_Tsurf_plot2, 
    np.multiply(CLi_omp, 100), 
    marker='*', 
    linestyle='-', 
    label=f'({label_tsurf})'
)

# Finalize plot
plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=20, color='k')
#plt.xlabel('t$_{sim}$ (s)', fontsize = 20)
plt.ylabel("C$_{Li-sep}^{OMP}$ (%)", fontsize=20, color='k')
plt.axhline(3, color='red', linestyle='--', linewidth=2, label='y = 3')
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim([0.0, 8])
plt.xlim([0, 600])
#plt.yscale('log')
plt.tight_layout()  
#plt.legend(loc='best', fontsize=12)
plt.grid()
plt.show()




paths = [
 r'C:\UEDGE_run_Shahinul\NSTX_U\PePi8MW_Dn0.35Chi0.5_fneut0.35_check_saveall\\'
]

it = np.linspace(1, 500, 500)
dt = 10e-3

myleg = [ 'P - 6 MW', 'Y$_{ad}$ = high']

marker = 'o' 
line_style = '-.sr' 

def read_csv(filepath):
    try:
        return pd.read_csv(filepath).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

Li_data = {}
for path in paths:
    Li_data[path] = {
        'phi_Li': read_csv(os.path.join(path, 'Phi_Li.csv')),  
        'Li_all': read_csv(os.path.join(path, 'Li_all.csv')) 
    }

plt.figure()
start = 0
for i, path in enumerate(paths):  
    phi_Li = Li_data[path]['phi_Li']  
    plt.plot(start+it*dt, phi_Li.reshape(-1), linewidth=2, marker = '*', markersize = 12, label=myleg[i] if i < len(myleg) else f'Path {i}')  # Simple label for each path

plt.xlabel('t$_{simulation}$ (s)', fontsize=18)
plt.ylabel('$\phi_{Li}^{source}$ (atom/s)', fontsize=18)
plt.grid(True)
#plt.yscale('log')
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
#plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig('ne_odiv.png', dpi=300, bbox_inches='tight')
plt.show()


it = np.linspace(1, 500, 500)
dt = 10e-3
ix = 5

for i, path in enumerate(paths):  # Use 'paths' instead of 'path'
    Li_source_odiv = Li_data[path]['Li_all'][:,0]  
    Li_source_idiv = Li_data[path]['Li_all'][:,1] 
    Li_source_wall = Li_data[path]['Li_all'][:,3] 
    Li_radiaiton = Li_data[path]['Li_all'][:,4]
    Li_flux_odiv = Li_data[path]['Li_all'][:,5]
    Li_flux_wall = Li_data[path]['Li_all'][:,6]
    Li_flux_idiv = Li_data[path]['Li_all'][:,7]
    Li_pump_odiv = Li_data[path]['Li_all'][:,8]
    Li_pump_wall = Li_data[path]['Li_all'][:,9]
    Li_pump_idiv = Li_data[path]['Li_all'][:,10]
    Li_ionization = Li_data[path]['Li_all'][:,11]
    
Li_source_odiv = Li_source_odiv[:-1] + phi_Li.reshape(-1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

ax1.plot(it * dt, Li_source_odiv, linewidth=2, marker='*', color='blue', markersize=8, label='Odiv') 
ax1.plot(it * dt, abs(Li_source_idiv[:-1]), linewidth=2, marker='d', color='red', markersize=8, label='idiv') 
ax1.plot(it * dt, Li_source_wall[:-1], linewidth=2, marker='h', color='green', markersize=8, label='wall') 
ax1.set_ylabel(r'$\phi_{Li}^{source}$ (atom/s)', fontsize=18)
ax1.set_yscale('log')
ax1.set_ylim([1e19, 2e23])
ax1.tick_params(axis='both', which='major', labelsize=14)

ax1.set_xlim([0, ix])
ax1.grid(True)
ax1.legend(loc='best', fontsize=14, ncol=2)

ax2.plot(it * dt, Li_flux_odiv[:-1], linewidth=2, marker='*', color='blue', markersize=8, label='Odiv') 
ax2.plot(it * dt, abs(Li_flux_idiv[:-1]), linewidth=2, marker='d', color='red', markersize=8, label='idiv') 
ax2.plot(it * dt, Li_flux_wall[:-1], linewidth=2, marker='h', color='green', markersize=8, label='wall') 
ax2.set_xlabel(r't$_{simulation}$ (s)', fontsize=18)
ax2.set_ylabel(r'$\phi_{Li}^{deposit}$ (atom/s)', fontsize=18)
ax2.set_yscale('log')
ax2.set_ylim([1e19, 2e23])
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True)
#ax2.legend(loc='best', fontsize=14)
plt.tight_layout()

plt.savefig('ne_odiv_shared_x.png', dpi=300, bbox_inches='tight')

plt.show()


plt.figure()    
plt.plot(it*dt, Li_source_odiv, linewidth=2, marker = '*', color ='blue', markersize = 8, label= 'Odiv') 
plt.plot(it*dt, abs(Li_source_idiv[:-1]), linewidth=2, marker = 'd', color ='red', markersize = 8, label= 'idiv') 
plt.plot(it*dt, Li_source_wall[:-1], linewidth=2, marker = 'h', color ='green', markersize = 8, label= 'wall') 
# plt.plot(start+it*dt, Li_ionization[:-1], linewidth=2, marker = '<', color ='cyan', markersize = 8, label= 'ionization') 
plt.xlabel('t$_{simulation}$ (s)', fontsize=18)
plt.ylabel('$\phi_{Li}^{source}$ (atom/s)', fontsize=18)
plt.grid(True)
plt.yscale('log')
ymax = np.max(Li_ionization)
plt.ylim([1e19, 2e23])
plt.xlim([0, ix])
#plt.xticks(np.arange(0, 0.21, 0.05), fontsize=14)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.legend(loc='best', fontsize=14, ncol =2)
plt.tight_layout()
plt.savefig('ne_odiv.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure()    
plt.plot(max_Tsurf, Li_source_odiv, linewidth=2, marker = '*', color ='green', markersize = 8, label= 'Odiv') 
plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=20, color='k')
plt.ylabel('$\phi_{Li}^{source}$ (atom/s)', fontsize=18)
plt.grid(True)
plt.yscale('log')
ymax = np.max(Li_ionization)
plt.ylim([1e21, 2e23])
plt.xlim([0, 600])
#plt.xticks(np.arange(0, 0.21, 0.05), fontsize=14)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
#plt.legend(loc='best', fontsize=14, ncol =2)
plt.tight_layout()
plt.savefig('ne_odiv.png', dpi=300, bbox_inches='tight')
plt.show()

Li_source = Li_source_odiv + abs(Li_source_idiv[:-1]) + Li_source_wall[:1]
Li_pump =  Li_pump_odiv +  abs(Li_pump_idiv) +  Li_pump_wall

plt.plot(start+it*dt, Li_flux_odiv[:-1], linewidth=2, marker = '*', color ='blue', markersize = 8, label= 'Odiv') 
plt.plot(start+it*dt, abs(Li_flux_idiv[:-1]), linewidth=2, marker = 'd', color ='red', markersize =8, label= 'idiv') 
plt.plot(start+it*dt, Li_flux_wall[:-1], linewidth=2, marker = 'h', color ='green', markersize =8, label= 'wall') 

plt.xlabel('t$_{simulation}$ (s)', fontsize=18)
plt.ylabel('$\phi_{Li}^{deposit}$ (atom/s)', fontsize=18)
plt.grid(True)
#plt.yscale('log')
ymax = np.max(Li_source_odiv)
plt.ylim([1e19, 2e23])
plt.xlim([0, ix])
#plt.xticks(np.arange(0, 0.21, 0.05), fontsize=14)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.yscale('log')
plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig('ne_odiv.png', dpi=300, bbox_inches='tight')
plt.show()




plt.figure()
plt.plot(start+it*dt, Li_radiaiton[:-1]/1e6, linewidth=2, marker = 's', color ='red', markersize = 8, label= 'Li-rad') 
plt.xlabel('t$_{simulation}$ (s)', fontsize=18)
plt.ylabel('$P_{Li-rad}$ (MW)', fontsize=18)
plt.grid(True)
#plt.yscale('log')
ymax = np.max(Li_radiaiton/1e6)
plt.ylim([0, 1])
plt.xlim([0, ix])
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig('ne_odiv.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(Li_source , Li_pump[:-1], '--r', linewidth=2, marker = '*', markersize=8)
plt.xlabel('$\phi_{Li}^{source}$ (atom/s)', fontsize=18)
plt.ylabel('$\phi_{Li}^{pump}$ (atom/s)', fontsize=18)
plt.title('Li Sourcing via sputtering is pumped out by BC on wall and div')
plt.grid(True)
plt.box(True)
#plt.xlim([0, 1])
#plt.xticks(np.arange(0, 0.21, 0.05), fontsize=14)
#plt.yscale('log')
#plt.legend(loc='best', fontsize=14, ncol=2, title_fontsize='13')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#ymax = np.max(phi_Li_odiv)
#plt.ylim([1e15, 2e22])
plt.tight_layout()
#plt.xlim([0, 30])
plt.savefig('Li_all.png', dpi=300, bbox_inches='tight')
plt.show()  

y = np.array([-0.09580207, -0.0883297, -0.07360328, -0.05940535, -0.04597069, 
            -0.03356609, -0.02240577, -0.01260851, -0.00402888,  0.00984205, 
             0.03007019,  0.04991498,  0.06846575,  0.08608859,  0.10297708, 
             0.1194364,   0.13546967,  0.15123152,  0.1666732,   0.18186758, 
             0.19708181,  0.21228164,  0.22750439,  0.2429528,   0.25864637])




cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()


q_perp_dir = os.path.join(data_path, 'q_perp')
print(f"Checking q_perp directory: {q_perp_dir}")

 
for i in range(1, nx, 1):
    filename = os.path.join(q_perp_dir, f'q_perpit_{i}.0.csv')
    df = pd.read_csv(filename)
    q_data = df  
    #Li_source_odiv = Li_data[path]['Li_all'][:,0] 
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    #plt.figure(figsize=(6, 4))
    ax.plot(y[:], q_data/1e6, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{div}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("q$_{\perp}^{Odiv}$ (MW/m$^2$)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim([0, 5.2])
plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()


Tsurf_dir = os.path.join(data_path, 'Tsurf_Li')
print(f"Checking Tsurf_dir directory: {Tsurf_dir}")

 
for i in range(1, nx, 1):
    filename = os.path.join(Tsurf_dir, f'T_surfit_{i}.0.csv')
    df = pd.read_csv(filename)
    q_data = df  
    #Li_source_odiv = Li_data[path]['Li_all'][:,0] 
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    #plt.figure(figsize=(6, 4))
    ax.plot(y[:], q_data, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{div}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("T$_{surf}$ (C)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim([0, 600])
plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()

Tsurf_dir = os.path.join(data_path, 'T_e')
print(f"Checking Tsurf_dir directory: {Tsurf_dir}")

 
for i in range(1, nx, 1):
    filename = os.path.join(Tsurf_dir, f'T_e_{i}.0.csv')
    df = pd.read_csv(filename)
    q_data = df  
    data = np.loadtxt(filename)
    numbers = data[105, :-1]
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    ax.plot(y[:], numbers, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{div}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("T$_{e}$ (eV)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()

Tsurf_dir = os.path.join(data_path, 'n_e')
print(f"Checking Tsurf_dir directory: {Tsurf_dir}")

 
for i in range(1, nx, 1):
    filename = os.path.join(Tsurf_dir, f'n_e_{i}.0.csv.npy')
   
    data = np.load(filename)
    numbers = data[105, :-1]
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    ax.plot(y[:], numbers, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{div}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("n$_{e}$ (m$^{-3}$)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()

Tsurf_dir = os.path.join(data_path, 'n_e')
print(f"Checking Tsurf_dir directory: {Tsurf_dir}")

yyc = np.array([-0.01547601, -0.01356714, -0.01062088, -0.00878213, -0.00706762,
       -0.00514406, -0.00318701, -0.00153801, -0.00041783,  0.00012821,
        0.00040441,  0.00071499,  0.00106491,  0.00146118,  0.00191115,
        0.00242573,  0.00300786,  0.00366598,  0.00440803,  0.00524829,
        0.0062062 ,  0.00729005,  0.00851666,  0.00991239,  0.01149827,
        0.01234134])
 
for i in range(1, nx, 1):
    filename = os.path.join(Tsurf_dir, f'n_e_{i}.0.csv.npy')
   
    data = np.load(filename)
    numbers = data[68, :-1]
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    ax.plot(yyc[:-1], numbers, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{omp}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("n$_{e}$ (m$^{-3}$)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim([0, 1e20])
#plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()

Tsurf_dir = os.path.join(data_path, 'n_Li3')
print(f"Checking Tsurf_dir directory: {Tsurf_dir}")

yyc = np.array([-0.01547601, -0.01356714, -0.01062088, -0.00878213, -0.00706762,
       -0.00514406, -0.00318701, -0.00153801, -0.00041783,  0.00012821,
        0.00040441,  0.00071499,  0.00106491,  0.00146118,  0.00191115,
        0.00242573,  0.00300786,  0.00366598,  0.00440803,  0.00524829,
        0.0062062 ,  0.00729005,  0.00851666,  0.00991239,  0.01149827,
        0.01234134])
 
for i in range(1, nx, 1):
    filename = os.path.join(Tsurf_dir, f'n_Li3_{i}.0.csv')
   
    data = np.loadtxt(filename)
    numbers = data[68, :-1]
    norm = plt.Normalize(1, np.max(Li_source_odiv))
    Li_flux_sput_Odiv_int_value =  Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))
       
    ax.plot(yyc[:-1], numbers, linewidth=2, marker='*', color=color, 
            label=myleg[i//2] if i//2 < len(myleg) else f'Path {i}')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize = 14)
ax.grid()
plt.xlabel("r$_{omp}$ - r$_{sep}$ (m)",fontsize=16)
plt.ylabel("n$_{Li3}$ (m$^{-3}$)",fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid()
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
plt.ylim([0, 1e19])
#plt.xlim([-0.1, 0.3])
plt.grid()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

data_path = r'C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb\\'

cmap = plt.get_cmap('turbo')  
nx = 500
fig, ax = plt.subplots()


q_perp_dir = os.path.join(data_path, 'q_perp')
print(f"Checking q_perp directory: {q_perp_dir}")


Li_source_odiv = Li_data[path]['Li_all'][:, 0]
norm = plt.Normalize(1, np.max(Li_source_odiv))


for i in range(1, nx, 1):
    filename = os.path.join(q_perp_dir, f'q_perpit_{i}.0.csv')
    
  
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue

  
    df = pd.read_csv(filename)
    q_data = df.values.flatten()  

    
    Li_flux_sput_Odiv_int_value = Li_source_odiv[i]
    color = cmap(norm(Li_flux_sput_Odiv_int_value))

    
    label = myleg[i // 2] if i // 2 < len(myleg) else f'Path {i}'
    ax.plot(y[:], q_data / 1e6, linewidth=2, marker='*', color=color, label=label)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for older versions of matplotlib
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize=14)


ax.grid()
plt.xlabel("r$_{div}$ - r$_{sep}$ (m)", fontsize=16)
plt.ylabel("q$_{\perp}^{Odiv}$ (MW/m$^2$)", fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.ylim([0, 6])
plt.xlim([-0.1, 0.3])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('Iteration_vs_heat.png', dpi=300)
plt.show()

data_pathnew = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_g128638\Li_Charge_States\Li_t1s_final_code\Sherwood_conference\PePi4MW\BC_n6e19_PePei4MW' 
nx = 21
zm = np.array([-1.46304807, -1.46302318, -1.46294318, -1.46290645, -1.46302463,
       -1.46330185, -1.46374697, -1.46444043, -1.4657913 , -1.46860783,
       -1.4743821 , -1.43673347, -1.27827892, -1.05656845, -0.89418889,
       -0.73449329, -0.57661493, -0.41856946, -0.24664877, -0.03003537,
        0.21746861,  0.46463413,  0.71043825,  0.94987593,  1.1374877 ,
        1.17681127,  1.07401023,  0.94476059,  0.82185689,  0.68567947,
        0.54101642,  0.3917916 ,  0.23928941,  0.08347535, -0.07457002,
       -0.26555928, -0.48428369, -0.69484072, -0.89354749, -1.07255523,
       -1.23096203, -1.36362301, -1.43835039, -1.4904326 , -1.51042288,
       -1.52491369, -1.53755058, -1.5490099 , -1.55960073, -1.56949581,
       -1.57880112, -1.58759221, -1.5959338 , -1.6  ])


# filename_CLi = r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX_g128638\Li_Charge_States\Li_t1s_final_code\Sherwood_conference\PePi4MW\BC_n6e19_PePei4MW\C_Li\C_Li_sep_all_21.0.csv'

# C_Li_data1 = pd.read_csv(filename_CLi).values.flatten()  # Ensure 1D array
   
# plt.plot(zm[:-1], C_Li_data1*100, '--*r')
# plt.xlabel('Z (m)', fontsize=20)
# plt.ylabel("C$_{Li, sep}^{average}$ (%)", fontsize=20, color='k')
# plt.xticks(fontsize=14) 
# plt.yticks(fontsize=14) 
# #plt.legend(loc='best', fontsize=14)
# plt.grid()
# plt.yscale('log')
# plt.show()






