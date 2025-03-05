import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def eval_Li_evap_at_T_Cel(temperature):
    """Calculate lithium evaporation flux at a given temperature in Celsius."""
    a1 = 5.055
    b1 = -8023.0
    xm1 = 6.939
    tempK = temperature + 273.15

    if np.any(tempK <= 0):
        raise ValueError("Temperature must be above absolute zero (-273.15°C).")

    vpres1 = 760 * 10 ** (a1 + b1 / tempK)  # Vapor pressure
    sqrt_argument = xm1 * tempK

    if np.any(sqrt_argument <= 0):
        raise ValueError("Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap


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

nx = nx_P9
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




def process_dataset(data_path, nx, dt):
    """Process dataset to extract and compute required values."""
    max_value_tsurf = []
    max_q = []
    evap_flux_max = []
    max_q_Li_list = []
    C_Li_omp = []


    q_perp_dir = os.path.join(data_path, 'q_perp')
    T_surf_dir = os.path.join(data_path, 'Tsurf_Li')
    q_Li_dir = os.path.join(data_path, 'q_Li_surface')
    C_Li_dir = os.path.join(data_path, 'C_Li_omp')
 

    for i in range(1, nx): 
        filename_tsurf = os.path.join(T_surf_dir, f'T_surfit_{i}.0.csv')
        filename_qsurf = os.path.join(q_perp_dir, f'q_perpit_{i}.0.csv')
        filename_qsurf_Li = os.path.join(q_Li_dir, f'q_Li_surface_{i}.0.csv')
        filename_C_Li = os.path.join(C_Li_dir, f'CLi_prof{i}.0.csv')
       

        max_tsurf = np.nan
        max_q_i = np.nan
        evap_flux = np.nan
        max_q_Li_i = np.nan
        C_Li_i = np.nan
      

        try:
            df_tsurf = pd.read_csv(filename_tsurf)
            max_tsurf = np.max(df_tsurf.values)
        except FileNotFoundError:
            max_tsurf = np.nan  
        
        max_value_tsurf.append(max_tsurf)

        try:
            df_qsurf = pd.read_csv(filename_qsurf)
            max_q_i = np.max(df_qsurf.values)
            
            df_qsurf_Li = pd.read_csv(filename_qsurf_Li)
            max_q_Li_i = np.max(df_qsurf_Li.values)
            
            sep = 9
            df_C_Li = pd.read_csv(filename_C_Li)
            C_Li_i = (df_C_Li.values[sep])
     
            
        except FileNotFoundError:
            max_q_i = np.nan
            max_q_Li_i = np.nan
            C_Li_i = np.nan
            Gamma_Li_i = np.nan
        
        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)
   
    def replace_with_linear_interpolation(arr):
 
        arr = pd.Series(arr)

    # Perform linear interpolation in both directions
        arr_interpolated = arr.interpolate(method='linear', limit_direction='both')

    # Ensure that the interpolation doesn't leave any NaN at the ends
        arr_interpolated = arr_interpolated.fillna(method='bfill').fillna(method='ffill')

    # Return as a numpy array
        return arr_interpolated.to_numpy()
    

    # def replace_with_linear_interpolation(arr):
    #     arr = pd.Series(arr)
    #     return arr.interpolate(method='linear', limit_direction='both').to_numpy()

    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)
 
    for max_tsurf in max_value_tsurf:
        if not np.isnan(max_tsurf):
            try:
                evap_flux = eval_Li_evap_at_T_Cel(max_tsurf)  # Make sure this function is defined
            except Exception as e:
                print(f"Error calculating evaporation flux: {e}. Skipping.")
        else:
            evap_flux = np.nan
        evap_flux_max.append(evap_flux)

    evap_flux_max = replace_with_linear_interpolation(evap_flux_max)

    max_q = np.array(max_q)
    evap_flux_max = np.array(evap_flux_max)
    q_surface = max_q - 2.26e-19 * evap_flux_max

    time_axis = dt * np.arange(1, len(max_q) + 1)

    return max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li_list, C_Li_omp


colors = ['r', 'g', 'k', 'b', 'm', 'y', 'c', 'purple']  # More colors


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

xset = 5
for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li,  C_Li_omp = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, linestyle='-', linewidth =2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])
    ax2.plot(time_axis, max_value_tsurf,  linestyle='-',linewidth =2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])
    ax3.plot(time_axis, C_Li_omp*100 , linestyle='--', linewidth =2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])

ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, xset])
ax1.set_ylim([0, 6])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, xset])
#ax2.set_xlabel('t$_{sim}$ (s)', fontsize=18)
#ax2.legend(loc='best', fontsize=12, ncol=2)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 20])
ax3.set_xlim([0, xset])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
#ax3.legend(loc='best', fontsize=12, ncol=2)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

colors = ['r', 'g', 'k', 'b', 'm', 'y', 'c', 'purple']

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

xset = 5  
user_defined_start_time = 3  

for idx, dataset in enumerate(datasets):
    # Assign a color from the predefined list
    color = colors[idx % len(colors)]  # Cycle through the colors list

    # Process the dataset
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    # Special handling for idx == 3
    if idx == 3:
        time_axis = time_axis + user_defined_start_time
        color = 'k'  # Override the color to black

    # Plot the data
    ax1.plot(time_axis, np.array(max_q_Li) / 1e6, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax2.plot(time_axis, max_value_tsurf, linestyle='-', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)
    ax3.plot(time_axis, C_Li_omp * 100, linestyle='--', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

# Labeling and styling the plots
ax1.set_ylabel('q$_{s}^{max}$ (MW/m$^2$)', fontsize=18)
ax1.set_xlim([0, xset])
ax1.set_ylim([0, 8])
ax1.legend(loc='best', fontsize=12, ncol=2)
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=14)

ax2.set_ylabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
ax2.set_ylim([0, 750])
ax2.set_xlim([0, xset])
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=14)

ax3.set_ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
ax3.set_ylim([0, 20])
ax3.set_xlim([0, xset])
ax3.set_xlabel('t$_{sim}$ (s)', fontsize=18)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()


for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li, C_Li_omp = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )

    # Set color for idx == 3
    if idx == 3:
        color = 'k'  # Black for the 4th dataset
    else:
        color = colors[idx % len(colors)]  # Cycle through colors

    plt.plot(max_value_tsurf, C_Li_omp*100, linestyle='--', linewidth=2, label=f'{dataset["label_tsurf"]}', color=color)

# Labels, Axes, and Grid
plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("C$_{Li-sep}^{omp}$ (%)", fontsize=18)
plt.axhline(3, color='black', linestyle=':', linewidth=2, label='y = 3')  # Reference line at y=3
plt.ylim([0, 15])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)

# Optional: Uncomment the next line to include a legend
# plt.legend()

plt.tight_layout()
plt.show()




def process_Gamma_Li(data_path, nx):

    
    Gamma_Li_dir = os.path.join(data_path, 'n_Li1')
    Gamma_Li = []
    
    for i in range(1, nx):
        filename_Gamma_Li = os.path.join(Gamma_Li_dir, f'n_Li1_{i}.0.csv')
        
        Gamma_Li_i = np.nan  
        
        try:
     
            df_Gamma_Li = pd.read_csv(filename_Gamma_Li)
            Gamma_Li_i = np.max(df_Gamma_Li.values)
        
        except FileNotFoundError:
            print(f"File not found: {filename_Gamma_Li}")
            Gamma_Li_i = np.nan  # If file not found, set to NaN
        
        except Exception as e:
            print(f"Error reading {filename_Gamma_Li}: {e}")
            Gamma_Li_i = np.nan  # If any other error occurs, set to NaN
        
   
        Gamma_Li.append(Gamma_Li_i)

    
    return Gamma_Li



for idx, dataset in enumerate(datasets):
    max_value_tsurf, max_q, evap_flux_max, q_surface, time_axis, max_q_Li,  C_Li_omp  = process_dataset(
        dataset['path'], dataset['nx'], dataset['dt']
    )
    Gamma_Li = process_Gamma_Li(dataset['path'], dataset['nx'])

    plt.plot(max_value_tsurf, Gamma_Li ,  linestyle='-',linewidth =2, label=f'{dataset["label_tsurf"]}', color=colors[idx % len(colors)])

plt.xlabel("T$_{surf}^{max}$ ($^\circ$C)", fontsize=18)
plt.ylabel("$\Gamma_{Li}^{net}$ (/m2s)", fontsize=18)

#plt.ylim([0, 25])
plt.xlim([0, 700])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.show()

y = np.array([-0.04339678, -0.03580569, -0.0287926, -0.02241815, -0.01662467,
                    -0.01138454, -0.00667084, -0.00222074, 0.00074189, 0.00253791, 0.00484185,
                    0.00742937, 0.01039286, 0.0138512, 0.01783639, 0.02244914, 0.02780647,
                    0.03406285, 0.04135004, 0.04984127, 0.05964307, 0.07102367, 0.08421488,
                    0.09925538, 0.10724673])

sxnp_nx = np.array([4.03551331e-08, 4.10567926e-02, 4.11997059e-02, 4.06769506e-02,
       3.91185743e-02, 3.65631064e-02, 3.31731544e-02, 2.93343129e-02,
       2.64046084e-02, 6.62187340e-02, 7.25192941e-02, 6.84025675e-02,
       6.78708695e-02, 6.55699145e-02, 6.60136191e-02, 6.56662642e-02,
       6.58871054e-02, 6.65948323e-02, 6.62120841e-02, 6.74108774e-02,
       6.93008565e-02, 7.01755463e-02, 7.24351892e-02, 7.52719375e-02,
       7.78466636e-02, 7.86308190e-08])


DATA_PATH = r'C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb'


def get_color(value, cmap, norm):
    """Map a value to a color based on colormap and normalization."""
    return cmap(norm(value))

def replace_with_linear_interpolation(arr):
    """Replace NaN values in an array with linear interpolation."""
    arr = pd.Series(arr)
    arr_interpolated = arr.interpolate(method='linear', limit_direction='both')
    return arr_interpolated.fillna(method='bfill').fillna(method='ffill').to_numpy()

def plot_data(ax, x, y, color, label=None):
    """Plot data on a given axis."""
    ax.plot(x, y, marker='*', color=color, label=label)


def process_dataset(data_path, nx):
  
    max_value_tsurf, max_q, max_q_Li_list, C_Li_omp = [], [], [], []

    
    dirs = {
        "q_perp": os.path.join(data_path, 'q_perp'),
        "Tsurf_Li": os.path.join(data_path, 'Tsurf_Li'),
        "q_Li_surface": os.path.join(data_path, 'q_Li_surface'),
        "C_Li_omp": os.path.join(data_path, 'C_Li_omp')
    }

    for i in range(1, nx):
        # File paths
        files = {
            "tsurf": os.path.join(dirs["Tsurf_Li"], f'T_surfit_{i}.0.csv'),
            "qsurf": os.path.join(dirs["q_perp"], f'q_perpit_{i}.0.csv'),
            "qsurf_Li": os.path.join(dirs["q_Li_surface"], f'q_Li_surface_{i}.0.csv'),
            "C_Li": os.path.join(dirs["C_Li_omp"], f'CLi_prof{i}.0.csv')
        }

        
        max_tsurf, max_q_i, max_q_Li_i, C_Li_i = np.nan, np.nan, np.nan, np.nan

        
        try:
            df_tsurf = pd.read_csv(files["tsurf"])
            max_tsurf = np.max(df_tsurf.values)
        except FileNotFoundError:
            pass
        max_value_tsurf.append(max_tsurf)

        
        try:
            df_qsurf = pd.read_csv(files["qsurf"])
            max_q_i = np.max(df_qsurf.values)

            df_qsurf_Li = pd.read_csv(files["qsurf_Li"])
            max_q_Li_i = np.max(df_qsurf_Li.values)

            sep = 8  
            df_C_Li = pd.read_csv(files["C_Li"])
            C_Li_i = df_C_Li.values[sep]
        except FileNotFoundError:
            pass

        max_q.append(max_q_i)
        max_q_Li_list.append(max_q_Li_i)
        C_Li_omp.append(C_Li_i)

   
    max_value_tsurf = replace_with_linear_interpolation(max_value_tsurf)
    max_q = replace_with_linear_interpolation(max_q)
    max_q_Li_list = replace_with_linear_interpolation(max_q_Li_list)
    C_Li_omp = replace_with_linear_interpolation(C_Li_omp)

    return max_value_tsurf, max_q, max_q_Li_list, C_Li_omp


# def process_and_plot(directory, is_2D, file_prefix, y, max_value_tsurf, cmap, norm, ylabel, output_file):

#     fig, ax = plt.subplots()
#     nx = len(max_value_tsurf)

#     for i in range(1, nx):
#         filename_csv = os.path.join(directory, f'{file_prefix}_{i}.0.csv')
#         filename_npy = os.path.join(directory, f'{file_prefix}_{i}.0.csv.npy')

#         # Check if the .csv file exists; if not, check for .npy
#         if os.path.exists(filename_csv):
#             filename = filename_csv
#         elif os.path.exists(filename_npy):
#             filename = filename_npy
#         else:
#             print(f"File not found: {filename_csv} or {filename_npy}")
#             continue

#         try:
           
#             if filename.endswith('.csv') or filename.endswith('.txt'):
#                 data = np.loadtxt(filename)
#             elif filename.endswith('.npy'):
#                 data = np.load(filename)
#             else:
#                 print(f"Unsupported file format: {filename}")
#                 continue

           
#             if is_2D:
#                 numbers = data[105, :-1]  
#             else:
#                 numbers = data.flatten()[:-1]  

          
#             if len(numbers) != len(y):
#                 print(f"Data length mismatch in file {filename}: len(numbers)={len(numbers)}, len(y)={len(y)}")
#                 continue

           
#             color = get_color(max_value_tsurf[i], cmap, norm)

            
#             plot_data(ax, y, numbers, color, label=f'Iteration {i}' if i % 10 == 0 else "")
#         except Exception as e:
#             print(f"Error processing file {filename}: {e}")
#             continue

    
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax)
#     cbar.set_label('T$_{surf}^{max}$ ($^\circ$C)', fontsize=14)

    
#     ax.grid()
#     plt.xlabel("r$_{div}$ - r$_{sep}$ (m)", fontsize=16)
#     plt.ylabel(ylabel, fontsize=20)
#     plt.tick_params(axis='both', labelsize=16)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(output_file, dpi=300)
#     plt.show()

# if __name__ == "__main__":
#     nx = nx_P9  
#     y = np.array([-0.04339678, -0.03580569, -0.0287926, -0.02241815, -0.01662467,
#                         -0.01138454, -0.00667084, -0.00222074, 0.00074189, 0.00253791, 0.00484185,
#                         0.00742937, 0.01039286, 0.0138512, 0.01783639, 0.02244914, 0.02780647,
#                         0.03406285, 0.04135004, 0.04984127, 0.05964307, 0.07102367, 0.08421488,
#                         0.09925538, 0.10724673])

    
#     max_value_tsurf, max_q, max_q_Li_list, C_Li_omp = process_dataset(DATA_PATH, nx)

    
#     cmap = plt.get_cmap('viridis')
#     norm = Normalize(vmin=np.min(max_value_tsurf), vmax=np.max(max_value_tsurf))

#     var = 'n_Li1'
#     name = 'n_Li1'
#     unit = 'n$_e$ (/m$^{-3}$)'
    
    # process_and_plot(
    #     directory=os.path.join(DATA_PATH, var),
    #     is_2D=True,
    #     file_prefix=name,
    #     y=y,
    #     max_value_tsurf=max_value_tsurf,
    #     cmap=cmap,
    #     norm=norm,
    #     ylabel=unit,
    #     output_file="output_plot.png"
    # )