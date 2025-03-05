import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


DATA_PATH = r'C:\UEDGE_run_Shahinul\NSTX_U\PePi8MW_Dn0.35Chi0.5_fneut0.35_check_saveall'


DATA_PATHS = [
    r'C:\UEDGE_run_Shahinul\NSTX_U\PePi8MW_Dn0.35Chi0.5_fneut0.35_check_saveall'
]

marker = 'o' 
line_style = '-.sr' 

def read_csv(filepath):
    try:
        return pd.read_csv(filepath).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

Li_data = {}
for path in DATA_PATHS:
    Li_data[path] = {
        'phi_Li': read_csv(os.path.join(path, 'Phi_Li.csv')),  
        'Li_all': read_csv(os.path.join(path, 'Li_all.csv')) 
    }

it = np.linspace(1, 500, 500)
dt = 10e-3
ix = 5

for path in DATA_PATHS:
    if Li_data[path]['phi_Li'] is None or Li_data[path]['Li_all'] is None:
        print(f"Skipping path {path} due to missing data.")
        continue

    phi_Li = Li_data[path]['phi_Li'].flatten()  # Ensure phi_Li is 1D
    Li_all = Li_data[path]['Li_all']

    if Li_all.shape[1] < 12:
        print(f"Skipping path {path} due to insufficient columns in Li_all.")
        continue

    Li_source_odiv = Li_all[:, 0]
    Li_source_idiv = Li_all[:, 1]
    Li_source_wall = Li_all[:, 3]
    Li_radiation = Li_all[:, 4]
    Li_flux_odiv = Li_all[:, 5]
    Li_flux_wall = Li_all[:, 6]
    Li_flux_idiv = Li_all[:, 7]
    Li_pump_odiv = Li_all[:, 8]
    Li_pump_wall = Li_all[:, 9]
    Li_pump_idiv = Li_all[:, 10]
    Li_ionization = Li_all[:, 11]

    if Li_source_odiv[:-1].shape != phi_Li.shape:
        print("Shape mismatch between Li_source_odiv and phi_Li.")
        continue

    Li_source_odiv = Li_source_odiv[:-1] + phi_Li

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

def process_and_plot(directory, is_2D, is_sxnp, is_evap, sxnp, evap, file_prefix, y, max_value_tsurf, cmap, norm, ylabel, output_file):
    fig, ax = plt.subplots()
    nx = len(max_value_tsurf)

    for i in range(1, nx):
        filename_csv = os.path.join(directory, f'{file_prefix}_{i}.0.csv')
        filename_npy = os.path.join(directory, f'{file_prefix}_{i}.0.csv.npy')

       
        if os.path.exists(filename_csv):
            filename = filename_csv
        elif os.path.exists(filename_npy):
            filename = filename_npy
        else:
            print(f"Warning: Missing both {filename_csv} and {filename_npy}. Skipping iteration {i}.")
            continue

        try:
            
            if filename.endswith('.csv') or filename.endswith('.txt'):
                data = np.loadtxt(filename)
            elif filename.endswith('.npy'):
                data = np.load(filename)
            else:
                print(f"Unsupported file format: {filename}")
                continue

         
            if is_sxnp:
                numbers = data.flatten()[:-1] / sxnp
            elif is_2D:
                numbers = data[105, :-1]
            elif is_evap:
                numbers = data.flatten()[:-1] * evap
            else:
                numbers = data.flatten()[:-1]

           
            if len(numbers) != len(y):
                print(f"Data length mismatch in file {filename}: len(numbers)={len(numbers)}, len(y)={len(y)}")
                continue

            
            #color = get_color(max_value_tsurf[i], cmap, norm)
            color = get_color(Li_source_odiv[i], cmap, norm)

           
            plot_data(ax, y, numbers, color, label=f'Iteration {i}' if i % 10 == 0 else "")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue


    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('T$_{surf}^{max}$ ($^\circ$C)', fontsize=14)
    cbar.set_label('$\phi_{Li}$ (atom/s)', fontsize=14)

   
    ax.grid()
    plt.xlabel("r$_{div}$ - r$_{sep}$ (m)", fontsize=16)
    plt.ylabel(ylabel, fontsize=20)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.ylim([0, np.max(numbers)*1.1])  
    plt.xlim([-0.1, 0.25])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


# Main script
if __name__ == "__main__":
    nx = 500
    y = np.array([-0.09580207, -0.0883297, -0.07360328, -0.05940535, -0.04597069, 
               -0.03356609, -0.02240577, -0.01260851, -0.00402888,  0.00984205, 
                0.03007019,  0.04991498,  0.06846575,  0.08608859,  0.10297708, 
                0.1194364,   0.13546967,  0.15123152,  0.1666732,   0.18186758, 
                0.19708181,  0.21228164,  0.22750439,  0.2429528,   0.25864637])
    sxnp = np.array([4.03551331e-08, 4.10567926e-02, 4.11997059e-02, 4.06769506e-02,
       3.91185743e-02, 3.65631064e-02, 3.31731544e-02, 2.93343129e-02,
       2.64046084e-02, 6.62187340e-02, 7.25192941e-02, 6.84025675e-02,
       6.78708695e-02, 6.55699145e-02, 6.60136191e-02, 6.56662642e-02,
       6.58871054e-02, 6.65948323e-02, 6.62120841e-02, 6.74108774e-02,
       6.93008565e-02, 7.01755463e-02, 7.24351892e-02, 7.52719375e-02,
       7.78466636e-02])
    evap = 2.44e-19
    is_evap = True

    max_value_tsurf, max_q, max_q_Li_list, C_Li_omp = process_dataset(DATA_PATH, nx)

    cmap = plt.get_cmap('turbo')
   # norm = Normalize(vmin=np.min(max_value_tsurf), vmax=np.max(max_value_tsurf))
    norm = Normalize(vmin=0*np.min(Li_source_odiv), vmax=np.max(Li_source_odiv))

    var = 'Gamma_net'
    name = 'Gamma_Li_surface'
    unit = '$\Gamma_{Li}^{net}$ (/m$^{-2}$s)'
    
    # var = 'Gamma_Li'
    # name = 'Total_Li_flux'
    # unit = '$\Gamma_{Li}^{emit}$ (/m$^{-2}$s)'

    process_and_plot(
        directory=os.path.join(DATA_PATH, var),
        is_2D=False,
        is_sxnp=False,
        is_evap=False,  
        sxnp=sxnp,
        evap=evap,
        file_prefix=name,
        y=y,
        max_value_tsurf=max_value_tsurf,
        cmap=cmap,
        norm=norm,
        ylabel=unit,
        output_file="output_plot.png"
    )