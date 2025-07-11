import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


DATA_PATH = r'C:\UEDGE_run_Shahinul\NSTX_PoP\PePi2.0MW'


DATA_PATHS = [
    r'C:\UEDGE_run_Shahinul\NSTX_PoP\PePi2.0MW'
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

it = np.linspace(1, 1000, 1000)
dt = 5e-3
ix = 5
ixmp = 36 

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
    Li_source_odiv = Li_source_odiv[:-1] + phi_Li

    #if Li_source_odiv[:-1].shape != phi_Li.shape:
    #    print("Shape mismatch between Li_source_odiv and phi_Li.")
     #   continue

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
                numbers = data[52, :-1]
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
    #plt.yscale('log')
    plt.ylim([0, np.max(numbers)*1.1])  
    plt.xlim([-0.06, 0.20])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()



if __name__ == "__main__":
    nx = 1000
    y = np.array([-0.05544489, -0.0507309 , -0.04174753, -0.03358036, -0.02614719,
       -0.01935555, -0.01316453, -0.00755603, -0.00245243,  0.00497426,
        0.012563  ,  0.01795314,  0.02403169,  0.03088529,  0.03865124,
        0.04744072,  0.05723254,  0.06818729,  0.0804908 ,  0.09413599,
        0.10907809,  0.12501805,  0.14181528,  0.15955389,  0.17792796])
    
    sxnp = np.array([1.65294727e-08, 1.68072047e-02, 1.57913285e-02, 1.47307496e-02,
       1.37802350e-02, 1.28710288e-02, 1.19247238e-02, 1.09516290e-02,
       1.02117906e-02, 2.14465429e-02, 1.11099457e-02, 1.26587791e-02,
       1.45917191e-02, 1.66915905e-02, 1.94826593e-02, 2.23582531e-02,
       2.53806793e-02, 2.94667140e-02, 3.39163144e-02, 3.85707117e-02,
       4.34572856e-02, 4.70735328e-02, 5.17684150e-02, 5.64336990e-02,
       5.97825750e-02])
    
    evap = 2.44e-19
    is_evap = True

    max_value_tsurf, max_q, max_q_Li_list, C_Li_omp = process_dataset(DATA_PATH, nx)

    cmap = plt.get_cmap('turbo')
    #norm = Normalize(vmin=np.min(max_value_tsurf), vmax=np.max(max_value_tsurf))
    norm = Normalize(vmin=0*np.min(Li_source_odiv), vmax=np.max(Li_source_odiv))

    var = 'Gamma_net'
    name = 'Gamma_Li_surface'
    unit = '$\Gamma_{Li}^{net}$ (/m$^{-2}$s)'
    
    
    
    var = 'q_perp'
    name = 'q_perpit'
    unit = '$q{\perp}^{odiv}$ (/W$m^{-2}$)'
    is_2D=False,
    
    # var = 'n_e'
    # name = 'n_e_0.0.csv.npy'
    # unit = '$n_{e}^{odiv}$ (/m$^{-3}$)'
    # is_2D=True,
    
    # var = 'T_e'
    # name = 'T_e'
    # unit = '$T_{e}^{odiv}$ (eV)'
    # is_2D=True,
    
    

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