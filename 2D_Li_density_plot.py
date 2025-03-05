import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize

paths = [
#r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX-U_g116313\nsep_5.7e19_for_power_scan\nsep_5.7e19_for_power_scan\PePi10MW_check_power_particle\PePi10MW'
r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX-U_g116313\Case_for_02132024_meeting\PePi6MW_Dn0.35Chi0.5_utstep_ph2_data_analysis'
]


com_data = {}
bbb_data = {}


def read_csv(filepath):
    try:
        return pd.read_csv(filepath).values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


for path in paths:
    com_data[path] = {
        'vol': read_csv(os.path.join(path, 'vol.csv')),
        'rm': read_csv(os.path.join(path, 'rm.csv')),
        'rm1': read_csv(os.path.join(path, 'rm1.csv')),
        'rm2': read_csv(os.path.join(path, 'rm2.csv')),
        'rm3': read_csv(os.path.join(path, 'rm3.csv')),
        'rm4': read_csv(os.path.join(path, 'rm4.csv')),
        'parlength': read_csv(os.path.join(path, 'parlength.csv')),
        'conlength': read_csv(os.path.join(path, 'connlen.csv')),
        'yyrb': read_csv(os.path.join(path, 'yyrb.csv')),
        'yylb': read_csv(os.path.join(path, 'yylb.csv')),
        'yyc': read_csv(os.path.join(path, 'yyc.csv')),
        'zm': read_csv(os.path.join(path, 'zm.csv')),
        'zm1': read_csv(os.path.join(path, 'zm1.csv')),
        'zm2': read_csv(os.path.join(path, 'zm2.csv')),
        'zm3': read_csv(os.path.join(path, 'zm3.csv')),
        'zm4': read_csv(os.path.join(path, 'zm4.csv')),
        'angfx': read_csv(os.path.join(path, 'angfx.csv')),
        'sx': read_csv(os.path.join(path, 'sx.csv')),
        'sy': read_csv(os.path.join(path, 'sy.csv')),
        'vxLi1': read_csv(os.path.join(path, 'vxLi1.csv')),
        'vyLi1': read_csv(os.path.join(path, 'vyLi1.csv')),
    }

    bbb_data[path] = {
        'Te': read_csv(os.path.join(path, 'te.csv')),
        'natomLi': read_csv(os.path.join(path, 'natomLi.csv')),
        'nLi+': read_csv(os.path.join(path, 'nLi+.csv')),
        'nLi2': read_csv(os.path.join(path, 'nLi2.csv')),
        'nLi3': read_csv(os.path.join(path, 'nLi3.csv')),
        'Li_sput_odiv': read_csv(os.path.join(path, 'Li_sput_Odiv.csv')),
        'Li_sput_Idiv': read_csv(os.path.join(path, 'Li_sput_Idiv.csv')),
        'Ti': read_csv(os.path.join(path, 'ti.csv')),
        'ne': read_csv(os.path.join(path, 'ne.csv')),
        'ni': read_csv(os.path.join(path, 'ni.csv')),
        'natom': read_csv(os.path.join(path, 'natom.csv')),
        'uup': read_csv(os.path.join(path, 'uup.csv')),
        'up': read_csv(os.path.join(path, 'up.csv')),
        'feex': read_csv(os.path.join(path, 'feex.csv')),
        'feix': read_csv(os.path.join(path, 'feix.csv')),
        'fnix': read_csv(os.path.join(path, 'fnix.csv')),
        'fnix_neu': read_csv(os.path.join(path, 'fnix_neu.csv')),
        'fniy_neu': read_csv(os.path.join(path, 'fniy_neu.csv')),
        'pri': read_csv(os.path.join(path, 'pri.csv')),
        'pre': read_csv(os.path.join(path, 'pre.csv')),
        'pr': read_csv(os.path.join(path, 'pr.csv')),
        'fniy': read_csv(os.path.join(path, 'fniy.csv')),
        'feey': read_csv(os.path.join(path, 'feey.csv')),
        'feiy': read_csv(os.path.join(path, 'feiy.csv')),
        'erlrc': read_csv(os.path.join(path, 'erlrc.csv')),
        'erliz': read_csv(os.path.join(path, 'erliz.csv')),
        'pradhyd': read_csv(os.path.join(path, 'pradhyd.csv')),
        'psor': read_csv(os.path.join(path, 'psor.csv')),
        'psorrg': read_csv(os.path.join(path, 'psorrg.csv')),
        'prad': read_csv(os.path.join(path, 'prad.csv')),
        'qpar': read_csv(os.path.join(path, 'q_para.csv')),
        'qpar_odiv': read_csv(os.path.join(path, 'q_para_odiv.csv')),
        'qperp_odiv': read_csv(os.path.join(path, 'q_perp_div.csv'))
    }
    
    
q_perp_max = []
for i, path in enumerate(paths):
    if 'qperp_odiv' in bbb_data[path]:
        qperp = bbb_data[path]['qperp_odiv']
        zm = com_data[path]['zm']
        rm = com_data[path]['rm']
        rm1 = com_data[path]['rm1']
        rm2 = com_data[path]['rm2']
        rm3 = com_data[path]['rm3']
        rm4 = com_data[path]['rm4']
        zm1 = com_data[path]['zm1']
        zm2 = com_data[path]['zm2']
        zm3 = com_data[path]['zm3']
        zm4 = com_data[path]['zm4']
        vol = com_data[path]['vol']
        up = bbb_data[path]['up']
        vxLi1 = com_data[path]['vxLi1']
        vyLi1 = com_data[path]['vyLi1']
   
      
datasets = [
    
    {
    # 'path': r'C:\Users\islam9\OneDrive - LLNL\Desktop\NSTX-U_g116313\nsep_5.7e19_for_power_scan\nsep_5.7e19_for_power_scan\PePi9MW',
    'path': r'C:\UEDGE_run_Shahinul\NSTX_U\PePi9MW_Dn0.35Chi0.5_dt10ms_fngxrb',
# %%
     'nx':300,

     'dt': 10e-3,
     'label_tsurf': 'P$_{in}$: 8MW'
 },
    
    
    


]




for dataset in datasets:
    data_path = dataset['path'].rstrip("\\")
    nx = dataset['nx']

    n_Li1_dir = os.path.join(data_path, 'n_Li1')
    n_Li2_dir = os.path.join(data_path, 'n_Li2')
    n_Li3_dir = os.path.join(data_path, 'n_Li3')
    n_e_dir = os.path.join(data_path, 'n_e')
    Te_dir = os.path.join(data_path, 'T_e')


    for i in range(1, nx):
        filename_nLi1 = os.path.join(n_Li1_dir, f'n_Li1_{i}.0.csv')
        filename_nLi2 = os.path.join(n_Li2_dir, f'n_Li2_{i}.0.csv')
        filename_nLi3 = os.path.join(n_Li3_dir, f'n_Li3_{i}.0.csv')
        filename_ne = os.path.join(n_e_dir, f'n_e_{i}.0.csv.npy')
        filename_Te = os.path.join(Te_dir, f'T_e_{i}.0.csv')

        if not os.path.exists(filename_nLi1):
            print(f"File not found: {filename_nLi1}")  
            continue

        try:  
            n_Li1_data = np.loadtxt(filename_nLi1)
            n_Li2_data = np.loadtxt(filename_nLi2)
            n_Li3_data = np.loadtxt(filename_nLi3)
            ne_data = np.load(filename_ne)
            Te_data = np.loadtxt(filename_Te)
            C_Li = (n_Li1_data+ n_Li2_data+n_Li3_data)/ne_data
        except Exception as e:
            print(f"Error loading file {filename_nLi1}: {e}") 
            


for i, path in enumerate(paths):
    if 'qperp_odiv' in bbb_data[path]:
        qperp = bbb_data[path]['qperp_odiv']
        zm = com_data[path]['zm']
        rm = com_data[path]['rm']
        rm1 = com_data[path]['rm1']
        rm2 = com_data[path]['rm2']
        rm3 = com_data[path]['rm3']
        rm4 = com_data[path]['rm4']
        zm1 = com_data[path]['zm1']
        zm2 = com_data[path]['zm2']
        zm3 = com_data[path]['zm3']
        zm4 = com_data[path]['zm4']


com_ny = 24
com_nx = 103
zshift = 0.0  # Example vertical shift
com_zm = [zm, zm1, zm2, zm3, zm4]
com_rm = [rm, rm1, rm2, rm3, rm4]

patches = []
for iy in range(com_ny + 2):
    for ix in range(com_nx + 2):
        # Extract corners for the polygon
        rcol = [
            com_rm[1][ix, iy],  # rm1 (left edge)
            com_rm[2][ix, iy],  # rm2 (right edge)
            com_rm[4][ix, iy],  # rm4 (top edge)
            com_rm[3][ix, iy],  # rm3 (bottom edge)
        ]
        zcol = [
            com_zm[1][ix, iy] + zshift,  # zm1
            com_zm[2][ix, iy] + zshift,  # zm2
            com_zm[4][ix, iy] + zshift,  # zm4
            com_zm[3][ix, iy] + zshift,  # zm3
        ]
        # Create polygon and append
        #polygon = Polygon(np.column_stack((rcol, zcol)), True)
        # Create polygon and append
        polygon = Polygon(np.column_stack((rcol, zcol)), closed=True)
        patches.append(polygon)


# Plotting the polygons
# fig, ax = plt.subplots(figsize=(8, 6))
# collection = PatchCollection(patches, cmap='viridis', alpha=0.7)

# ax.add_collection(collection)
# ax.set_xlim(np.min([rm1, rm2, rm3, rm4]), np.max([rm1, rm2, rm3, rm4]))
# ax.set_ylim(np.min([zm1, zm2, zm3, zm4]) + zshift, np.max([zm1, zm2, zm3, zm4]) + zshift)

# # Plot styling
# ax.set_xlabel("Radial Position (r)", fontsize=14)
# ax.set_ylabel("Axial Position (z)", fontsize=14)
# ax.set_title("Polygon Grid Visualization", fontsize=16)

# plt.colorbar(collection, ax=ax, label="Polygon Intensity (Example)")
# plt.grid(True)
# plt.show()



var = C_Li[1:,:]*1e2 #n_Li2_data/1e18 #n_Li1_data
var = Te_data #n_Li1_data
#var = n_Li3_data

cmap = 'turbo' 
title = "n$_{Li3+}$ (10$^{18}$ m$^{-3}$)"
title = "C$_{Li}$ (%)"
#title = "T$_e$ (eV)"

vals=np.zeros((com_nx+2)*(com_ny+2))

for iy in np.arange(0,com_ny+2):
    for ix in np.arange(0,com_nx+2):
        k=ix+(com_nx+2)*iy
        vals[k] = var[ix,iy]


vmax = 100#np.max(vals)
vmin = 0#np.min(vals)
norm = Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(8, 6))
p = PatchCollection(patches, norm=norm, cmap=cmap)
p.set_array(vals)

ax.add_collection(p)
ax.autoscale_view()


cbar = plt.colorbar(p)
cbar.ax.tick_params(labelsize=14)
ax.set_xlabel('R [m]', fontsize=14)
#ax.set_ylabel('Z [m]', fontsize=14)
ax.set_title(title, loc="left", fontsize= 18)
plt.xlim([0.5, 0.9])
plt.ylim([-1.65, -1.2]) 
plt.yticks(fontsize=14) 
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14)  
#plt.yscale('log')
#plt.yticks([])
plt.gca().tick_params(axis='y', which='both', left=False, labelleft=False)
#plt.yticks(fontsize=14) 
plt.grid(True)

plt.plot(rm2[:, 9], zm2[:, 9], '--g', linewidth=4)


plt.show()



