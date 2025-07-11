###Shahinul@08082024
"""
a simple script for solving, dT/dt = alpha d**2T/dx**2 + alpha d**2T/dy**2
left boundary is the LM surface, so, q = k dt/dx => q*dx/k
C is considered as a 2nd layer, so, C properties are used
"""
from uedge import *
import pandas as pd
import numpy as np
from uedge.rundt import *
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


q_data = np.load('q_data.npy') # pd.read_csv('./q_data.csv')
# Given data, com.yyrb
x_odiv = com.yyrb
x_data = x_odiv.reshape(-1)

dt = 1e-4    # Time step size (seconds)
t_sim = 5e-3
Nt = int(t_sim/dt)  # 100    # Number of time steps
DEPTH = 0.05
LI_DENSITY  =  535 
heat_vaporization_J_per_mol = 147000  # J/mol Heat of Vaporization 	147 kJ/mol
avogadro_number = 6.022e23
heat_per_atom = heat_vaporization_J_per_mol / avogadro_number

li_coeff = 2.44e-19 
Tinbc = 25 
T_air = 25
h_air = 10.0  # heat transfer coefficient (W/m^2K), vary from 5-25
h=0
I0 = 5.39 * 1.60218e-19      # 1st ionization potential [J]
I1 = 75.6 * 1.60218e-19      # 2nd ionization potential [J]
I2 = 122.4 * 1.60218e-19     # 3rd ionization potential [J]
energy_per_Li3_neutralization = I0 + I1 + I2
energy_per_Li2_neutralization = I0 + I1
energy_per_Li1_neutralization = I0


Lx = 0.05
Ly = x_data[-1] - x_data[0] # Length of the plate in y-direction (meters)
Nx = 91 # len(q_data)*1 #20     # Number of spatial points in x-direction
Ny = len(q_data)      # Number of spatial points in y-direction

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

print('dx is:', dx)
print('Nx is:', Nx)

graphite_x_pos = int(Nx/10) #int(0.01 * scale_factor)

#Li property
thermal_conductivity = 64  # W/(m·K)




try:

    Tin_case = np.load('T_surf2D.npy')
    print("Initial temperature read from npload: ", np.max(Tin_case[:,1]))
    print("Size of the initial conditions is :", len(Tin_case))
except (FileNotFoundError, ValueError) as e:

    Tin_case = 25
    print("Initial Temp is fixed: ", Tin_case)
    print("Error encountered:", e)



def eval_Li_evap_at_T_Cel(temperature):
    a1 = 5.055  
    b1 = -8023.0
    xm1 = 6.939 
    tempK = temperature + 273.15

    if np.any(tempK <= 0):
        raise ValueError("Temperature must be above absolute zero (-273.15°C).")

    vpres1 = 760 * 10**(a1 + b1/tempK)  
    
    sqrt_argument = xm1 * tempK
    if np.any(sqrt_argument <= 0):
        raise ValueError(f"Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap

def flux_Li_Ad_atom(final_temperature, Yield=1e-3, YpsYad=1, eneff=0.9, A=1e-7):
    yadoyps = YpsYad  # ratio of ad-atom to physical sputtering yield
    eneff = eneff  # effective energy (eV), 0.09
    aad = A  # cont
    ylid = Yield
    ylit = 0.001
    ft=0
    kB = 1.3806e-23
    eV = 1.6022e-19
    tempK = final_temperature + 273.15
    fd = bbb.fnix[com.nx,:,0]*np.cos(com.angfx[com.nx,:])/com.sxnp[com.nx,:]
    fneutAd = 1
    #fluxAd = (fneutAd*(fd*ylid + ft*ylit)*yadoyps)/(1 + aad*np.exp(eV*eneff/(kB*tempK)))
    fluxAd = fd*yadoyps/(1 + aad*np.exp(eV*eneff/(kB*tempK)))

    return fluxAd

def solve_heat_equation_2d(Lx, Ly, Nx, Ny, dt, Nt, boundary_conditions, graphite_x_pos):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    lithium_thickness = graphite_x_pos * dx  # Physical thickness of Li
    thickness_mm = lithium_thickness * 1e3
    print("Li thickness (mm):", thickness_mm)

    volume_li = Lx * lithium_thickness * DEPTH
    mass_li = LI_DENSITY * volume_li  # kg
    print("Lithium mass (g):", mass_li * 1e3)

    temp_surf = []
    no_itera = []

    
    T = np.zeros((Ny, Nx))
    try:
        T[:] =Tin_case
        print("Initial read from last stage, Temp is:", np.max(T[:,1:]))
    except (FileNotFoundError, ValueError) as e:
          T[:] = 25 
          print("Error encountered:", e)
    temp_surf = []
    no_itera = []

    for n in range(Nt):
        T_new = T.copy()
        
       
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                T_ij = T[j, i]
                if i < graphite_x_pos:  # Lithium region
                    kappa = Li_thermal_conductivity(T_ij)
                    cp = specific_heat_Cp(T_ij)
                    rho = Li_rho(T_ij)
                else:  # Graphite region
                    kappa = C_thermal_conductivity(T_ij)
                    cp = C_specific_heat(T_ij)
                    rho = graphite_density(T_ij) 

                alpha = 1 / (rho * cp)

                T_ip = T[j, i + 1]
                T_im = T[j, i - 1]
                if i + 1 < graphite_x_pos:
                    kappa_ip = Li_thermal_conductivity(T_ip)
                else:
                    kappa_ip = C_thermal_conductivity(T_ip)
                if i - 1 < graphite_x_pos:
                    kappa_im = Li_thermal_conductivity(T_im)
                else:
                    kappa_im = C_thermal_conductivity(T_im)
                kappa_xp = 0.5 * (kappa + kappa_ip)
                kappa_xm = 0.5 * (kappa + kappa_im)

           
                T_jp = T[j + 1, i]
                T_jm = T[j - 1, i]
            
                if i < graphite_x_pos:
                    kappa_jp = Li_thermal_conductivity(T_jp)
                    kappa_jm = Li_thermal_conductivity(T_jm)
                else:
                    kappa_jp = C_thermal_conductivity(T_jp)
                    kappa_jm = C_thermal_conductivity(T_jm)
                    
                kappa_yp = 0.5 * (kappa + kappa_jp)
                kappa_ym = 0.5 * (kappa + kappa_jm)

                alpha = kappa / (rho * cp)

                term_x = (kappa_xp * (T_ip - T_ij) - kappa_xm * (T_ij - T_im)) / dx**2
                term_y = (kappa_yp * (T_jp - T_ij) - kappa_ym * (T_ij - T_jm)) / dy**2

                T_new[j, i] = T_ij + dt * (term_x + term_y) / (rho * cp)


        evap_flux = eval_Li_evap_at_T_Cel(T_new[:, 1])
        fluxPhysSput = bbb.sputflxrb[:,1,0]/com.sxnp[com.nx,:]
        fluxAd = flux_Li_Ad_atom(T_new[:, 1],Yield=1e-3, YpsYad=1e-3, eneff=0.9, A=1e-7)
        
        Gamma = evap_flux + fluxPhysSput + fluxAd
        Gamma_incident_Li1 = (bbb.fnix[com.nx,:,2]/com.sxnp[com.nx,:])
        Gamma_incident_Li2 = (bbb.fnix[com.nx,:,3]/com.sxnp[com.nx,:])
        Gamma_incident_Li3 = (bbb.fnix[com.nx,:,4]/com.sxnp[com.nx,:])
        Gamma_all =Gamma_incident_Li1+Gamma_incident_Li2+Gamma_incident_Li3
        Gamma_net = Gamma - Gamma_all
        
        L_f_per_mol = 3000.0       # J/mol
        M_li = 6.941e-3            # kg/mol
        L_f_kg = L_f_per_mol / M_li  # J/kg
        L_f = np.where(T_new[:, 1] == 180, L_f_kg, 0.0)
        area = Lx * DEPTH

        if np.any(L_f > 0):
            q_latent = (mass_li * L_f) / (dt * area)
            q_latent = np.clip(q_latent, 0, 2e6)  # Clip to ≤2 MW/m²
        else:
            q_latent = 0.0
            
        q_ion = (energy_per_Li3_neutralization * Gamma_incident_Li1
                 + energy_per_Li2_neutralization * Gamma_incident_Li2
                 + energy_per_Li1_neutralization * Gamma_incident_Li3)    
                    
        heat_flux_Li_surface = q_data - heat_per_atom *(Gamma_net) - q_latent + q_ion
    
        kappa_left = np.vectorize(Li_thermal_conductivity)(T_new[:, 0])

        T_new[:, 0] = T[:,1]+ ((q_data - heat_per_atom *(Gamma_net) - q_latent)*dx)/kappa_left

        heat_flux_Li_surface = heat_flux_Li_surface 

        T_new[:, -1] = T[:,-2] 

        T_new[0, :] = T[1,:] 
        T_new[-1, :] = T[-2,:]
        
      

        max_change = np.max(np.abs(T_new - T))
        
        temp_surface = np.max(T_new[:, 1])  
        temp_surf.append(temp_surface)
       
        no_it = n
        no_itera.append(no_it)
        
        
        T = T_new.copy()  
        ###set convergence condition,
        if max_change < 1e-5:
            print(f"Convergence achieved at time step {n*dt} s with maximum change {max_change:.2e}")
            break
      
    
    return T,  no_itera, temp_surf, Gamma_net, heat_flux_Li_surface


def calculate_stability_factor(kappa, delta_t, rho, C_p, delta_x):
  
    stability_factor = (kappa * delta_t) / (rho * C_p * delta_x**2)
    return stability_factor

def check_stability(stability_factor, criterion=0.5):
    
    return stability_factor <= criterion


# Parameters
Lx = 0.005   # Length of the plate in x-direction (meters)
Ly = x_data[-1] - x_data[0] # Length of the plate in y-direction (meters)
Nx = 10 # len(q_data)*1 #20     # Number of spatial points in x-direction
Ny = len(q_data)      # Number of spatial points in y-direction

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

print('dx is:', dx)
print('Nx is:', Nx)


Lx = 0.005
dx = Lx/(Nx-1)
Lx = 0.05 # https://www.sciencedirect.com/science/article/pii/S0920379613006583 
Nx = int(Lx / dx) + 1
dx = Lx / (Nx - 1)  

print('dx is:', dx)
print('Nx is:', Nx)

##Run for 3 seconds 
dt = 1e-4      # Time step size (seconds)
t_sim = 5e-3
Nt = int(t_sim/dt)  # 100    # Number of time steps

graphite_x_pos = int(Nx/10) #int(0.01 * scale_factor)

rho_C = 1883 # Andrei Khodak
#C_Cp = 710

def Li_rho(T_C):
    """Density as a function of temperature (T in Celsius)."""
    T_K = T_C + 273.15  # Convert Celsius to Kelvin
    return 562 - 0.10 * T_K

def Li_thermal_conductivity(T_C):
    T_K = T_C + 273.15  # Convert Celsius to Kelvin
    if 200 < T_K < 453: 
        return 44 + 0.02019 * T_K + 8037 / T_K
    else:
        return 33.25+ 0.0368 * T_K + 1.096e-5 * (T_K)**2


def specific_heat_Cp (T):
    T_K = T + 273.15  # Convert Celsius to Kelvin
    if 200 < T_K < 453: 
        return (-6.999e8/(T_K)**4 + 1.087e4/(T_K)**2 + 3.039 + 5.605e-6*(T_K)**2)*1e3
    else:
      #  return 21.42 + 0.05230 * T_K + 1.371e-5 * (T_K)**2
        return (1.044e5/(T_K)**2 - 135.1/(T_K) + 4.180)*1e3

#def C_thermal_conductivity(T):   
  
#     return  134 - 0.1074 * T + 3.719e-5 * T**2

###Andrei Khodak data for NSTX-U, 02/19/2025
def C_specific_heat(T):

    Cp = -0.000000 * T**4 + 0.000003 * T**3 - 0.004663 * T**2 + 3.670527 * T + 630.194408
    return Cp

T_exp = np.array([0, 25, 125, 300, 425, 600, 725, 1200, 1225, 1725, 2000])  # °C
K_exp = np.array([105.26, 102.56, 93.02, 80.0, 72.73, 64.52, 59.70, 46.51, 45.98, 37.38, 33.90])  # W/m·K

def quadratic_func(T, a, b, c):
    return a * T**2 + b * T + c

popt, _ = curve_fit(quadratic_func, T_exp, K_exp)
a, b, c = popt

def C_thermal_conductivity(T):
    return a * T**2 + b * T + c

T_graphite = np.array([0, 20, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])  # °C
density_graphite = np.array([1.82, 1.81, 1.79, 1.77, 1.73, 1.69, 1.66, 1.63, 1.60, 1.57, 1.54, 1.50]) * 1e3  # kg/m³

coeffs = np.polyfit(T_graphite, density_graphite, 3)

def graphite_density(T_C):

    a, b, c, d = coeffs
    return a * T_C**3 + b * T_C**2 + c * T_C + d


h = 0  # Convective heat transfer coefficient (W/(m²·K))
T_ambient = 25  # Ambient temperature (°C)


def left_boundary(y,dy):
    # Use q_data directly, since it's in W/m² and is applied to the boundary, q*dy/k
    return np.interp(y, np.linspace(0, Ly, len(q_data)), q_data)*dy / thermal_conductivity  

def right_boundary(y,T_ambient):
    return T_ambient 

def bottom_boundary(x, h, T_ambient, dx):
     return T_ambient + (h * (T_ambient - 0)) * dx / 0.025 
  
def top_boundary(x, h, T_ambient, dx):
    return T_ambient + (h * (T_ambient - 0)) * dx / 0.025  

boundary_conditions = {
    'left': left_boundary,
    'right': right_boundary,
    'bottom': bottom_boundary,
    'top': top_boundary
}

# Interpolate q_data to match the number of y points
Ny_data = len(q_data)
q_data = np.interp(np.linspace(0, 1, Ny), np.linspace(0, 1, Ny_data), q_data)

'''
q_loss = q_Evap + q_rad
q_Evap = 0.136*Gamma_evap/6.023e23
q_rad = 0.046*Area*5.67e-8*(T**4-T_amb**4) , Tamb = 26, emissivity = 0.046, J. Nucl. Mater, 438 2013 422
'''

# Solve the heat equation
T = solve_heat_equation_2d(Lx, Ly, Nx, Ny, dt, Nt, boundary_conditions, graphite_x_pos)

q_perp_div = q_data

     
Gamma_net = T[3]   
q_Li_surf = T[4]  

t_surf = T[0][:,1] ###change to UEDGE grid size
print("size Tsurf", len(t_surf))
print("maximum Tsurf is ", np.max(t_surf))
T2D = T[0]#[:,1:]
np.save('T_surf2D', T2D)
print("initial temperature read from npload and the size is: ", len(T2D))
print("maximum surface temp is :", np.max(T2D[:,1]))

print("size q_perp", len(q_perp_div))

if  len(t_surf)!= len(q_perp_div):
    t_original = np.linspace(0, 1, len(T[0][:,1]))
    t_new =  np.linspace(0, 1, len(q_perp_div))
    T_interpolated = np.interp(t_new,t_original, T[0][:,1])
    #print("size T interpolated", T_interpolated)
    Y_new = Y[:,1]
    x_orig = np.linspace(0, 1, len(Y[:,1]))  
    x_new = np.linspace(0, 1, len(x_data))
    y_data_inter = np.interp(x_new, x_orig, Y_new)
    print("size Y interpolated", y_data_inter)
    plt.figure(figsize=(8, 6))
    plt.plot(Y[:,1], T[0][:,1], color='red', label='Original', marker = 'o')
    plt.plot(y_data_inter, T_interpolated, color='black', label='interpolate', marker = '*')
    plt.legend()
    plt.xlabel("r (m)",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.ylabel("Tsurf (C)",fontsize=16)
    plt.show()

else:
    T_interpolated = t_surf



def eval_Li_evap_at_T_Cel(temperature):
    a1 = 5.055  #  Antioine eq coefficients
    b1 = -8023.0 #
    xm1 = 6.939 
    tempK = temperature  + 273.15
    vpres1 = 760 * 10**(a1 + b1/tempK)  # 1 atm ~ Barr = 760 Torr
    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(xm1 * tempK)

    return fluxEvap


def flux_Li_Phys_Sput(Yield=1e-3, UEDGE= False):
    if UEDGE == False:
    
        kB = 1.3806e-23
        eV = 1.6022e-19
        fd = bbb.fnix[com.nx,:,0]*np.cos(com.angfx[com.nx,:])/com.sxnp[com.nx,:]
        ft = 0
        fli = 0*(ft+fd)*0.65
        ylid = Yield
        ylit = 0.001
        ylili = 0.3
        fneut = 0.35
        fneutAd = 1
        fluxPhysSput = fneut*(fd*ylid + ft*ylit + fli*ylili)
        
        return fluxPhysSput
    else:
        print('UEDGE model for Li physical sputtering')
        fluxPhysSput = bbb.sputflxrb[:,1,0]/com.sxnp[com.nx,:]
        return fluxPhysSput
        
def flux_Li_Ad_atom(final_temperature, Yield=1e-3, YpsYad=1, eneff=0.9, A=1e-7):
    yadoyps = YpsYad  # ratio of ad-atom to physical sputtering yield
    eneff = eneff  # effective energy (eV), 0.09
    aad = A  # cont
    ylid = Yield
    ylit = 0.001
    ft=0
    kB = 1.3806e-23
    eV = 1.6022e-19
    tempK = final_temperature + 273.15
    fd = bbb.fnix[com.nx,:,0]*np.cos(com.angfx[com.nx,:])/com.sxnp[com.nx,:]
    fneutAd = 1
    #fluxAd = (fneutAd*(fd*ylid + ft*ylit)*yadoyps)/(1 + aad*np.exp(eV*eneff/(kB*tempK)))
    fluxAd = fd*yadoyps/(1 + aad*np.exp(eV*eneff/(kB*tempK)))

    return fluxAd

final_temperature =  T_interpolated

###Calculate Li fluxes according to surface temp and sputtering yield
fluxEvap = eval_Li_evap_at_T_Cel(final_temperature)
fluxPhysSput = flux_Li_Phys_Sput(UEDGE= True)
fluxAd= flux_Li_Ad_atom(final_temperature,Yield=1e-3, YpsYad=1e-3, eneff=0.9, A=1e-7)

fluxPhysSput = bbb.sputflxrb[:,1,0]/com.sxnp[com.nx,:]

tot = fluxAd  + fluxEvap

#tot = fluxAd +  fluxEvap
print("length of total flux", len(tot))
x = com.yyrb

df=pd.DataFrame(tot)     
df.to_csv('Gamma_Li_tot.csv', index=False, header=False)

df=pd.DataFrame(final_temperature)     
df.to_csv('Tsurf_profile.csv', index=False, header=False)


print("UEDGE sput flux is used")
print("sum of ad atom", np.sum(fluxAd*com.sxnp[com.nx,:]))
print("sum of Phy Sput", np.sum(fluxPhysSput*com.sxnp[com.nx,:]))
print("sum of Evaporation", np.sum(fluxEvap*com.sxnp[com.nx,:]))

Phys_sput_sum = np.sum(fluxPhysSput*com.sxnp[com.nx,:])
print(f'Sum of Sputtering is    : {Phys_sput_sum}')

Evap_ad_sum = np.sum(tot*com.sxnp[com.nx,:])
print(f'Sum of evap, phy sput, and ad-atom is :{Evap_ad_sum}')


