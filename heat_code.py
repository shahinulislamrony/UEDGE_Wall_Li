import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def eval_Li_evap_at_T_Cel(temperature):
    a1 = 5.055  
    b1 = -8023.0
    xm1 = 6.939 
    tempK = temperature + 273.15

    if np.any(tempK <= 0):
        raise ValueError("Temperature must be above absolute zero (-273.15°C).")

    vpres1 = 760 * 10**(a1 + b1 / tempK)  

    sqrt_argument = xm1 * tempK
    if np.any(sqrt_argument <= 0):
        raise ValueError(f"Invalid value for sqrt: xm1 * tempK has non-positive values.")

    fluxEvap = 1e4 * 3.513e22 * vpres1 / np.sqrt(sqrt_argument)  # Evaporation flux
    return fluxEvap


def flux_Li_Phys_Sput(Yield=1e-3, UEDGE=False, bbb=None, com=None):
    if not UEDGE:
        kB = 1.3806e-23
        eV = 1.6022e-19
        fd = bbb.fnix[com.nx, :, 0] * np.cos(com.angfx[com.nx, :]) / com.sxnp[com.nx, :]
        ft = 0
        fli = 0 * (ft + fd) * 0.65
        ylid = Yield
        ylit = 0.001
        ylili = 0.3
        fneut = 0.35
        fneutAd = 1
        fluxPhysSput = fneut * (fd * ylid + ft * ylit + fli * ylili)
        return fluxPhysSput
    else:
        print('UEDGE model for Li physical sputtering')
        fluxPhysSput = bbb.sputflxrb[:, 1, 0] / com.sxnp[com.nx, :]
        return fluxPhysSput


def flux_Li_Ad_atom(final_temperature, Yield=1e-3, YpsYad=1, eneff=0.9, A=1e-7, bbb=None, com=None):
    yadoyps = YpsYad  # ratio of ad-atom to physical sputtering yield
    eneff = eneff  # effective energy (eV), 0.9
    aad = A  # constant
    ylid = Yield
    ylit = 0.001
    ft = 0
    kB = 1.3806e-23
    eV = 1.6022e-19
    tempK = final_temperature + 273.15
    fd = bbb.fnix[com.nx, :, 0] * np.cos(com.angfx[com.nx, :]) / com.sxnp[com.nx, :]
    fneutAd = 1
    fluxAd = fd * yadoyps / (1 + aad * np.exp(eV * eneff / (kB * tempK)))

    return fluxAd


def solve_heat_equation_2d(
    Lx, Ly, Nx, Ny, dt, Nt,
    q_data, Tin_case,
    DEPTH, LI_DENSITY,
    heat_per_atom, graphite_x_pos,
    eval_Li_evap_at_T_Cel, flux_Li_Ad_atom,
    bbb, com,
    energy_per_Li1_neutralization,
    energy_per_Li2_neutralization,
    energy_per_Li3_neutralization,
    Li_thermal_conductivity,
    C_thermal_conductivity,
    specific_heat_Cp,
    C_specific_heat,
    Li_rho,
    graphite_density
):

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    lithium_thickness = graphite_x_pos * dx  # Physical thickness of Li
    thickness_mm = lithium_thickness * 1e3
    print("Li thickness (mm):", thickness_mm)

    volume_li = Lx * lithium_thickness * DEPTH
    mass_li = LI_DENSITY * volume_li  # kg
    print("Lithium mass (g):", mass_li * 1e3)
    print("time step is:", dt)

    temp_surf = []
    no_itera = []

    T = np.zeros((Ny, Nx))
    try:
        T[:] = Tin_case
        print("Initial read from last stage, Temp max:", np.max(T[:, 1]))
    except (FileNotFoundError, ValueError) as e:
        T[:] = 25
        print("Error encountered, setting initial temp to 25°C:", e)

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

        # Calculate evaporation and sputtering fluxes
        evap_flux = eval_Li_evap_at_T_Cel(T_new[:, 1])
        fluxPhysSput = bbb.sputflxrb[:, 1, 0] / com.sxnp[com.nx, :]
        fluxAd = flux_Li_Ad_atom(T_new[:, 1], Yield=1e-3, YpsYad=1e-3, eneff=0.9, A=1e-7, bbb=bbb, com=com)

        Gamma = evap_flux + fluxPhysSput + fluxAd

        Gamma_incident_Li1 = bbb.fnix[com.nx, :, 2] / com.sxnp[com.nx, :]
        Gamma_incident_Li2 = bbb.fnix[com.nx, :, 3] / com.sxnp[com.nx, :]
        Gamma_incident_Li3 = bbb.fnix[com.nx, :, 4] / com.sxnp[com.nx, :]

        Gamma_all = Gamma_incident_Li1 + Gamma_incident_Li2 + Gamma_incident_Li3
        Gamma_net = Gamma - Gamma_all

        # Latent heat calculations
        L_f_per_mol = 3000.0       # J/mol (your value)
        M_li = 6.941e-3            # kg/mol
        L_f_kg = L_f_per_mol / M_li  # J/kg

        latent_T_range = (T_new[:, 1] > 178) & (T_new[:, 1] < 182)
        L_f = np.zeros_like(T_new[:, 1])
        L_f[latent_T_range] = L_f_kg

        area = Lx * DEPTH

        if np.any(L_f > 0):
            q_latent = (mass_li * L_f) / (dt * area)
            q_latent = np.clip(q_latent, 0, 2e6)  # Clip max 2 MW/m²
        else:
            q_latent = np.zeros_like(T_new[:, 1])

        q_ion = (energy_per_Li1_neutralization * Gamma_incident_Li1
                 + energy_per_Li2_neutralization * Gamma_incident_Li2
                 + energy_per_Li3_neutralization * Gamma_incident_Li3)

        heat_flux_Li_surface = q_data - heat_per_atom * Gamma_net - q_latent + q_ion

        # Boundary condition at left edge (Li surface)
        kappa_left = np.vectorize(Li_thermal_conductivity)(T_new[:, 0])
        T_new[:, 0] = T[:, 1] + (heat_flux_Li_surface * dx) / kappa_left

        # Other boundary conditions: Neumann (zero-gradient)
        T_new[:, -1] = T[:, -2]  # right edge
        T_new[0, :] = T[1, :]    # bottom edge
        T_new[-1, :] = T[-2, :]  # top edge

        max_change = np.max(np.abs(T_new - T))
        temp_surface = np.max(T_new[:, 1])
        temp_surf.append(temp_surface)
        no_itera.append(n)

        T = T_new.copy()

        if max_change < 1e-5:
            print(f"Convergence achieved at time step {n*dt:.5f} s with max change {max_change:.2e}")
            break

    return T, no_itera, temp_surf, Gamma_net, heat_flux_Li_surface, q_ion


def run_heat_simulation( bbb, com, dt=5e-3):
    q_data = np.load('q_data.npy')
    x_odiv = com.yyrb
    x_data = x_odiv.reshape(-1)

    t_sim = dt
    Nt = int(t_sim / dt)
    DEPTH = 0.05
    LI_DENSITY = 535  # kg/m³ for Li

    heat_vaporization_J_per_mol = 147000
    avogadro_number = 6.022e23
    heat_per_atom = heat_vaporization_J_per_mol / avogadro_number

    I0 = 5.39 * 1.60218e-19
    I1 = 75.6 * 1.60218e-19
    I2 = 122.4 * 1.60218e-19
    energy_per_Li3_neutralization = I0 + I1 + I2
    energy_per_Li2_neutralization = I0 + I1
    energy_per_Li1_neutralization = I0

    Lx = 0.05
    Ly = x_data[-1] - x_data[0]
    Nx = 91
    Ny = len(q_data)

    graphite_x_pos = int(Nx / 10)

    try:
        Tin_case = np.load('T_surf2D.npy')
        print("Initial temperature read from np.load: max =", np.max(Tin_case[:, 1]))
    except (FileNotFoundError, ValueError) as e:
        Tin_case = 25
        print("Initial Temp is fixed to:", Tin_case)

    # Material property functions
    def Li_rho(T_C):
        T_K = T_C + 273.15
        return 562 - 0.10 * T_K

    def Li_thermal_conductivity(T_C):
        T_K = T_C + 273.15
        if 200 < T_K < 453:
            return 44 + 0.02019 * T_K + 8037 / T_K
        else:
            return 33.25 + 0.0368 * T_K + 1.096e-5 * T_K**2

    def specific_heat_Cp(T):
        T_K = T + 273.15
        if 200 < T_K < 453:
            return (-6.999e8 / T_K**4 + 1.087e4 / T_K**2 + 3.039 + 5.605e-6 * T_K**2) * 1e3
        else:
            return (1.044e5 / T_K**2 - 135.1 / T_K + 4.180) * 1e3

    def C_specific_heat(T):
        return -0.000000 * T**4 + 0.000003 * T**3 - 0.004663 * T**2 + 3.670527 * T + 630.194408

    T_exp = np.array([0, 25, 125, 300, 425, 600, 725, 1200, 1225, 1725, 2000])
    K_exp = np.array([105.26, 102.56, 93.02, 80.0, 72.73, 64.52, 59.70, 46.51, 45.98, 37.38, 33.90])
    popt, _ = curve_fit(lambda T, a, b, c: a * T**2 + b * T + c, T_exp, K_exp)
    a, b, c = popt

    def C_thermal_conductivity(T):
        return a * T**2 + b * T + c

    T_graphite = np.array([0, 20, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800])
    density_graphite = np.array([1.82, 1.81, 1.79, 1.77, 1.73, 1.69, 1.66, 1.63, 1.60, 1.57, 1.54, 1.50]) * 1e3
    coeffs = np.polyfit(T_graphite, density_graphite, 3)

    def graphite_density(T_C):
        a_, b_, c_, d_ = coeffs
        return a_ * T_C**3 + b_ * T_C**2 + c_ * T_C + d_

    T, no_itera, temp_surf, Gamma_net, heat_flux_Li_surface, q_ion = solve_heat_equation_2d(
        Lx, Ly, Nx, Ny, dt, Nt,
        q_data, Tin_case,
        DEPTH, LI_DENSITY,
        heat_per_atom, graphite_x_pos,
        eval_Li_evap_at_T_Cel, flux_Li_Ad_atom,
        bbb, com,
        energy_per_Li1_neutralization,
        energy_per_Li2_neutralization,
        energy_per_Li3_neutralization,
        Li_thermal_conductivity,
        C_thermal_conductivity,
        specific_heat_Cp,
        C_specific_heat,
        Li_rho,
        graphite_density
    )

    return T, no_itera, temp_surf, Gamma_net, heat_flux_Li_surface, q_ion


# === Run simulation ===
if __name__ == "__main__":
    #dt = 1e-4  # time step size (s)
    # Assume bbb and com are imported/provided from your UEDGE environment
    from uedge import bbb, com  # <-- adjust imports as per your setup

    T2D, no_it, temp_surf_history, Gamma_net, q_Li_surf, q_ion = run_heat_simulation(bbb, com,dt)

    np.save('T_surf2D.npy', T2D)
    print("Saved T_surf2D.npy. Shape:", T2D.shape)
    print("Max surface temperature:", np.max(T2D[:, 1]))

    t_surf = T2D[:, 1]
    q_perp_div = np.load('q_data.npy')

    if len(t_surf) != len(q_perp_div):
        t_original = np.linspace(0, 1, len(t_surf))
        t_new = np.linspace(0, 1, len(q_perp_div))
        T_interpolated = np.interp(t_new, t_original, t_surf)

        Y = np.load('Y_positions.npy')  # ensure this file exists in your setup
        x_data = com.yyrb.reshape(-1)
        x_orig = np.linspace(0, 1, len(Y[:, 1]))
        x_new = np.linspace(0, 1, len(x_data))
        y_data_inter = np.interp(x_new, x_orig, Y[:, 1])

        plt.figure(figsize=(8, 6))
        plt.plot(Y[:, 1], t_surf, 'ro-', label='Original')
        plt.plot(y_data_inter, T_interpolated, 'k*-', label='Interpolated')
        plt.xlabel("r (m)", fontsize=16)
        plt.ylabel("Tsurf (°C)", fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        T_interpolated = t_surf

    final_temperature = T_interpolated
    fluxEvap = eval_Li_evap_at_T_Cel(final_temperature)
    tot = fluxEvap + fluxAd

    print("Evaporation flux range:", np.min(fluxEvap), np.max(fluxEvap))

    # Save CSV files
    pd.DataFrame(Gamma_net).to_csv('Gamma_Li_net.csv', index=False, header=False)
    pd.DataFrame(q_Li_surf).to_csv('q_Li_surface.csv', index=False, header=False)
    pd.DataFrame(q_ion).to_csv('q_ion_heating.csv', index=False, header=False)



    # === Print diagnostics ===
  #  area_factor = com.sxnp[com.nx, :]
   # print("UEDGE sput flux is used")
   # print(f"Sum of ad-atom flux         : {np.sum(fluxAd * area_factor):.3e}")
   # print(f"Sum of Physical Sputtering  : {np.sum(fluxPhysSput * area_factor):.3e}")
   # print(f"Sum of Evaporation          : {np.sum(fluxEvap * area_factor):.3e}")
   # print(f"Total Li flux (Ad + Evap)   : {np.sum(tot * area_factor):.3e}")




