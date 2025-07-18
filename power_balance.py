
###Power and particle balance analysis, Shahinul 07/16/2024
from uedge import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as mana
import UEDGE_utils.analysis as ana
import UEDGE_utils.plot as utplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from runcase import *


setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=1.0, kyi=1.0, difni=0.5,nonuniform = True)
setBoundaryConditions(ncore=6.2e19, pcoree=1.5e6, pcorei=1.5e6, recycp=0.98)
setimpmodel(impmodel=True,sput_factor=1)

bbb.fphysyrb = 1.00  ## run UEDGE manually and get a converged solution first 
bbb.cion = 3
bbb.oldseec = 0
bbb.restart = 1
bbb.nusp_imp = 3
bbb.icntnunk = 0

hdf5_restore("./final.hdf5") # converged solution with Li atoms and ions for bbb.isteon=0


bbb.ftol=1e20
bbb.ftol=1e20;bbb.issfon=0; bbb.exmain()



fniy_core = np.sum(np.abs(bbb.fniy[:,0,0]))
fniy_wall = np.sum(np.abs(bbb.fniy[:,com.ny,0]))
fnix_odiv = np.sum(np.abs(bbb.fnix[com.nx,:,0]))
fnix_idiv = np.sum(np.abs(bbb.fnix[0,:,0]))

###Particle flux (1e23 /s)
fniy_core=fniy_core/1e23 # input flux, 
Phi_wall = fniy_wall/1e23
Phi_Odiv = fnix_odiv/1e23 
Phi_Idiv = fnix_idiv/1e23
#Ionization source
Ionization = (np.sum(np.sum(np.abs(bbb.psor[:,:,0]))))/1e23
recombination = (np.sum(np.sum(np.abs(bbb.psorrg[:,:,0]))))/1e23

###particle sink due to pump, check recy,  Gamma_out = R_N * Gamma_in*A;
###so, (1-R_N)*Gamma_in*A is the pumping flux for main ion
###Neutral source: no puff, so, only comes from recycling neutral atom, as no mol, so, the neutral atom density  should be equal to the R_N*Area*Gamma_in
###For example, if recycp = 1, then fnix_odiv = fnix_odiv_neu

Plasma_pump_Odiv = np.sum((1-bbb.recycp[0])*bbb.fnix[com.nx,:,0])
Plasma_pump_Idiv = np.sum((1-bbb.recycp[0])*bbb.fnix[0,:,0])
#Plasma_pump_wall = np.sum((1-bbb.recycw[0])*bbb.fnix[:,com.ny,0])
total = (Plasma_pump_Odiv+ Plasma_pump_Idiv)/1e23

Ion_plus_pump = Ionization-recombination-total

###Now check neutral balance
#print("----neutral and main ion balance------------")
#print("---No gas puff, so, atom density = plasma flux on plate----")
fniy_core_neu = (np.sum((np.abs(bbb.fniy[:,0,1]))))/1e23
fniy_wall_neu = (np.sum((np.abs(bbb.fniy[:,com.ny,1]))))/1e23
fnix_odiv_neu =(np.sum(np.abs(bbb.fnix[com.nx,:,1])))/1e23
fnix_idiv_neu = (np.sum((np.abs(bbb.fnix[0,:,1]))))/1e23
total_neu = fniy_core_neu + fniy_wall_neu + fnix_odiv_neu + fnix_idiv_neu

All_flux = Phi_wall+ Phi_Odiv + Phi_Idiv - fniy_core
Deviation = All_flux - (Ionization + recombination+total+fniy_core)
Error = np.divide(Deviation,All_flux)

print("----neutral and main ion balance------------")
print("---No gas puff, so, atom density = plasma flux on plate----")

print("-----check -----------------")
print("D+ Recycling factor, bbb.recycp           =", bbb.recycp[0])
print("neutral albedo on wall                    =", bbb.recycp[1])
print("neutral albedo at inn div                 =", bbb.albedolb[0])
print("neutral albedo at out div                 =", bbb.albedorb[0])
print("neutral albedo at the wall (ny+1)         =", bbb.albedoo[0])

print("BC for the neutral density at core        =", bbb.isngcore[0]) 

print("----Wall condition ------------")
print(f"neutral atom flux from the wall (1e23/s) = {fniy_wall_neu:.2f}")
print(f"main ion flux on the wall (1e23/s)       = {Phi_wall:.2f}")
print("-----------------------")

print(f"neutral atom flux from the core (1e23/s) = {fniy_core_neu:.2f}")
print(f"main ion flux on the core (1e23/s)       = {fniy_core:.2f}")
print("-----outer divertor--------------")
print(f"neutral atom flux from the Odiv (1e23/s) = {fnix_odiv_neu:.2f}")
print(f"main ion flux on the Odiv (1e23/s)       = {Phi_Odiv:.2f}")
print("----------Inner div---------------")
print(f"neutral atom flux from the Idiv (1e23/s) = {fnix_idiv_neu:.2f}")
print(f"main ion flux on the Idiv (1e23/s)       = {Phi_Idiv:.2f}")
print("-----------------------------")

print(f"flux on the wall (1e23/s) = {Phi_wall:.2f}")
print(f"flux on the Odiv (1e23/s) = {Phi_Odiv:.2f}")
print(f"flux on the Idiv (1e23/s) = {Phi_Idiv:.2f}")
print(f"flux on the core (1e23/s) = {fniy_core:.2f}")
print("----Sum N+E+W-S------")
print(f"Plasma fluxes  (1e23 /s) = {All_flux:.2f}")
print("-----check source and sink terms-------")
print(f"Ionization (1e23/s)           = {Ionization:.2f}")
print(f"Recombination (1e23/s)        = {recombination:.2f}")
print(f"Plasma sink due to (1-R)Gamma = {total:.2f}")
print(f"Ionization -Recombin.(1e23/s) = {Ion_plus_pump:.2f}")


with open('particle.txt', 'w') as f:
    # Write the values to the file with commas
    f.write(f"Phi_wall (1e23/s)      = {Phi_wall}\n")
    f.write(f"Odiv     (1e23/s)      = {Phi_Odiv}\n")
    f.write(f"Indiv    (1e23/s)      = {Phi_Idiv}\n")
    f.write(f"Core     (1e233/s)     = {fniy_core}\n")
    f.write(f"Ionization (1e23/s)    = {Ionization}\n")
    f.write(f"Recombination (1e23/s) = {recombination}")
    



###Li balance

print("------Li balance-----------------")

Plasma_pump_Odiv = np.sum((1-bbb.recycp[1])*bbb.fnix[com.nx,:,2:5])
Plasma_pump_Idiv = np.sum((1-bbb.recycp[1])*bbb.fnix[0,:,2:5])
Plasma_pump_wall = np.sum((1-bbb.recycw[1])*bbb.fniy[:,com.ny,2:5])
print(f'Li pump Odiv {Plasma_pump_Odiv:2f}')
print(f'Li pump Idiv: {Plasma_pump_Idiv:.2f}')
print(f'Li pump wall: {Plasma_pump_wall:.2f}')

total = (Plasma_pump_Odiv+ Plasma_pump_Idiv)/1e23

Ion_plus_pump = Ionization-recombination-total

fniy_core_Li = np.sum(np.abs(bbb.fniy[:,0,2:5]))
fniy_wall_Li = np.sum(np.abs(bbb.fniy[:,com.ny,2:5]))
fnix_odiv_Li = np.sum(np.abs(bbb.fnix[com.nx,:,2:5]))
fnix_idiv_Li = np.sum(np.abs(bbb.fnix[0,:,2:5]))


###Particle flux (1e20 /s)
fniy_core_Li  =fniy_core_Li/1e20 # input flux, 
Phi_wall_Li = fniy_wall_Li/1e20
Phi_Odiv_Li = fnix_odiv_Li/1e20 
Phi_Idiv_Li = fnix_idiv_Li/1e20
#Ionization source

Ionization_Li = (np.sum(np.sum(np.abs(bbb.psor[:,:,2:5]))))/1e20
recombination_Li = (np.sum(np.sum(np.abs(bbb.psorrg[:,:,1]))))/1e20

All_flux = Phi_wall_Li+ Phi_Odiv_Li + Phi_Idiv_Li - fniy_core_Li
Deviation = All_flux - (Ionization_Li - recombination_Li+ fniy_core_Li)
Error = np.divide(Deviation,All_flux)


print(f"Li flux on the wall (1e20/s) = {Phi_wall_Li:.2f}")
print(f"Li flux on the Odiv (1e20/s) = {Phi_Odiv_Li:.2f}")
print(f"Li flux on the Idiv (1e20/s) = {Phi_Idiv_Li:.2f}")
print(f"Li flux on the core (1e20/s) = {fniy_core_Li:.2f}")
print("----Sum N+E+W-S------")
print(f"Li Plasma fluxes  (1e20 /s) = {All_flux:.2f}")
print("-----check source and sink terms-------")
print(f"Ionization (1e20/s)           = {Ionization_Li:.2f}")
print(f"Recombination (1e20/s)        = {recombination_Li:.2f}")

print(f"Ionization -Recombin.(1e20/s) = {Ion_plus_pump:.2f}")

###particle sink due to pump, check recy,  Gamma_out = R_N * Gamma_in*A;
###so, (1-R_N)*Gamma_in*A is the pumping flux for main ion
###Neutral source: no puff, so, only comes from recycling neutral atom, as no mol, so, the neutral atom density  should be equal to the R_N*Area*Gamma_in
###For example, if recycp = 1, then fnix_odiv = fnix_odiv_neu

Plasma_pump_Odiv = np.sum((1-bbb.recycp[0])*bbb.fnix[com.nx,:,0])
Plasma_pump_Idiv = np.sum((1-bbb.recycp[0])*bbb.fnix[0,:,0])
#Plasma_pump_wall = np.sum((1-bbb.recycw[0])*bbb.fnix[:,com.ny,0])
total = (Plasma_pump_Odiv+ Plasma_pump_Idiv)/1e23

Ion_plus_pump = Ionization-recombination-total

##Power balance



Kinetic= 0.5*bbb.mi[0]*bbb.up[:,:,0]**2*bbb.fnix[:,:,0]
    
   

hrad = -bbb.erliz[:,:] - bbb.erlrc[:,:]
pradhyd = np.sum(bbb.pradhyd*com.vol)

Prad_imp = np.sum(bbb.prad[:,:]*com.vol)# Here 0
print("Prad", Prad_imp)

pwrx = bbb.feex+bbb.feix
pwry = bbb.feey+bbb.feiy
pbindx = bbb.fnix[:,:,0]*bbb.ebind*bbb.ev
pbindy = bbb.fniy[:,:,0]*bbb.ebind*bbb.ev
prad_all = np.sum(bbb.erliz+bbb.erlrc) + Prad_imp


bbb.pradpltwl()
print("len(bbb.pwr_plth[:,1])")

Radiation_flux_odiv = np.sum(bbb.pwr_plth[:,1]*com.sx[com.nx+1,:])
Radiation_flux_Idiv = np.sum(bbb.pwr_plth[:,0]*com.sx[0,:])
Radiation_flux_wall = np.sum(bbb.pwr_wallh*com.sy[:,com.ny])
Total_rad_flux = Radiation_flux_odiv+ Radiation_flux_Idiv +Radiation_flux_wall

print(f"Radiation flux on Odiv (MW) = {Radiation_flux_odiv/1e6:.3f}")
print(f"Radiation flux on Idiv (MW) = {Radiation_flux_Idiv/1e6:.3f}")
print(f"Radiation flux on wall (MW) = {Radiation_flux_wall/1e6:.3f}")
print(f"Total Radiation        (MW) = {Total_rad_flux/1e6:.3f}")

pcore = np.sum(pwry[:,0])
pInnerTarget = np.sum((-pwrx-pbindx)[0,:])+np.sum(abs(Kinetic[0,:]))
pOuterTarget = np.sum((pwrx+pbindx)[com.nx,:])+ np.sum(abs(Kinetic[com.nx,:]))
pCFWall = np.sum((pwry+pbindy)[:,com.ny])+ np.sum(abs(Kinetic[:,com.ny]))



P_balance = pInnerTarget + pOuterTarget + pCFWall + prad_all
P_net =np.abs(P_balance - pcore)/pcore                                                  
                                                  
pPFWallInner = np.sum((-pwry-pbindy)[:com.ixpt1[0]+1,0])
pPFWallOuter = np.sum((-pwry-pbindy)[com.ixpt2[0]+1:,0])

##power crosess the separatrix
P_SOL= np.sum(bbb.feey[:,com.iysptrx]+bbb.feiy[:,com.iysptrx])/1e6

print("SOL Power = %.4g MW" % P_SOL)

print("---Power balance analysis------")
print(f"Core power (MW)   = {pcore/1e6:.2f}")
print(f"SOL power  (MW)   = {P_SOL:.2f}")
print(f"Idiv (MW)         = {pInnerTarget/1e6:.2f}")
print(f"Odiv (MW)         = {pOuterTarget/1e6:.2f}")
print(f"Wall (MW)         = {pCFWall/1e6:.2f}")
print(f"Power lossess(MW) = {prad_all/1e6:.2f}")
print(f"P_in -Pout / P_in = {P_net:.2f}")



#P_balance = (Phi_wall + Phi_Odiv +  Phi_Odiv) - (Ionization - recomination) ;

with open('power.txt', 'w') as f:
    f.write(f"pcore (W) = {pcore}\n")
    f.write(f"Odiv (W)  = {pOuterTarget}\n")
    f.write(f"Indiv (W) = {pInnerTarget}\n")
    f.write(f"Wall (W)  ={pCFWall}\n")
    f.write(f"Loss (W)  ={prad_all}\n")
    
###
# Power to inner wall above X-point
nx = com.nx
ny = com.ny
ixmp = bbb.ixmp
fniy = bbb.fniy
fnix = bbb.fnix
erliz = bbb.erliz
erlrc = bbb.erlrc
prad = bbb.prad
vol = com.vol
ev = bbb.ev
allsum = 0.0

ixpt1 = com.ixpt1[0]
ixpt2 = com.ixpt2[0]
iysptrx = com.iysptrx
xoleg = np.s_[ixpt2+1:nx+1]  # x indices of divertor outer leg cells
xileg = np.s_[1:ixpt1+1]     # x indices of divertor inner leg cells
ysol = np.s_[iysptrx+1:ny+1] # y indices outside LCFS
print("ysol", ysol)
print("x odiv", xoleg)
s = np.s_[ixpt1:ixmp]

print("----Check power to each segment-----")
powerToInnerWall = np.sum(pwry[s,ny])/1e6
print("Power to inner wall: %.4g MW" % powerToInnerWall)

# Power to outer wall above X-point 
s = np.s_[ixmp:ixpt2+2]
powerToOuterWall = sum(pwry[s,ny])/1e6
print("Power to outer wall: %.4g MW" % powerToOuterWall)


# Power into outer leg
powerEnteringOuterLeg = sum(pwrx[ixpt2,ysol])/1e6
print("Power entering outer div: %.4g MW" % powerEnteringOuterLeg)


# Power into inner leg
powerEnteringInnerLeg = sum(pwrx[ixpt1,ysol])/1e6
print("Power entering inner div: %.4g MW" % np.abs(powerEnteringInnerLeg))


##power crosess the separatrix
P_SOL= np.sum(bbb.feey[:,com.iysptrx]+bbb.feiy[:,com.iysptrx])/1e6

print("SOL Power = %.4g MW" % P_SOL)

print("-----Power loss in each region----")
Odiv_loss  = powerEnteringOuterLeg - pOuterTarget/1e6
Indiv_loss = np.abs(powerEnteringInnerLeg) - pInnerTarget/1e6


print(f"Power loss in O-div (MW)   =   {Odiv_loss:.2f}")
print(f"Power loss in I-div (MW)   =   {Indiv_loss:.2f}")

