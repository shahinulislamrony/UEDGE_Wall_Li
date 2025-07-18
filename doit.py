import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from uedge import *
import uedge_mvu.plot as mp
import uedge_mvu.utils as mu
import uedge_mvu.analysis as mana
import UEDGE_utils.analysis as ana
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from runcase import *



setGrid()
setPhysics(impFrac=0,fluxLimit=True)
setDChi(kye=1.0, kyi=1.0, difni=0.5,nonuniform = True)
setBoundaryConditions(ncore=6.0e19, pcoree=2.5e6, pcorei=2.5e6, recycp=0.98)
setimpmodel(impmodel=True)

bbb.fphysyrb = 1.0
bbb.cion=3
bbb.oldseec=0
bbb.restart=1

hdf5_restore("./final.hdf5") # converged solution with Li atoms and ions for bbb.isteon=0


bbb.ftol=1e20
bbb.ftol=1e20;bbb.issfon=0; bbb.exmain()

####Save output as ASCII file

###q perp on target
q_perp_div = ana.PsurfOuter()
q_perp_div = (q_perp_div*np.cos(com.angfx[com.nx,:]))/com.sx[com.nx,:]

q_para_odiv = ana.qsurfparOuter()

rrf = ana.getrrf()
PionParallelKE = 0.5*bbb.mi[0]*bbb.up[:,:,0]**2*bbb.fnix[:,:,0]
fthtx = bbb.feex+bbb.feix
qheat = bbb.fetx[com.ixpt2,:] \
             +bbb.fnix[com.ixpt2,:,0]*bbb.ebind*bbb.ev \
             +PionParallelKE[com.ixpt2,:] 

q_para_all = qheat/com.sx/rrf

df=pd.DataFrame(q_para_all)     
df.to_csv('q_para.csv', index=False, header=False)


df=pd.DataFrame(q_para_odiv)     
df.to_csv('q_para_odiv.csv', index=False, header=False)


df=pd.DataFrame(q_perp_div)     
df.to_csv('q_perp_div.csv', index=False, header=False)

df=pd.DataFrame(bbb.te/bbb.ev)     
df.to_csv('Te.csv', index=False, header=False)

df=pd.DataFrame(bbb.ti/bbb.ev)     
df.to_csv('Ti.csv', index=False, header=False)

df=pd.DataFrame(bbb.ne)     
df.to_csv('ne.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,0])     
df.to_csv('ni.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,1])     
df.to_csv('natom.csv', index=False, header=False)

df=pd.DataFrame(bbb.ng[:,:,1])     
df.to_csv('natomLi.csv', index=False, header=False)


df=pd.DataFrame(bbb.sputflxrb[:,1,0])     
df.to_csv('Li_sput_Odiv.csv', index=False, header=False)

df=pd.DataFrame(bbb.sputflxlb[:,1,0])     
df.to_csv('Li_sput_Idiv.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,2])     
df.to_csv('nLi+.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,3])     
df.to_csv('nLi2.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,2])     
df.to_csv('nLi+.csv', index=False, header=False)

df=pd.DataFrame(bbb.ni[:,:,4])     
df.to_csv('nLi3.csv', index=False, header=False)


df=pd.DataFrame(bbb.feex)     
df.to_csv('feex.csv', index=False, header=False)

df=pd.DataFrame(bbb.feix)     
df.to_csv('feix.csv', index=False, header=False)

df=pd.DataFrame(bbb.feiy)     
df.to_csv('feiy.csv', index=False, header=False)

df=pd.DataFrame(bbb.feey)     
df.to_csv('feey.csv', index=False, header=False)

df=pd.DataFrame(bbb.fnix[:,:,0])     
df.to_csv('fnix.csv', index=False, header=False)

df=pd.DataFrame(bbb.fnix[:,:,1])     
df.to_csv('fnix_neu.csv', index=False, header=False)

df=pd.DataFrame(bbb.fniy[:,:,0])     
df.to_csv('fniy.csv', index=False, header=False)

df=pd.DataFrame(bbb.fniy[:,:,1])     
df.to_csv('fniy_neu.csv', index=False, header=False)


df=pd.DataFrame(com.yyc)     
df.to_csv('yyc.csv', index=False, header=False)

df=pd.DataFrame(com.yyrb)     
df.to_csv('yyrb.csv', index=False, header=False)


df=pd.DataFrame(com.yylb)     
df.to_csv('yylb.csv', index=False, header=False)


#df=pd.DataFrame(bbb.pradc)     
#df.to_csv('pradc.csv', index=False, header=False)

df=pd.DataFrame(bbb.prad)     
df.to_csv('prad.csv', index=False, header=False)

#df=pd.DataFrame(bbb.pradiz)     
#df.to_csv('pradiz.csv', index=False, header=False)

#df=pd.DataFrame(bbb.pradrc)     
#df.to_csv('pradrc.csv', index=False, header=False)

#df=pd.DataFrame(bbb.pradfft)     
#df.to_csv('pradfft.csv', index=False, header=False)

#df=pd.DataFrame(bbb.pradimpt)     
#df.to_csv('pradimpt.csv', index=False, header=False)

df=pd.DataFrame(bbb.pri[:,:,0])     
df.to_csv('pri.csv', index=False, header=False)

df=pd.DataFrame(bbb.pre)     
df.to_csv('pre.csv', index=False, header=False)

df=pd.DataFrame(bbb.pr)     
df.to_csv('pr.csv', index=False, header=False)

df=pd.DataFrame(bbb.uup[:,:,0])     
df.to_csv('uup.csv', index=False, header=False)

df=pd.DataFrame(bbb.up[:,:,0])     
df.to_csv('up.csv', index=False, header=False)

df=pd.DataFrame(bbb.up[:,:,2])     
df.to_csv('uLi+.csv', index=False, header=False)

df=pd.DataFrame(bbb.up[:,:,3])     
df.to_csv('uLi2+.csv', index=False, header=False)

df=pd.DataFrame(bbb.up[:,:,4])     
df.to_csv('uLi3+.csv', index=False, header=False)



df=pd.DataFrame(bbb.ni[:,:,0])     
df.to_csv('ni.csv', index=False, header=False)

df=pd.DataFrame(com.rm[:,:,0])     
df.to_csv('rm.csv', index=False, header=False)

df=pd.DataFrame(com.zm[:,:,0])   
df.to_csv('zm.csv', index=False, header=False)

df=pd.DataFrame(com.zm[:,:,1])   
df.to_csv('zm1.csv', index=False, header=False)

df=pd.DataFrame(com.zm[:,:,2])   
df.to_csv('zm2.csv', index=False, header=False)

df=pd.DataFrame(com.zm[:,:,3])   
df.to_csv('zm3.csv', index=False, header=False)

df=pd.DataFrame(com.zm[:,:,4])   
df.to_csv('zm4.csv', index=False, header=False)

df=pd.DataFrame(com.rm[:,:,1])     
df.to_csv('rm1.csv', index=False, header=False)

df=pd.DataFrame(com.rm[:,:,2])     
df.to_csv('rm2.csv', index=False, header=False)

df=pd.DataFrame(com.rm[:,:,3])     
df.to_csv('rm3.csv', index=False, header=False)

df=pd.DataFrame(com.rm[:,:,4])     
df.to_csv('rm4.csv', index=False, header=False)



df=pd.DataFrame(com.sx)     
df.to_csv('sx.csv', index=False, header=False)

df=pd.DataFrame(com.sy)     
df.to_csv('sy.csv', index=False, header=False)

df=pd.DataFrame(com.vol)     
df.to_csv('vol.csv', index=False, header=False)

df=pd.DataFrame(com.angfx)     
df.to_csv('angfx.csv', index=False, header=False)


df=pd.DataFrame(com.dx)     
df.to_csv('dx.csv', index=False, header=False)

df=pd.DataFrame(com.dy)     
df.to_csv('dy.csv', index=False, header=False)

df=pd.DataFrame(com.gxf)     
df.to_csv('gxf.csv', index=False, header=False)

#df=pd.DataFrame(com.xcs)     
#df.to_csv('xcs.csv', index=False, header=False)

#df=pd.DataFrame(com.xfs)     
#df.to_csv('xfs.csv', index=False, header=False)


df=pd.DataFrame(com.yyf)     
df.to_csv('yyf.csv', index=False, header=False)

df=pd.DataFrame(bbb.swbind)     
df.to_csv('swbind.csv', index=False, header=False)

#total pwr flux to outer wall
df=pd.DataFrame(bbb.swallt)     
df.to_csv('swallt.csv', index=False, header=False)

#radiation pwr flux to PF wall
df=pd.DataFrame(bbb.spfwallr)     
df.to_csv('spfwallr.csv', index=False, header=False)

##[W/m**2]#radiation pwr flux to outer wall
df=pd.DataFrame(bbb.swallr)     
df.to_csv('swallr.csv', index=False, header=False)

###ion and elctron pwr flux to outer wall
df=pd.DataFrame(bbb.swalli)     
df.to_csv('swalli.csv', index=False, header=False)

df=pd.DataFrame(bbb.swalle)     
df.to_csv('swalle.csv', index=False, header=False)

df=pd.DataFrame(bbb.erliz)     
df.to_csv('erliz.csv', index=False, header=False)

df=pd.DataFrame(bbb.erlrc)     
df.to_csv('erlrc.csv', index=False, header=False)

#df=pd.DataFrame(bbb.ebind)     
#df.to_csv('ebind.csv', index=False, header=False)

df=pd.DataFrame(bbb.pradhyd)     
df.to_csv('pradhyd.csv', index=False, header=False)

df=pd.DataFrame(bbb.psor[:,:,0])     
df.to_csv('psor.csv', index=False, header=False)

df=pd.DataFrame(bbb.psorrg[:,:,0])     
df.to_csv('psorrg.csv', index=False, header=False)













