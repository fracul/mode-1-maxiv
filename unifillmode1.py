#!/sw/easybuild_milan/software/Python/3.10.8-GCCcore-12.2.0/bin/python
"""
Script for running calculations of complex coherent frequencies on a cluster. Takes at least three arguments:
1. Directory for saving output
2. RF Voltage
3. Parking frequency offset a parked main cavity from the -1 revolution harmonic
4, 5, 6... Beam currents
"""

import numpy as np
import time
import os, sys
import scanDetune, utility

sd = utility.StorageRing('./maxiv_delivery201901.inp')
sd.taue = np.inf
taue = 0.025194
sds = utility.StorageRing('./maxiv_delivery201901.inp')
sds.nbunch = 1
sds.alphac *= 176
#cavfile = 'cav_params_5main.json'
cavfile = 'cav_params_2xMAXIV_4main.json'

def prepBoschAnalysis(sringsingle=None,flatvrf=False,**kwargs):

    if isinstance(flatvrf,float):
        sd.vrf = flatvrf
        sds.vrf = flatvrf
        flatvrf = False
    
    ba = scanDetune.BoschAnalysis(sd,cavfile,flatvrf=flatvrf,scaledetune=True,
                                  sringsingle=sringsingle,
                                  **kwargs)

    return ba

def saveTinstOutput(tinst,basedir):

    profzip = np.zeros((tinst.dist.shape[0],tinst.dist.shape[1]+1))
    profzip[:,0] = tinst.time[:,0]
    profzip[:,1:] = np.real(tinst.dist[:])
    np.savetxt(basedir+'/profiles.dat',profzip)

    resarray = np.array(zip(np.arange(tinst.sring.nbunch),
                            tinst.time_off.real,tinst.blen.real,
                            tinst.landau_phasor[:,1].real,
                            tinst.landau_phasor[:,1].imag))
    np.savetxt(basedir+'/time_off_blen_lcphasor.dat',resarray)
    
def saveOutput(bainst,basedir,step=''):
    
    bd = basedir
    if step: step = '_'+str(step)

    if hasattr(bainst,'ruthd1_v'):
        for n in range(len(bainst.current)):
            mode1dat = np.array([bainst.deltaf[n],
                                 bainst.hcfield[n],
                                 bainst.blenbar[n],
                                 bainst.ltunebar[n],
                                 np.absolute(bainst.ffactbar[n]),
                                 np.real(bainst.ruthd[n]),
                                 np.imag(bainst.ruthd[n]),
                                 np.real(bainst.ruthd1[n,:,0]),
                                 np.imag(bainst.ruthd1[n,:,0]),
                                 np.real(bainst.ruthd1[n,:,1]),
                                 np.imag(bainst.ruthd1[n,:,1]),
                                 np.real(bainst.ruthd_v[n]),
                                 np.imag(bainst.ruthd_v[n]),
                                 np.real(bainst.ruthq_v[n]),
                                 np.imag(bainst.ruthq_v[n]),
                                 np.real(bainst.ruthd1_v[n,:,0]),
                                 np.imag(bainst.ruthd1_v[n,:,0]),
                                 np.real(bainst.ruthd1_v[n,:,1]),
                                 np.imag(bainst.ruthd1_v[n,:,1])]).T
            np.savetxt(basedir+'/mode01freqgrate'+step+'_cstep'+str(n)+'.dat',mode1dat)

    if hasattr(bainst,'tinsts') and not bainst.deltinsts:
        for i,tis in enumerate(bainst.tinsts):
            bd = basedir+'/tinst'+str(i)
            for j,t in enumerate(tis):
                bdir = bd+str(j)+step
                os.system('mkdir -p '+bdir)
                saveTinstOutput(t,bdir)

    if not step:
        detarray = np.zeros((len(bainst.current)+1,2))
        detarray[0,0] = bainst.additional_resonator[1]-99.36321e6
        detarray[1:,0] = bainst.vrf
        detarray[1:,1] = bainst.current
        #detarray[2:,0] = bainst.detune
        #detarray[2:,1] = bainst.deltaf[0]
        np.savetxt(basedir+'/README',detarray,fmt=['%.4f','%.4f'])

def main():
    vrf = float(sys.argv[2])
    mainp = float(sys.argv[3])
    current = np.array(sys.argv[4:],dtype=float)
    #detune = np.array(sys.argv[4:],dtype=float)
    dets = 1/np.arange(0.02,0.16,0.002)[:16]**2

    ba = prepBoschAnalysis(current=current,detune=dets,flatvrf=vrf,deltinsts=True,zerofreq=True,
                           transkwargs={'formcalc':'full','fill':np.ones(sd.nbunch)},
                           brentthreshold=1600,use_boschwr=False,omegasapprox=True,additional_resonator=(0*310.4e3,99.36321e6+mainp,3688.))#brentthreshold=900
    ba.getVrfDetune()
    print(ba.vrf)
    #os.system('mkdir '+sys.argv[1])
    saveOutput(ba,sys.argv[1])

    return

    for t in ba.tinsts[0]:
        t.runIterations(50,blenskip=5)
    saveOutput(ba,sys.argv[1],step=1)

    ba = prepBoschAnalysis(current=current,detune=detune,flatvrf=vrf,
                           transkwargs={'formcalc':'full','fill':1.0},
                           brentthreshold=1e6,additional_resonator=(0*310.4e3,99.36321e6+float(sys.argv[3]),3688.))
    ba.getVrfDetune()
    #os.system('mkdir '+sys.argv[1])
    saveOutput(ba,sys.argv[1],step=2)

if __name__=="__main__":
    main()
    

    
