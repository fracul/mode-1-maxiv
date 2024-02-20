# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:12:14 2017
@author: terols
"""

#import init_param as g
import numpy as np
from scipy.constants import c, pi
from math import sqrt, asin, atan2, sin
#asin, sqrt, atan2
from scipy.interpolate import UnivariateSpline
from cmath import rect

import matplotlib.pyplot as plt
import scipy.integrate as integrate

#from scipy.constants import pi
#from math import asin, sqrt, atan2
#
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from plot_functions import*

class TORing:

    def __init__(self,inst):

        self.hNum = inst.nbunch
        self.nHarm = np.array([inst.nharm])
        self.complexFF = 1
        self.C = inst.sring.length
        self.frf = inst.sring.frf
        self.Vrf = inst.vrf
        self.E = inst.sring.energy
        self.alpha = inst.sring.alphac
        self.sigmaE = inst.sring.espread
        self.U0 = inst.sring.eloss

global g

def init_param(inst):
    global g
    g = TORing(inst)

def bunch_length(Ib,Vm,Vh):
    
    natural_length()
    theoretical_max_length()
    
    # Create array for bunch center and length
    center = np.zeros(g.hNum)
    sigmaZ = np.zeros(g.hNum)
    sigmat = np.zeros(g.hNum)
    
    prof = profiles(Ib,Vm,Vh)
    
    z = c*prof[0,:]
    
    for b in range(g.hNum):
    
        (center[b], sigmaZ[b]) = calc_sigma(z,prof[b+1,:]) # Calculate sigma
        sigmat[b] = sigmaZ[b]/c
            
    print('Bunch length from tracking = {0:0.2f} +- {1:0.2f} ps or {2:0.2f} +- {3:0.2f} mm' .format(np.mean(sigmat)*1e12, np.std(sigmat)*1e12, np.mean(sigmaZ)*1000, np.std(sigmaZ)*1000))                           
#          fwhm = calc_fwhm(posz,profile)
    
    
    g.complexFF = 1
    F = form_factor(prof,g.nHarm)
#    print(np.angle(F[1,:])/3)
        
    f = open('bunch_lengths.txt','w')
    f.write(str('{0:>12}{1:>12}{2:>12}\n'.format('Center [m]','Bunch length [m]', 'Harmonic form factor amplitude', 'Harmonic form factor phase')))
    for b in range(g.hNum):
        f.write(str('{0:12.5f}{1:12.5f}'.format(center[b],sigmaZ[b])))
        for i in range(len(g.nHarm)+1):
            f.write(str('{0:12.5f}{1:12.5f}'.format(np.absolute(F[i,b]),np.angle(F[i,b]))))
        f.write('\n')
    f.close()    
    
    
    f = open('profiles_%d.txt' % (np.sum(Ib!=0),),'w')
    for i in range(len(z)):
        f.write(str('{0:12.5f}'.format(z[i])))
    f.write('\n')
    for b in range(g.hNum):
        for i in range(len(z)):
            f.write(str('{0:12.5f}'.format(prof[b][i])))
        f.write('\n')
    f.close()

    #Plot
    #plot_profiles(z,prof)  
    #plot_fields(Ib,Vm,Vh)
    #plot_bunch_length(sigmaZ,sigmat) 

    return F, center/c, sigmaZ/c

def calc_sigma(pos,profile):

#	maxVal = max(profile)
#	minVal = min(profile)
 
     # Rescale
#	profile=(profile-minVal)/maxVal # Why is this needed?
 
     # Mean of probability dsitribution
	aux = profile*pos 
	center = np.sum(aux)/np.sum(profile)
#	print(center)
 
#	print(np.trapz(pos*profile,pos)/np.trapz(profile,pos))

     # Standard deviation of probability dsitribution 
	aux = profile*(pos-center)**2 
	sigma  = sqrt(np.sum(aux)/np.sum(profile))	
 
#	print(sigma)
#	print(sqrt(np.trapz((pos-center)**2*profile,pos)/np.trapz(profile,pos)))

	return (center,sigma)	

def natural_length():
    
    # Required parametes   
    T = g.C/c
    omegarf = 2*pi*g.frf 
    pm0 = pi - asin(g.U0/g.Vrf) # Synchronous phase without HC
    
    # Define longitudinal position
    nPoints = 50001
    interval = 1000e-12
    time = np.linspace(-interval,interval,nPoints)      
    z = c*time
    
    # Natural bunch length    
    pot0 = g.alpha*g.Vrf/(g.E*T*omegarf)*(np.cos(pm0 - omegarf/c*z) - np.cos(pm0)) - g.alpha/(g.E*g.C)*g.U0*z 
    profile0 = np.exp(-pot0/(g.alpha**2*g.sigmaE**2))
                
    area = np.trapz(profile0,z)
    profile0 = 1/area*profile0
    
    (center, sigmaZ0) = calc_sigma(z,profile0) # Calculate sigma
    sigmat0 = sigmaZ0/c
    
    print('Natural bunch length= {0:0.2f} ps or {1:0.2f} mm' .format(sigmat0*1e12,sigmaZ0*1000))
    
def theoretical_max_length():
    
    # Required parametes   
    T = g.C/c
    omegarf = 2*pi*g.frf 

    # Flat potential conditions
        
    VhFP = g.Vrf*sqrt(1.0/g.nHarm[0,0]**2-(g.U0/g.Vrf)**2/(g.nHarm[0,0]**2-1)) # Flat potential harmonic voltage
                
    phFP = atan2(-g.nHarm[0,0]*g.U0/g.Vrf,sqrt((g.nHarm[0,0]**2-1)**2-(g.nHarm[0,0]**2*g.U0/g.Vrf)**2))/g.nHarm[0,0] # Flat potential harmonic phase   
        
    pmFP = pi - asin(g.nHarm[0,0]**2/(g.nHarm[0,0]**2-1.0)*g.U0/g.Vrf) # Perturbed synchronous phase
        
    # Define longitudinal position
    nPoints = 501
    interval = 1000e-12
    time = np.linspace(-interval,interval,nPoints)      
    z = c*time    
    
    # Theoretical max bunch length            
    potFP = g.alpha/(g.E*T*omegarf)*(g.Vrf*(np.cos(pmFP - omegarf/c*z) - np.cos(pmFP)) + VhFP/g.nHarm[0,0]*(np.cos(g.nHarm[0,0]*phFP - g.nHarm[0,0]*omegarf/c*z) - np.cos(g.nHarm[0,0]*phFP))) - g.alpha/(g.E*g.C)*g.U0*z 
            
    profileFP = np.exp(-potFP/(g.alpha**2*g.sigmaE**2))
                
    area = np.trapz(profileFP,z)
    profileFP = 1/area*profileFP
    
    (center, sigmaZFP) = calc_sigma(z,profileFP) # Calculate sigma
    sigmatFP = sigmaZFP/c
     
    print('Flat potential bunch length according to harmonic of 1st harmonic cavity = {0:0.2f} ps or {1:0.2f} mm' .format(sigmatFP*1e12, sigmaZFP*1000))

def profiles(Ib,Vm,Vh):

    global g
    
    # Define longitudinal position
    nPoints = 501
    interval = 1000e-12
    time = np.linspace(-interval,interval,nPoints).reshape(1,-1)
    z = c*time
                      
    # Create array for potentials
    rows = g.hNum+1
    columns = z.shape[1]     
    potentials = np.zeros((rows,columns))
    potentials[0,:] = time 
       
    # Create array for profiles
    rows = g.hNum+1
    columns = z.shape[1]      
    profiles = np.zeros((rows,columns))
    profiles[0,:] = time 
          
#    # Create array for bunch center and length
#    center = np.zeros(g.hNum)
#    sigmaZ = np.zeros(g.hNum)
#    sigmat = np.zeros(g.hNum)    
    
    # Voltage amplitudes and phases   
    Am = np.absolute(Vm)
    pm = -np.angle(Vm)
    Ah = np.absolute(Vh)
    ph = (np.angle(Vh) + pi/2)/g.nHarm

    
#    pref = pi - np.arcsin(g.U0/g.Vrf)
    
            
    # Required parametes   
    T = g.C/c
    omegarf = 2*pi*g.frf
         
    for b in range(g.hNum):
        #print(b)
        
        pmS = pi/2 - pm[b] # Main cavity phase in sine definition

        #raise ValueError
        
#        z = z + pref*c/omegarf
        
#        print(pref*c/omegarf)
        
#        pmS = pref+phi[b]
                                                
#        pot = g.alpha/(g.E*T*omegarf)*(Am[b]*(np.cos(pmS - omegarf/c*z) - np.cos(pmS)) + np.sum(Ah[:,[b]]/g.nHarm*(np.cos(g.nHarm*ph[:,[b]] - g.nHarm*omegarf/c*z) - np.cos(g.nHarm*ph[:,[b]])),axis=0)) - g.alpha/(g.E*g.C)*g.U0*z
        
        pot = g.alpha/(g.E*T*omegarf)*(Am[b]*(np.cos(pmS - omegarf/c*z) - np.cos(pmS)) + np.sum(Ah[:,[b]]/g.nHarm*(np.cos(g.nHarm*ph[:,[b]] - g.nHarm*omegarf/c*z) - np.cos(g.nHarm*ph[:,[b]])),axis=0)) - g.alpha/(g.E*g.C)*g.U0*z
        
#        pot = g.alpha/(g.E*T*omegarf)*(Am[b]*(np.cos(pmS - omegarf/c*z) - np.cos(pmS)) + Ah[0,b]/g.nHarm*(np.cos(g.nHarm*ph[0,b] - g.nHarm*omegarf/c*z)) - np.cos(g.nHarm*ph[0,b])) - g.alpha/(g.E*g.C)*g.U0*z
                                        
#        pot = g.alpha/(g.E*T*omegarf)*(Am[b]*(-np.cos(pmS + omegarf/c*z) + np.cos(pmS)) + np.sum(Ah[:,[b]]/g.nHarm*(-np.cos(g.nHarm*ph[:,[b]] + g.nHarm*omegarf/c*z) + np.cos(g.nHarm*ph[:,[b]])),axis=0)) - g.alpha/(g.E*g.C)*g.U0*z

        profile = np.exp(-pot/(g.alpha**2*g.sigmaE**2))
                
        potentials[b+1,:] = pot
        area = np.trapz(profile,z)
#        area = np.trapz(profile)
        profiles[b+1,:] = 1/area*profile
        
#        (center[b], sigmaZ[b]) = calc_sigma(z,profile) # Calculate sigma
#        sigmat[b] = sigmaZ[b]/c
        
#    plot_profiles(z,profiles)
                
#    print(potentials[1,:])
                         
    return profiles
                       
               
    
def form_factor(profiles, harmonics):
    
    # Complex form factor has not been benchmarked yet!
    
    harmonics = np.insert(harmonics,0,1,0)
           
    F = np.ones((len(harmonics),g.hNum), dtype = complex) 
    
    Ts = (profiles[0,1]-profiles[0,0]) # Time step
        
    for k in range(0,g.hNum):
        time = profiles[0,:]
        profile = profiles[k+1,:]
        
        # Move profiles to center to calculate the phase shift caused by the form factor        
 #       maxIndex = np.argmax(profile)   
#        print(time[maxIndex])
#        print(phi[k]/(2*pi*g.frf))
 #       time = time - time[maxIndex]
                
        # Move profiles according to bunch phase
        
#        time = time + 0.00058/(2*pi*g.frf)
                            
        # Move profiles according to center of mass
        
 #       com = np.trapz(time*profile,time)/np.trapz(profile,time)
        
#        print(com)
        
#        time = time - com
        
        # Move profiles according to reference
#        pref = pi - np.arcsin(g.U0/g.Vrf)
        
#        time = time + 0.02409/(2*pi*g.frf)
        
        
        # Calculate form factor
 #       frHC = g.nHarm*g.frf+g.detuneHC
                        
        FHarm = np.trapz(profile*np.exp(1j*2*pi*harmonics*g.frf*time),time).reshape(-1,1) 
 #       FHarm = np.trapz(profile*np.exp(1j*2*pi*harmonics*frHC*time),time).reshape(-1,1) 
    
        F0 = np.trapz(profile,time)
                                
        if g.complexFF:
            F[:,[k]] = np.conj(FHarm/F0)
 #           F[:,[k]] = (FHarm/F0)
        else:
            F[:,[k]] = np.absolute(FHarm)/F0   
            
#    plt.plot(time,profile)
#    print(np.absolute(F))
#    print(np.angle(F))
                                     
    return F    
