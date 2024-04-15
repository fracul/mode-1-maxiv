import numpy as np
import simplejson

"""
Functions that can be used to determine different cavity settings. The module is generally initialised with a Json
file with all the cavity parameters which are then stored for calls to all functions.
"""

def init(filename):
    global cavparams
    with open(filename,'r') as f: cavparams=simplejson.load(f)
    
def lorentz(a,b,c,pot):

    return a/(1-1j*b*(pot-c))

def getDetune(cavname,potpos,temperature=0):
    
    ptp = cavparams[cavname]['potentiometer']
    cvp = cavparams[cavname]['cavity']
    if temperature==0: temperature = ptp['T']
    detune = (potpos-ptp['fres']-ptp['dfresdT']*(temperature-ptp['T']))*np.sqrt(ptp['Q'])*cvp['fres']/cvp['Q']/2.

    return detune

def getPotpos(cavname,detune,temperature=0):

    ptp = cavparams[cavname]['potentiometer']
    cvp = cavparams[cavname]['cavity']
    if temperature==0: temperature = ptp['T']    
    potpos = detune/np.sqrt(ptp['Q'])/cvp['fres']*cvp['Q']*2+ptp['fres']+ptp['dfresdT']*(temperature-ptp['T'])

    return potpos

def mainDetune(current,voltage,urad,ffact=1):

    mcvp = cavparams['main']['cavity']
    #psi = np.arctan(-2*current*mcvp['Rs']/(voltage)*urad/voltage)
    #detune = mcvp['fres']*psi/mcvp['Q']/3.
    tanpsi = -2*current*ffact*mcvp['Rs']/voltage*np.sqrt(1-(urad/voltage)**2)
    detune = mcvp['fres']*tanpsi/mcvp['Q']/2.

    return detune

def activeHCDetune(current,voltage,phi,ffact=1):
    lcvp = cavparams['LC']['cavity']
    tanpsi = -2*current*ffact*lcvp['Rs']/voltage*np.sin(phi)
    detune = lcvp['fres']*tanpsi/lcvp['Q']/2.

    return detune
    
def detuneFromField(cavname,field,current,blen=200e-12):

    cvp = cavparams[cavname]['cavity']
    formfact = np.exp(-(2*np.pi*cvp['fres']*blen)**2/2.)
    cosdet = field/2./current/formfact/cvp['Rs']

    detune = np.tan(np.arccos(cosdet))*cvp['fres']/2./cvp['Q']

    return detune

def formFactCalc(s,field,phase,nharm):
    
    time = np.arange(-1e-9,1e-9,0.001)
    phi_rf = np.arcsin(s.eloss/s.vrf+field*np.sin(phase)/s.vrf)
    cosphis0 = np.sqrt(1-(eloss/vrf)**2)
    qso2 = s.nbunch*s.alphac*s.vrf*cosphis0/2/np.pi/s.energy
    k = field/s.vrf
    potential = qso**2/s.nbunch/s.alphac/s.espread**2*(np.cos(phi_rf)-np.cos(phi_rf+2*np.pi*s.frf*time)
                                                       +k/nharm*(np.cos(phase)-np.cos(nharm*2*np.pi*s.frf*time+phase))
                                                       -(np.sin(phi_rf)+k*np.sin(phase))*2*np.pi*s.frf*time)

    dist = np.exp(-potential)
    formfact = np.absolute(np.trapz(np.exp(1j*nharm*s.frf*2*np.pi)*dist,x=time)/np.trapz(dist,x=time))

    return formfact

def semiFlat(rs,qfact,nharm,s,formfact=1,k_out=False):

    evby2irs = s.vrf/(2.*s.current*rs*formfact)
    ubyirs = s.eloss/(s.current*rs*formfact)
    ubyv = s.eloss/s.vrf
    a = (1-nharm*nharm)*evby2irs*evby2irs
    b = nharm*nharm+ubyirs
    c = ubyv*ubyv-1

    k = np.sqrt((-b+np.sqrt(b**2-4*a*c))/2./a)
    k_low = np.sqrt((-b-np.sqrt(b**2-4*a*c))/2./a)
    print(k, k_low)
    psi = np.arccos(k*s.vrf/(2.*s.current*formfact*rs))
    detune = np.tan(psi)/2./qfact*nharm*s.frf

    if k_out:
        return detune, k
    return detune

def fromTwoCavities(cavnames,field,phase,current,blen=200e-12,maxdetune=500e3):

    cvps = []
    amps = np.zeros(len(cavnames))
    maxpsis = np.zeros(len(cavnames))
    for i,c in enumerate(cavnames):
        cvp = cavparams[c]['cavity']
        formfact = np.exp(-(2*np.pi*cvp['fres']*blen)**2/2.)
        amp = formfact*cvp['Rs']*2*current
        cvps.append(cvp)
        amps[i] = amp
        maxpsis[i] = np.pi-np.arctan(2*cvp['Q']*maxdetune/cvp['fres'])

    psis = getPsis_scan(amps,field,phase,maxpsis)
    print(psis)
    detunes = -np.tan(psis)*np.array([cavparams[c]['cavity']['fres']/2./cavparams[c]['cavity']['Q'] for c in cavnames])

    print('Fields =', np.absolute(amp*np.cos(psis)))

    return detunes

def getPsis_scan(amps,field,phase,maxpsis,iterations=6):
    
    psis = []
    if amps.shape[0]>2:
        if np.sum(amps[:2])>np.absolute(field/np.cos(phase)):
            psi0 = maxpsis[-1]
            psis = [psi0]
            psis.extend(getPsis_scan(amps[:-1],field-0*np.absolute(amps[-1]*np.cos(psi0)),phase,maxpsis[:-1]))
            psis.reverse()            
        else:
            psi0 = phase
            psis = [psi0]
            psis.extend(getPsis_scan(amps[:-1],field-np.absolute(amps[-1]*np.cos(psi0)),phase,maxpsis[:-1]))
        
        return psis

    delta = 0.001
    downlim0 = phase
    uplim0 = np.pi
    downlim1 = maxpsis[1]
    uplim1 = phase+0.005
    
    for i in xrange(iterations):
        #phase_scan0 = np.arange(phase,np.pi,0.001)
        #phase_scan1 = np.arange(maxpsis[1],phase+0.005,0.001)

        phase_scan0 = np.arange(downlim0,uplim0,delta)
        phase_scan1 = np.arange(downlim1,uplim1,delta)
        
        ph_shpe = (len(phase_scan0),len(phase_scan1))
        phase0 = np.ones((ph_shpe))*phase_scan1
        phase1 = (np.ones((ph_shpe)).T*phase_scan0).T
        amp0 = amps[0]*np.cos(phase0)
        amp1 = amps[1]*np.cos(phase1)
        #raise ValueError
        
        fieldamp = np.sqrt(amp0**2+amp1**2+2*amp0*amp1*np.cos(phase0-phase1))
        fieldphase = np.pi/2.-np.arctan((amp0*np.cos(phase0)+amp1*np.cos(phase1))/(amp0*np.sin(phase0)+amp1*np.sin(phase1)))

        resind = np.argmin(np.absolute(field*np.exp(1j*phase)-fieldamp*np.exp(1j*fieldphase)))

        results = (phase_scan0[resind/ph_shpe[1]], phase_scan1[resind%ph_shpe[1]])
        downlim0 = results[0]-delta/2
        uplim0 = results[0]+delta/2
        downlim1 = max(maxpsis[1],results[1]-delta/2)
        uplim1 = results[1]+delta/2
        delta /= 100

    return results

def getPsis(amps,field,phase,maxpsis):

    from scipy import optimize
    
    psis = []
    if amps.shape[0]>2:
        if np.sum(amps)>field/np.cos(phase):
            psi0 = maxpsis[-1]
        else:
            psi0 = np.pi/2-phase
        psis = [psi0]
        psis.extend(getPsis(amps[:-1],field-amps[-1]*np.cos(psi0),phase,maxpsis[:-1]))
        psis.reverse()
        
        return psis        

    phase_scan = np.arange(np.pi/2.,np.pi,0.01)
    phase_scan0 = np.arange(phase,np.pi,0.01)
    phase_scan1 = np.arange(np.pi/2.,phase,0.01)
    def psi(x0):
        #x1 = np.arccos(np.sqrt(field**2-(amps[0]*np.cos(x0))**2)/amps[1])
        field_chi2 = lambda y: (field**2-(amps[0]*np.cos(x0))**2-(amps[1]*np.cos(y))**2-2*amps[0]*amps[1]*np.cos(x0)*np.cos(y)*np.cos(y-x0))**2
        opt_x1 = optimize.fmin(field_chi2,np.array([x0]))
        x1 = opt_x1[0]
        return np.pi/2.-np.arctan2((amps[0]*np.cos(x0)**2+amps[1]*np.cos(x1)**2),
                                   (amps[0]*np.cos(x0)*np.sin(x0)+amps[1]*np.cos(x1)*np.sin(x1)))
                                   

    chi2 = lambda x: (phase-psi(x[0]))**2
    optres = optimize.fmin(chi2,np.array([phase]))

    x0 = optres[0]
    field_chi2 = lambda y: (field**2-(amps[0]*np.cos(x0))**2-(amps[1]*np.cos(y))**2-2*amps[0]*amps[1]*np.cos(x0)*np.cos(y)*np.cos(y-x0))**2
    opt_x1 = optimize.fmin(field_chi2,np.array([x0]))
    x1 = opt_x1[0]
    
    return np.array([x0,x1])

def minimumCurrentFlatPotential(rs,nharm,s,formfact=1):
    """
    Old approach (attempting to get sinnphi to equal 1, for some reason gave an overestimate):
    const = 2*formfact*rs/s.vrf
    a = const*const
    b = -2*s.eloss/s.vrf*const
    c = (s.eloss/s.vrf)**2-1

    minI = (-b+np.sqrt(b*b-4*a*c))/2./a

    return minI    
    """
    
    a = nharm*nharm*nharm*nharm
    rhs = 4*(1-nharm*nharm)*s.vrf*s.vrf*((s.eloss/s.vrf)**2-1)
    b = 4*nharm*nharm*s.eloss
    c = 4*s.eloss*s.eloss-rhs

    minv = (-b+np.sqrt(b*b-4*a*c))/2./a
    minI = minv/2./rs/formfact

    minI = (-s.eloss+np.sqrt(rhs/4))/rs/formfact/nharm/nharm
    minI = (-s.eloss+np.sqrt((1-nharm*nharm)*(s.eloss**2-s.vrf**2)))/rs/formfact/nharm/nharm
    
    return minI
