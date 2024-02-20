import numpy as np
from scipy import constants, special
import utility
#from IPython.core.debugger import Tracer

class ImpedanceModel:

    def __init__(self,rhorw=0,radius=0.011,length=528.,rs=0,rfreq=6e9,qfact=1,
                 coat_thickness=0,coat_rhorw=0,filename='',flat=False,**kwargs):
        """
        Create a model of the longitudinal impedance
        Arguments (all optional):

        *rhorw*  - resistivity of the vacuum chamber [Ohms m], default: 0
        *radius* - half-aperture of vacuum chamber [m], default: 0.011
        *length* - length of the vacuum chamber (ring circumference) [m], default: 528
        *rs*     - shunt impedance of a broadband resonator impedance [Ohms], default: 0
        *rfreq*  - resonant frequency of the broadband resonator [Hz], default: 6e9
        *qfact*  - quality factor of the broadband resonator, default: 1
        *coat_thickness* - thickness of the vacuum chamber coating [m], default: 0
        *coat_rhorw*     - resistivity of the vacuum chamber coating [Ohms m], default: 0
        *flat*           - impedance model for flat vacuum chamber (only affects coated chamber model), default: False
        *filename*       - arbitrary impedance read from file, default: ''
        **kwargs         - additional keyword arguments to be passed to numpy.loadtxt to read the impedance file
        """

        self.rhorw = rhorw
        self.radius = radius
        self.length = length

        self.rwconst = np.sqrt(constants.mu_0/8.*self.rhorw*2*np.pi)*self.length/self.radius/np.pi
        self.s0_bane = (2*radius**2/np.sqrt(constants.mu_0/constants.epsilon_0)*rhorw)**(1/3.)
        self.rwconst_bane = 2*self.s0_bane/constants.c/radius**2/4./np.pi/constants.epsilon_0

        self.rfreq = rfreq
        self.bw = self.rfreq/qfact
        self.rs = rs

        self.propconst_coat = 0
        if coat_rhorw>0:
            self.rhorw_coat = coat_rhorw
            self.thick_coat = coat_thickness
            self.propconst_coat = np.sqrt(1j*constants.mu_0/self.rhorw_coat*2*np.pi)
            self.propconst = np.sqrt(1j*constants.mu_0/self.rhorw*2*np.pi)     

        if filename:
            self.fromfile = True
            self.imp = np.loadtxt(filename,unpack=True,**kwargs)
            numfreq = len(self.imp[0])
            self.freq = np.zeros(numfreq*2)
            self.impedance = np.zeros(numfreq*2,complex)
            self.freq[:numfreq] = -self.imp[0][numfreq::-1]*1e9
            self.freq[numfreq:] = self.imp[0]*1e9
            self.impedance[numfreq:] = (self.imp[1]-1j*self.imp[2])
            self.impedance[:numfreq] = (self.imp[1]+1j*self.imp[2])[numfreq::-1]
        else:
            self.fromfile = False

    def bigu(self,frequency):
        """
        M. Ivanyan and V. Tsakanov, Phys Rev ST Accel Beams, 7 114402
        for two metallic layers
        """
        #self.radius**2-2*self.radius/self.propconst_coat/frequency/constants.epsilon_0
        #th_chi3d3 = np.tanh(self.propconst_coat.real*np.sqrt(frequency)*self.thick_coat)
        #th_chi4d4 = np.tanh(self.propconst.real*np.sqrt(frequency)*0.002)
        #alpha = np.sqrt(self.rhorw/self.rhorw_coat)
        #return self.radius**2+2*self.radius/constants.epsilon_0/self.propconst_coat/np.sqrt(frequency)*(1+alpha*th_chi3d3*th_chi4d4)/(th_chi3d3+alpha*th_chi4d4)
    
        chi3 = np.real(self.propconst_coat*np.sqrt(frequency))
        chi4 = np.real(self.propconst*np.sqrt(frequency))
        r = lambda ci,ai,r: special.k0(ci*ai)*special.i0(ci*r)-special.i0(ci*ai)*special.k0(ci*r)
        s = lambda ci,ai,r: -special.k1(ci*ai)*special.i0(ci*r)-special.i1(ci*ai)*special.k0(ci*r)
        rp = lambda ci,ai,r: special.k0(ci*ai)*special.i1(ci*r)+special.i0(ci*ai)*special.k1(ci*r)
        sp = lambda ci,ai,r: -special.k1(ci*ai)*special.i1(ci*r)+special.i1(ci*ai)*special.k1(ci*r)

        a3 = self.radius+self.thick_coat
        const = 2*self.radius/constants.epsilon_0/chi3
        bigfraction = (chi4*r(chi4,np.inf,a3)*sp(chi3,a3,self.radius)
                       -chi3*rp(chi3,a3,self.radius)*rp(chi4,np.inf,a3))\
                       /(chi4*r(chi4,np.inf,a3)*s(chi3,a3,self.radius)-chi3*r(chi3,a3,self.radius)*rp(chi4,np.inf,a3))

        #raise ValueError()
                         
        return self.radius**2-const*bigfraction

    def calcImpedance(self,frequency):
        #return 2/constants.c/self.radius/(np.sqrt(np.absolute(frequency)/self.rhorw)/frequency*(1j+np.sign(frequency))
        #                                  -1j*np.pi*frequency*self.radius/constants.c)

        if self.rs!=0:
            imp_resonant = self.rs/(1+1j/self.bw*(self.rfreq**2-frequency*frequency)/frequency)
        else: imp_resonant = 0
            
        if self.propconst_coat!=0:
            imp_rw = -1j*120./2./np.pi/frequency*constants.c/self.bigu(frequency)
        else:
            imp_rw = self.rwconst*np.sqrt(np.absolute(frequency))*(1-1j*np.sign(frequency))
            
        if self.fromfile:
            imp_file = np.interp(frequency,self.freq,self.impedance.real)+1j*np.interp(frequency,self.freq,self.impedance.imag)
        else: imp_file = 0

        return imp_rw+imp_file+imp_resonant

    def calcImpedanceBane(self,frequency,flat=False,truncate=10.):

        kappa = frequency*2*np.pi/constants.c*self.s0_bane
        if flat:
            x = np.arange(0.01,truncate,0.01)
            integral = 1/np.sinh(x)/(2/(1-1j)*np.outer(1/np.sqrt(kappa),np.cosh(x))-1j*np.outer(kappa,np.sinh(x)/x))
            imp_rw = self.rwconst_bane*np.trapz(integral,x,axis=1)*self.length
        else: imp_rw = self.rwconst_bane/(2/(1.-1j)/np.sqrt(kappa)-1j*kappa/2.)*self.length

        return imp_rw

def calcEffectiveImpedance(s,impmodel,truncate=10000):

    #s = utility.StorageRing(srfile)
    revfreq = constants.c/s.length
    fp = np.arange(-truncate*0+1,truncate+1)*revfreq
    bspect = np.exp(-(2*np.pi*fp*s.blen)**2)
    zeff = np.sum(impmodel.calcImpedance(fp)*bspect)/np.sum(bspect)
    trnge = np.array(np.r_[-0*truncate:0,1:truncate+1],float)
    zoverneff = np.sum(impmodel.calcImpedance(fp)*bspect/trnge)/np.sum(bspect)
    zoverneff1 = np.sum(impmodel.calcImpedance(fp)*np.sqrt(bspect)*fp**2/trnge)/np.sum(fp**2*np.sqrt(bspect))

    return zeff, zoverneff, zoverneff1

def calcLossFactor(s,impmodel,truncate=10000,step=1):

    revfreq = constants.c/s.length
    fp = np.arange(-truncate,truncate+step/2.,step)*revfreq
    bspect = np.exp(-(2*np.pi*fp*s.blen)**2)

    return np.trapz(impmodel.calcImpedance(fp).real*bspect,x=fp)

def getBlen(x,zleff,blen0,sring):

    from scipy import optimize

    a = zleff
    b = blen0
    #zotter = sring.alphac/(sring.ltune*sring.energy)*(sring.length/b/3e8)**3/np.pi*a
    zotter = sring.alphac/(4*np.sqrt(np.pi)*sring.ltune**2*sring.energy)*(sring.length/b/constants.c/2./np.pi)**3*a
    blen_th = np.zeros(len(x))
    for i,c in enumerate(x):
        func = lambda xx: np.absolute(xx**3-xx-zotter*c)
        blen_th[i] = optimize.fmin(func,1.5,disp=0)*b
        
        #if fplot!=None:
        #    fplot.plot(current,blen_th,'x')
    
        #return np.sum((blen-blen_th)**2/blenerr**2)
    return blen_th

def robinsonInstability(tune,revfreq,rfreq,rs,qfactor,energy=1.5e9,alpha=0.003055,ffact=1):
    
    imp_tot = 0
    if not isinstance(ffact,(tuple,list,np.ndarray)):
        ffact = np.ones(len(rs))*ffact
    for r,rf,qf,f in zip(rs,rfreq,qfactor,ffact):
        i = ImpedanceModel(rs=r,qfact=qf,rfreq=rf)
        harm_num = 0*np.round(rf/revfreq)+176.
        onres_imp = i.calcImpedance(harm_num*revfreq)*f
        up_imp = i.calcImpedance((harm_num+tune)*revfreq)*f
        down_imp = i.calcImpedance((harm_num-tune)*revfreq)*f
        #imp_tot += harm_num*(np.real(up_imp)-np.real(down_imp)+1j*np.sqrt((2*np.pi*revfreq*tune)**2+(2*np.imag(onres_imp)-np.imag(up_imp)-np.imag(down_imp))/energy/tune*alpha*revfreq/2*0.25))
        imp_tot += harm_num*(np.real(up_imp)-np.real(down_imp)+1j*(2*np.imag(onres_imp)-np.imag(up_imp)-np.imag(down_imp)))
        #(2*np.imag(onres_imp)-np.imag(up_imp)-np.imag(down_imp)))

    return imp_tot/energy/tune*alpha*revfreq/2.#+1j*imp_tot.imag

def boschInstability(sring,rfreq,rs,qfactor,tinst,mode=1,singlep=False,consistent=False,fullform=False):

    import transient
    from scipy import math, optimize

    if fullform: ffact = np.mean(tinst.formfact,0)
    else: ffact = np.mean(np.absolute(tinst.formfact),0)
    v2 = np.mean(np.absolute(tinst.landau_phasor[:,1]))#/np.absolute(tinst.formfact[:,1]))
    blen = np.mean(tinst.blen.real)
    
    revfreq = sring.frf/sring.nbunch
    nharm = np.round(np.array(rfreq)/sring.frf)
    
    #Make sure that the phases are correct according to Bosch's convention
    #vphi1 = np.pi/2-np.mean(np.angle(tinst.mainrf_phasor))
    vphi1 = tinst.phi_rf
    if fullform: vphi2 = -np.pi-np.mean(np.angle(tinst.landau_phasor[:,1]))
    else: vphi2 = -np.pi-np.mean(np.angle(tinst.landau_phasor[:,1]/tinst.formfact[:,1]))
    phi2 = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf-rfreq[1])/rfreq[1])
    if singlep:
        vphi2 = -np.pi-phi2
        v2 = 2*rs[1]*ffact[1]*sring.current*np.cos(phi2)
        blen = sring.blen
        ffact = np.ones(2)
    
    #omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*revfreq/sring.energy*(ffact[0]*sring.vrf*np.cos(vphi1)+
    #                                                                      nharm[1]*ffact[1]*v2*np.cos(vphi2)))
    vphi1 = np.pi/2.-vphi1
    #vphi2 = vphi2
    if fullform:
        omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*revfreq/sring.energy*(np.absolute(ffact[0])*sring.vrf*np.sin(vphi1+np.angle(ffact[0]))+
                                                                              nharm[1]*np.absolute(ffact[1])*v2*np.sin(vphi2)))
    else:
        omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*revfreq/sring.energy*(ffact[0]*sring.vrf*np.sin(vphi1)+
                                                                              nharm[1]*ffact[1]*v2*np.sin(vphi2)))
    print 'Robinson tune:', omegar/2/np.pi/sring.frf*sring.nbunch

    omegag = 2*np.pi*sring.frf
    commconst = sring.alphac/sring.energy*revfreq
    a = commconst*omegag/2.*(sring.vrf*np.sin(vphi1)+nharm[1]*v2*np.sin(vphi2))
    b = commconst*omegag**2/6.*(sring.vrf*np.cos(vphi1)+nharm[1]**2*v2*np.cos(vphi2))
    c = -commconst*omegag**3/24.*(sring.vrf*np.sin(vphi1)+nharm[1]**3*v2*np.sin(vphi2))    
    landauD = 0.78*sring.alphac**2*sring.espread**2/np.sqrt(2*a)*np.absolute(3*c/2./a-(3*b/2./a)**2)
    #Tracer()()
    
    phi1 = np.arctan(2*qfactor[0]*(sring.frf-rfreq[0])/rfreq[0])

    def modeGrow(omegarg):

        phi1p = np.arctan(2*qfactor[0]*(sring.frf+omegarg/2/np.pi-rfreq[0])/rfreq[0])
        phi1m = np.arctan(2*qfactor[0]*(sring.frf-omegarg/2/np.pi-rfreq[0])/rfreq[0])
    
        phi2p = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf+omegarg/2/np.pi-rfreq[1])/rfreq[1])
        phi2m = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf-omegarg/2/np.pi-rfreq[1])/rfreq[1])
        #Tracer()()
    
        modefact = mode*(sring.frf*2*np.pi*blen)**(2*mode-2)/2.**mode/math.factorial(mode-1)
        boschgrow = mode*np.real(omegar)*np.imag(omegar)+8*sring.current*sring.alphac/sring.energy*modefact*revfreq*(ffact[0]*np.conjugate(ffact[0])*rs[0]*qfactor[0]*np.tan(phi1)*np.cos(phi1p)**2*np.cos(phi1m)**2
                                                                                +nharm[1]**(2*mode-2)*ffact[1]*np.conjugate(ffact[1])*rs[1]*qfactor[1]*np.tan(phi2)*np.cos(phi2p)**2*np.cos(phi2m)**2)
        bigomega = np.sqrt(mode**2*(np.real(omegar)**2-np.imag(omegar)**2)-sring.current*modefact*sring.alphac*sring.frf*2*np.pi/sring.energy*revfreq*(rs[0]*ffact[0]*np.conjugate(ffact[0])*(np.sin(2*phi1p)+np.sin(2*phi1m))
                                                                                                                        +nharm[1]**(2*mode-1)*rs[1]*ffact[1]*np.conjugate(ffact[1])*(np.sin(2*phi2p)+np.sin(2*phi2m))))
        return bigomega, boschgrow

    if consistent:
        tst = lambda x: (modeGrow(x[0])[0]-x[0])**2
        opti = optimize.fmin(tst,np.array([omegar/2.]))
        bigomega, boschgrow = modeGrow(opti[0])
    else:
        bigomega, boschgrow = modeGrow(omegar)        
        
    #currterm = -sring.current*modefact*sring.alphac*sring.frf*2*np.pi/sring.energy*revfreq*(rs[0]*ffact[0]**2*(np.sin(2*phi1p)+np.sin(2*phi1m))
    #+nharm[1]**(2*mode-1)*rs[1]*ffact[1]**2*(np.sin(2*phi2p)+np.sin(2*phi2m)))
    #deltaomega = np.sqrt(np.absolute(currterm))*np.sign(currterm)

    return bigomega+1j*boschgrow, omegar, landauD

def modeGrow(omegarg,sring,rfreq,rs,qfactor,ffact,mode=1,blen=4e-11,omegar=6000):

    from scipy import math

    nharm = np.round(np.array(rfreq)/sring.frf)
    revfreq = sring.frf/sring.nbunch
    
    phi1 = np.arctan(2*qfactor[0]*(sring.frf-rfreq[0])/rfreq[0])
    phi2 = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf-rfreq[1])/rfreq[1])

    phi1p = np.arctan(2*qfactor[0]*(sring.frf+omegarg/2/np.pi-rfreq[0])/rfreq[0])
    phi1m = np.arctan(2*qfactor[0]*(sring.frf-omegarg/2/np.pi-rfreq[0])/rfreq[0])
    
    phi2p = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf+omegarg/2/np.pi-rfreq[1])/rfreq[1])
    phi2m = np.arctan(2*qfactor[1]*(nharm[1]*sring.frf-omegarg/2/np.pi-rfreq[1])/rfreq[1])
    #Tracer()()
    
    modefact = mode*(sring.frf*2*np.pi*blen)**(2*mode-2)/2.**mode/math.factorial(mode-1)
    boschgrow = 8*sring.current*sring.alphac/sring.energy*modefact*revfreq*(ffact[0]**2*rs[0]*qfactor[0]*np.tan(phi1)*np.cos(phi1p)**2*np.cos(phi1m)**2
                                                                            +nharm[1]**(2*mode-2)*ffact[1]**2*rs[1]*qfactor[1]*np.tan(phi2)*np.cos(phi2p)**2*np.cos(phi1m)**2)
    bigomega = np.sqrt((mode*omegar)**2-sring.current*modefact*sring.alphac*sring.frf*2*np.pi/sring.energy*revfreq*(rs[0]*ffact[0]**2*(np.sin(2*phi1p)+np.sin(2*phi1m))
                                                                                                                     +nharm[1]**(2*mode-1)*rs[1]*ffact[1]**2*(np.sin(2*phi2p)+np.sin(2*phi2m))))
    
    return bigomega, boschgrow

def venturiniInstability(revfreq,rfreq,rs,qfactor,energy=1.5e9,alpha=0.003055,blen=200e-12,nbunch=32,current=0.5,omegas0=0):

    imp_tot = 0
    imzeff = 0
    rezeff = 0

    ig = current*2*np.pi*revfreq*nbunch*revfreq*alpha/energy
    k1sigmaz = 2*np.pi*nbunch*revfreq*blen
    ihat = np.sqrt(2)*np.pi*np.pi/special.gamma(0.25)**2*ig*(1-27/4.*k1sigmaz*k1sigmaz)
    
    for r,rf,qf in zip(rs,rfreq,qfactor):
        i = ImpedanceModel(rs=r,qfact=qf,rfreq=rf)
        harm_num = np.round(rf/revfreq)
        up_imp = i.calcImpedance((harm_num+1)*revfreq)
        down_imp = i.calcImpedance((harm_num-1)*revfreq)
        psip = np.arctan(qf*(rf/(harm_num*revfreq+revfreq)-(harm_num*revfreq+revfreq)/rf))
        psim = np.arctan(qf*(rf/(harm_num*revfreq-revfreq)-(harm_num*revfreq-revfreq)/rf))
        imzeff += -3*r/2.*(np.sin(2*psip)+np.sin(2*psim))
        #imzeff += harm_num/nbunch*(np.imag(up_imp)+np.imag(down_imp))
        #rezeff += harm_num/nbunch*(np.real(up_imp)-np.real(down_imp))
        rezeff = 3*r*(np.cos(psip)**2-np.cos(psim)**2)

    omegai = np.sqrt(ihat*imzeff)
    omegar = ihat*rezeff/2./omegai

    omegai_off = np.sqrt(-ig*imzeff+omegas0*omegas0)
    omegar_off = ig/2./omegai_off*rezeff

    return omegar+1j*omegai, omegar_off+1j*omegai_off

def heInstability(tinst,sdmulti=None):

    if sdmulti==None:
        sdmulti = tinst.sring
    k = np.arange(0.,sdmulti.nbunch)
    a = np.exp(-tinst.alpha[1]/tinst.revfreq)
    theta = (tinst.omegar[1]%(2*np.pi*sdmulti.frf))/tinst.revfreq
    denom = np.sqrt(1+a*a-2*a*np.cos(theta))
    deltatheta = np.arcsin((1-a)/denom*np.cos(theta/2.))
    
    epsilon_k = np.sin(np.pi/2.)-np.sin(np.pi/2.-(k+1)*2*np.pi/sdmulti.nbunch)
    theta_k = theta/2.+deltatheta-k/sdmulti.nbunch*theta
    dk = np.exp(-tinst.alpha[1]/tinst.revfreq*k/sdmulti.nbunch)

    f_thetam = np.sum(epsilon_k*dk*np.cos(theta_k))/sdmulti.nbunch/denom
    psis = np.pi/2.-np.mean(np.angle(tinst.mainrf_phasor))+np.mean(np.angle(tinst.formfact[:,1]))/tinst.nharm[1]
    const = 2*np.pi*tinst.nharm[1]**2*np.mean(np.absolute(tinst.formfact[:,1]))*sdmulti.nbunch*sdmulti.current*tinst.rs[1]/tinst.qfact[1]/sdmulti.vrf/np.absolute(np.cos(psis))

    return const*f_thetam



                

    
