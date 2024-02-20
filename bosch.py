import numpy as np
import transient
from scipy import math, optimize
from scipy.special import gamma
#from IPython.core.debugger import Tracer

class BoschInstability:
    """
    Class for calculating the growth rates and coherent frequencies of the Robinson modes using the approach of Bosch.
    """

    def __init__(self,sring,rfreq,rs,qfactor,tinst=None,vector=False,use_ltune=True):
        """
        Class initialisation:
        *sring*  - A utility.StorageRing instance
        *rfreq*  - A list of lengthe 2 with the resonant frequencies of the main and harmonic cavities
        *rs*     - A list of lengthe 2 with the total shunt impedances of the main and harmonic cavities
        *qfacor* - A list of lengthe 2 with the quality factors of the main and harmonic cavities
        *tinst*=None - If givenm, a transients.Transient class from which to extract the cavity voltages and phases. Otherwise, extract from the resonant frequencies and the beam current.
        *vector*-False - Attempt a calculation of the Robinson instabilities for different bunches individually (for comparison only, no physical basis)
        *use_ltune*=True - Use the incoherent synchrotron tune calculated from the gradient of the total RF voltage instead of the Robinson frequency defined by Bosch.
        """

        self.sring = sring
        self.tinst = tinst
        self.qfactor = np.array(qfactor)
        self.rs = np.array(rs)
        self.rfreq = np.array(rfreq)
        self.nharm = np.round(np.array(rfreq)/sring.frf)
        self.phi2 = np.arctan(2*qfactor[1]*(self.nharm[1]*sring.frf-rfreq[1])/rfreq[1])
        self.phi1 = np.arctan(2*qfactor[0]*(sring.frf-rfreq[0])/rfreq[0])        
        self.revfreq = sring.frf/sring.nbunch        

        if self.tinst!=None:

            if vector:
                self.ffact = np.absolute(tinst.formfact)
                self.v2 = np.absolute(tinst.landau_phasor[:,1])#/np.absolute(tinst.formfact[:,1]))
                self.blen = tinst.blen.real
        
                #Make sure that the phases are correct according to Bosch's convention
                #vphi1 = np.pi/2-np.mean(np.angle(tinst.mainrf_phasor))
                self.vphi1 = self.tinst.phi_rf
                self.vphi1 = np.pi/2.-np.angle(tinst.mainrf_phasor)
                self.vphi2 = -np.pi-np.angle(tinst.landau_phasor[:,1]/tinst.formfact[:,1])
            else:
                self.ffact = np.mean(np.absolute(tinst.formfact),0)
                self.v2 = np.mean(np.absolute(tinst.landau_phasor[:,1]))#/np.absolute(tinst.formfact[:,1]))
                self.blen = np.mean(tinst.blen.real)
        
                #Make sure that the phases are correct according to Bosch's convention
                #vphi1 = np.pi/2-np.mean(np.angle(tinst.mainrf_phasor))
                self.vphi1 = self.tinst.phi_rf
                self.vphi2 = -np.pi-np.mean(np.angle(tinst.landau_phasor[:,1]/tinst.formfact[:,1]))
        else:
            genphase = np.arctan(self.qfactor*(self.omegar/(self.nharm*self.sring.frf*2*np.pi)-2*np.pi*self.sring.frf*self.nharm/self.omegar))
            self.vphi1 = np.arcsin(self.sring.eloss/self.sring.vrf+np.sum(2*self.sring.current*self.rs[self.nharm!=1]*np.cos(genphase[self.nharm!=1])**2)/self.sring.vrf)            
            self.vphi2 = -np.pi-phi2
            self.ffact = np.ones(2)            
            self.v2 = 2*rs[1]*self.ffact[1]*sring.current*np.cos(self.phi2)
            self.blen = sring.blen
    
        #omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*revfreq/sring.energy*(ffact[0]*sring.vrf*np.cos(vphi1)+
        #                                                                      nharm[1]*ffact[1]*v2*np.cos(vphi2)))
        self.vphi1 = np.pi/2.-self.vphi1
        self.omegag = 2*np.pi*self.sring.frf        
        #vphi2 = vphi2
        if vector:
            self.omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*self.revfreq/sring.energy*(self.ffact[:,0]*sring.vrf*np.sin(self.vphi1)+
                                                                                            self.nharm[1]*self.ffact[:,1]*self.v2*np.sin(self.vphi2)))
            if use_ltune: self.omegas = self.omegag/self.sring.nbunch*self.tinst.ltune
            else: self.omegas = self.omegar[:]
        else:
            self.omegar = np.sqrt(sring.alphac*sring.frf*2*np.pi*self.revfreq/sring.energy*(self.ffact[0]*sring.vrf*np.sin(self.vphi1)+
                                                                                            self.nharm[1]*self.ffact[1]*self.v2*np.sin(self.vphi2)))
            self.omegas = self.omegar
            
        print 'Robinson tune:', self.omegar/2/np.pi/sring.frf*sring.nbunch

        self.vector = vector
        self.commconst = self.sring.alphac/self.sring.energy*self.revfreq        

    def calcLandauDamping(self):

        self.a = self.commconst*self.omegag/2.*(self.sring.vrf*np.sin(self.vphi1)+self.nharm[1]*self.v2*np.sin(self.vphi2))
        self.b = self.commconst*self.omegag**2/6.*(self.sring.vrf*np.cos(self.vphi1)+self.nharm[1]**2*self.v2*np.cos(self.vphi2))
        self.c = -self.commconst*self.omegag**3/24.*(self.sring.vrf*np.sin(self.vphi1)+self.nharm[1]**3*self.v2*np.sin(self.vphi2))    
        self.landauD = 0.78*self.sring.alphac**2*self.sring.espread**2/np.sqrt(2*self.a)*np.absolute(3*self.c/2./self.a-(3*self.b/2./self.a)**2)
        self.landauDD = 0.78*self.sring.alphac**2*self.sring.espread**2/self.omegas*np.absolute(3*self.c/self.omegas**2-(3*self.b/self.omegas**2)**2)
        #Tracer()()

    def getPhiPlusMinusArb(self,omegarg,qfactor,fres,bandno,mode1):

        freqlineplus = bandno*self.sring.frf+mode1*self.revfreq
        freqlineminus = bandno*self.sring.frf-mode1*self.revfreq        
        phip = np.arctan(2*qfactor*(freqlineplus+omegarg/2/np.pi-fres)/fres)
        phim = np.arctan(2*qfactor*(freqlineminus+omegarg/2/np.pi-fres)/fres)

        return phip, phim

    def calcArbMode(self,rs,qfactor,fres,mode=1,consistent=False,mode1=0,formfact=1,use_ltune=False):

        bandno = np.floor(fres/self.sring.frf)
        bigfreq = 2*np.pi*(mode1*self.revfreq+bandno*self.sring.frf)
        if use_ltune: ws = self.omegag/self.sring.nbunch*np.mean(self.tinst.ltune)
        else: ws = self.omegar

        def modeGrow(omegarg):

            phip, phim = self.getPhiPlusMinusArb(omegarg,qfactor,fres,bandno,mode1)
            #Tracer()()
    
            modefact = mode*(bigfreq*self.blen)**(2*mode-2)/2.**mode/math.factorial(mode-1)
            bigomega = np.sqrt((mode*ws)**2-self.sring.current*modefact*self.commconst*bigfreq*rs*formfact**2*(np.sin(2*phip)+np.sin(2*phim)))
            boschgrow = mode*mode*np.real(ws)*np.imag(ws)+bigfreq*self.sring.current*self.commconst/bigomega*modefact*formfact**2*rs*(np.cos(phim)**2-np.cos(phip)**2)
            
            return bigomega, boschgrow

        if consistent:
            tst = lambda x: (modeGrow(x[0])[0]-x[0])**2
            opti = optimize.fmin(tst,np.array([ws/2.]))
            bigomega_out, boschgrow_out = modeGrow(opti[0])
        else:
            bigomega_out, boschgrow_out = modeGrow(ws)
        
        #currterm = -self.sring.current*modefact*self.sring.alphac*self.sring.frf*2*np.pi/self.sring.energy*self.revfreq*(self.rs[0]*self.ffact[0]**2*(np.sin(2*phi1p)+np.sin(2*phi1m))
        #+self.nharm[1]**(2*mode-1)*self.rs[1]*self.ffact[1]**2*(np.sin(2*phi2p)+np.sin(2*phi2m)))
        #deltaomega = np.sqrt(np.absolute(currterm))*np.sign(currterm)

        return bigomega_out+1j*boschgrow_out        

    def getPhi12PlusMinus(self,omegarg):

        phi1p = np.arctan(2*self.qfactor[0]*(self.sring.frf+omegarg/2/np.pi-self.rfreq[0])/self.rfreq[0])
        phi1m = np.arctan(2*self.qfactor[0]*(self.sring.frf-omegarg/2/np.pi-self.rfreq[0])/self.rfreq[0])
    
        phi2p = np.arctan(2*self.qfactor[1]*(self.nharm[1]*self.sring.frf+omegarg/2/np.pi-self.rfreq[1])/self.rfreq[1])
        phi2m = np.arctan(2*self.qfactor[1]*(self.nharm[1]*self.sring.frf-omegarg/2/np.pi-self.rfreq[1])/self.rfreq[1])

        return phi1p, phi1m, phi2p, phi2m
    
    def calcModeGrowth(self,mode=1,consistent=False,mode1=0):
        
        def modeGrow(omegarg):

            phi1p, phi1m, phi2p, phi2m = self.getPhi12PlusMinus(omegarg+2*np.pi*mode1*self.revfreq)
            #Tracer()()
    
            modefact = mode*(self.sring.frf*2*np.pi*self.blen)**(2*mode-2)/2.**mode/math.factorial(mode-1)
            boschgrow = mode*mode*np.real(self.omegar)*np.imag(self.omegar)+8*self.sring.current*self.commconst*modefact*(self.ffact[0]**2*self.rs[0]*self.qfactor[0]*np.tan(self.phi1)*np.cos(phi1p)**2*np.cos(phi1m)**2
                                                                                                                     +self.nharm[1]**(2*mode-2)*self.ffact[1]**2*self.rs[1]*self.qfactor[1]*np.tan(self.phi2)*np.cos(phi2p)**2*np.cos(phi2m)**2)
            bigomega = np.sqrt((mode*self.omegar)**2-self.sring.current*modefact*self.commconst*self.sring.frf*2*np.pi*(self.rs[0]*self.ffact[0]**2*(np.sin(2*phi1p)+np.sin(2*phi1m))
                                                                                                                        +self.nharm[1]**(2*mode-1)*self.rs[1]*self.ffact[1]**2*(np.sin(2*phi2p)+np.sin(2*phi2m))))
            return bigomega, boschgrow

        if consistent:
            tst = lambda x: (modeGrow(x[0])[0]-x[0])**2
            opti = optimize.fmin(tst,np.array([self.omegar/2.]))
            bigomega_out, boschgrow_out = modeGrow(opti[0])
        else:
            bigomega_out, boschgrow_out = modeGrow(self.omegar)
        
        #currterm = -self.sring.current*modefact*self.sring.alphac*self.sring.frf*2*np.pi/self.sring.energy*self.revfreq*(self.rs[0]*self.ffact[0]**2*(np.sin(2*phi1p)+np.sin(2*phi1m))
        #+self.nharm[1]**(2*mode-1)*self.rs[1]*self.ffact[1]**2*(np.sin(2*phi2p)+np.sin(2*phi2m)))
        #deltaomega = np.sqrt(np.absolute(currterm))*np.sign(currterm)

        return bigomega_out+1j*boschgrow_out

    def modeCouple(self,omegaind,omegainq,consistent=False):
        
        ccnst = self.commconst*self.omegag*self.sring.current
        mode = 1
        #rep = 0

        def modeGrow(omegarg,couplegrow):
            
            phi1p, phi1m, phi2p, phi2m = self.getPhi12PlusMinus(omegarg)

            sin1plus = np.sin(2*phi1m)+np.sin(2*phi1p)
            sin2plus = np.sin(2*phi2m)+np.sin(2*phi2p)
            sin1minus = np.sin(2*phi1m)-np.sin(2*phi1p)
            sin2minus = np.sin(2*phi2m)-np.sin(2*phi2p)
            cos1minus = np.cos(phi1m)**2-np.cos(phi1p)**2
            cos2minus =  np.cos(phi2m)**2-np.cos(phi2p)**2
            cos1plus = np.cos(phi1m)**2+np.cos(phi1p)**2
            cos2plus =  np.cos(phi2m)**2+np.cos(phi2p)**2            
            
            bigAcurl = ccnst/2.*(self.rs[0]*self.ffact[0]**2*sin1plus+self.nharm[1]*self.rs[1]*self.ffact[1]**2*sin2plus)
            bigBcurl = ccnst/2.*(self.omegag*self.blen)**2*(self.rs[0]*self.ffact[0]**2*sin1plus+self.nharm[1]**3*self.rs[1]*self.ffact[1]**2*sin2plus)
            bigDcurl = ccnst/2.*(self.omegag*self.blen)*(self.rs[0]*self.ffact[0]**2*sin1minus+self.nharm[1]**2*self.rs[1]*self.ffact[1]**2*sin2minus)

            acurl = ccnst*(self.rs[0]*self.ffact[0]**2*cos1minus+self.nharm[1]*self.rs[1]*self.ffact[1]**2*cos2minus)+2*omegarg/self.sring.taue
            bcurl = ccnst*(self.omegag*self.blen)**2*(self.rs[0]*self.ffact[0]**2*cos1minus+self.nharm[1]**3*self.rs[1]*self.ffact[1]**2*cos2minus)+4*omegarg/self.sring.taue
            dcurl = ccnst*(self.omegag*self.blen)*(self.rs[0]*self.ffact[0]**2*cos1plus+self.nharm[1]**2*self.rs[1]*self.ffact[1]**2*cos2plus)

            #omegarg = omegarg+1j*omegaimag
            if not consistent:
                couplegrow = (acurl*(omegarg**2-(2*self.omegar)**2+bigBcurl)+bcurl*(omegarg**2-self.omegar**2+bigAcurl)-2*bigDcurl*dcurl)/2./omegarg/(2*omegarg**2-5*self.omegar**2+bigAcurl+bigBcurl)
                #rep += 1
            
            term2 = np.sqrt(complex((3*self.omegar**2+bigAcurl-bigBcurl)**2/4.+bigDcurl**2-dcurl**2+(acurl-2*omegarg*couplegrow)*(bcurl-2*omegarg*couplegrow)))
            #term2 = np.sqrt((3*self.omegar**2+bigAcurl-bigBcurl)**2/4.+bigDcurl**2-dcurl**2+(acurl-2*omegarg*couplegrow)*(bcurl-2*omegarg*couplegrow))
            term1 = (5*self.omegar**2-bigAcurl-bigBcurl)/2.
            bigomegad = np.sqrt(complex(term1-term2))
            bigomegaq = np.sqrt(complex(term1+term2))
            #bigomegad = np.sqrt(term1-term2)
            #bigomegaq = np.sqrt(term1+term2)
            
            if mode==1:
                if consistent:
                    couplegrow = (acurl*(bigomegad**2-(2*self.omegar)**2+bigBcurl)+bcurl*(bigomegad**2-self.omegar**2+bigAcurl)-2*bigDcurl*dcurl)/2./bigomegad/(2*bigomegad**2-5*self.omegar**2+bigAcurl+bigBcurl)                            
                return bigomegad, couplegrow
            elif mode==2:
                if consistent:
                    couplegrow = (acurl*(bigomegaq**2-(2*self.omegar)**2+bigBcurl)+bcurl*(bigomegaq**2-self.omegar**2+bigAcurl)-2*bigDcurl*dcurl)/2./bigomegaq/(2*bigomegaq**2-5*self.omegar**2+bigAcurl+bigBcurl)
                return bigomegaq, couplegrow

        if consistent:
            def tst(x):
                res = modeGrow(x[0],x[1])[0]
                return (np.real(res)-x[0])**2
            #tst = lambda x: (modeGrow(x[0],x[1])[0]-x[0])**2
            opti = optimize.fmin(tst,np.array([np.real(omegaind),np.imag(omegaind)]))
            bigomegad_out, boschgrowd_out = modeGrow(opti[0],opti[1])
            mode = 2
            opti = optimize.fmin(tst,np.array([np.real(omegainq),np.imag(omegainq)]))
            bigomegaq_out, boschgrowq_out = modeGrow(opti[0],opti[1])
        else:
            bigomegad_out, boschgrowd_out = modeGrow(np.real(omegaind),np.imag(omegaind))
            mode = 2
            bigomegaq_out, boschgrowq_out = modeGrow(np.real(omegainq),np.imag(omegainq))
        #Tracer()()

        return bigomegad_out, bigomegaq_out, boschgrowd_out, boschgrowq_out

def landauDampingFlat(sring,nharm):

    ubyv = sring.eloss/sring.vrf
    kflat = np.sqrt(1./nharm**2-1./(nharm**2-1.)*ubyv**2)
    tannphi = -nharm*ubyv/np.sqrt((nharm**2-1.)**2-(nharm*nharm*ubyv)**2)
    nphi = np.arctan(tannphi)
    phis = np.arcsin(ubyv-kflat*np.sin(nphi))

    omegag = 2*np.pi*sring.frf
    c = -sring.alphac*omegag**3/24./sring.energy*sring.frf/sring.nbunch*sring.vrf*(np.sin(np.pi/2-phis)-kflat*nharm**3*np.sin(np.pi/2-nphi))
    filling_height = sring.alphac*sring.alphac*sring.espread**2/2.

    return nphi,kflat,phis,c,filling_height

def flatDistribution(sring,nharm,time,nphi,kflat,phis):
    
    vrf_tot = sring.vrf*np.sin(2*np.pi*sring.frf*time+phis)-sring.eloss
    vlc = -sring.vrf*kflat*np.sin(2*np.pi*nharm*sring.frf*time-nphi)
    gammakvart = gamma(0.25)
    qso = np.sqrt(sring.nbunch*sring.alphac*sring.vrf*np.sqrt(1-(sring.eloss/sring.vrf)**2)/2./np.pi/sring.energy)
    sigmatau = 2*np.sqrt(np.pi)/gammakvart*(3./(nharm*nharm-1))**(1/4.)*np.sqrt(sring.nbunch*sring.alphac*sring.espread/qso)/2./np.pi/sring.frf
    dist = 8**0.25/gammakvart/gammakvart/sigmatau*np.sqrt(2*np.pi)*np.exp(-2*np.pi*np.pi*(time/sigmatau/gammakvart)**4)
    print 'Bunch length with and without lengthening (ps):', sring.espread*sring.alphac*sring.nbunch/2./np.pi/sring.frf/qso, sigmatau

    return time, vrf_tot, vlc, dist


            
