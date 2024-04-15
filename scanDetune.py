import transient, haissinski, bosch, cbm_solver, cbm_vlasov
import numpy as np
from numpy import ma
from scipy import special, optimize
#from IPython.core.debugger import Tracer

class BoschAnalysis:
    """
    Comprehensive class for calculating the stability of coupled-bunch modes 0 and 1 in the presence of Landau cavities. The first
    step is to calculate the bunch profiles. Other classes are then used to calculate predictions for the collective effects,
    particularly the complex coherent tune shifts and approximate growth rates.
    """    

    import cavity_tuning

    def __init__(self,sring,cavparamfile,nharm=3,current=np.arange(0,0.5,0.001),detune=np.arange(70,90,1)*1e3,flatvrf=False,sringsingle=None,brentthreshold=np.inf,brentkwargs={},boschkwargs={},transkwargs={},scaledetune=False,additional_resonator=None,forceflat=False,activeHC=False,activeHCBeta=1.,use_boschwr=False,omegasapprox=False,zerofreq=False,deltinsts=False):
        """
        Initialisation function. This function must be followed by calling the getVrfDetune function in order to start calculation of
        the results.
        *sring* - utility.StorageRing instance
        *cavparamfile* - json file with the cavity parameters in the storage ring
        *nharm*=3 - Operational RF harmonic of the Landau cavities.
        *current* - A range of currents over which to do the calculations.
        *detune* - The detuning of the Landau cavities over which to do the calculations. This combined with the *currents* argument
                   provides a grid for almost all of the results. [Hz]
        *flatvrf*=False - If True, attempt to calculate at the appropriate RF voltage for quartic potential at each bean current.
        *sringsingle*=None - utility.StorageRing instance that is a model of the *sring* instance but with just one RF bucket to speed
                             up the calculation (sringsingle.nbunch must be equal to 1 and sringsingle.alphac must be equal to
                             sring.alphac*sring.nbunch). This can speed up the calculations but is not appropriate for uneven fills.
        *brenthreshold*=np.inf - The threshold detuning below which to use the Brent optimisaion method to determine the bunch profiles.
                                 This can work well for overstretched bunches but not so well at large detuneings and is is
                                 not appropriate for nonuniform fills.
        *brentkwargs*={} - Dictionary of keyword arguments to pass to the brentOptimise function to perform the Brent optimisation.
        *boschkwargs*={} - Dictionary of keyword arguments to pass to the BoschAnalysis.calcModeGrowth function
        *transkwargs*={} - Dictionary of keyword arguments to pass to the transients.Transient class
        *scaledetune*=False - If true, interpret the *detune* argument as a scaled value to be adapted for each current. This can be
                              useful when trying to evaluate similar RF potentials over multiple beam currents.
        *additional_resonator*=None - Add an additional resonator impedance for calculations of the mode stabilities.
        *forceflat*=False - If True, force one Landau-cavity detuning to be the value that corresponds roughly to flat-potential
                            conditions.
        *activeHC*=False - If True, treat the Landau cavity as an active cavity, adjust its shunt impedance and quality factor
                           according to the *activeHCBeta* value and include an RF feedback in the Transient instance.
        *activeHCBeta*=1. - Coupling beta of the active harmonic cavity.
        *use_boschwr*=False - If True, use the Robinson frequency defined by Bosch instead of the incoherent synchrotron frequency
                              calculated from the slope of the total RF voltage at the synchronous phase in the calculations of the
                              complex coherent frequencies and growth rates.
        *omegasapprox*=False - If True, use the small coherent frequency shift approximation:
                               (\Omega^2-\omega_s^2)=2\omega_s*(\Omega-\omega)
        *zerofreq*=False - If True, calculate the coupling matrices assuming an incoherent synchrotron frequency of 0. This is
                           appropriate when trying to include Landau damping.
        *deltinsts*=False - If True, delete the transients.Transient instances at the end of the calculation to save memory: useful
                            when running multiple calculations consecutively.
        """
        
        #Need to scan current and detuning and make evaluate dipole and quadrupole instabilities
        self.sring = sring
        self.cavity_tuning.init(cavparamfile)
        self.cavparams = self.cavity_tuning.cavparams
        self.hcrs = self.cavparams['LC']['cavity']['Rs']
        self.hcq = self.cavparams['LC']['cavity']['Q']
        self.mainrs = self.cavparams['main']['cavity']['Rs']
        self.mainq = self.cavparams['main']['cavity']['Q']
        self.cscaleddetune = scaledetune
        self.activeHC = activeHC
        if isinstance(activeHCBeta,(float,int)):
            self.activeHCBeta = np.ones(2)*activeHCBeta
        else:
            self.activeHCBeta = activeHCBeta
        
        self.brentthreshold = brentthreshold
        self.brentkwargs = {'delta':0.3,'niterations':1,'formcalc':'full','blength':True,'fill':1.0}
        self.brentkwargs.update(brentkwargs)
        self.transkwargs = self.brentkwargs.copy()
        self.brentkwargs.pop('fill')
        self.transkwargs.pop('niterations')
        self.transkwargs.pop('delta')
        self.transkwargs.update({'niterations':50,'blenskip':5})
        self.transkwargs.update(transkwargs)
        self.titerkwargs = {}
        self.titerkwargs.update({'niterations':self.transkwargs.pop('niterations')})
        self.titerkwargs.update({'blenskip':self.transkwargs.pop('blenskip')})
        self.boschkwargs = {'consistent':True}#,'singlep':False}
        self.boschkwargs.update(boschkwargs)
        self.use_boschwr = use_boschwr
        self.zerofreq = zerofreq
        self.omegasapprox = omegasapprox
        self.deltinsts = deltinsts
        
        self.current = current
        self.detune = detune
        self.nharm = nharm
        self.flatvrf = flatvrf
        self.flatpot = 2*self.cavparams['LC']['cavity']['Rs']/self.sring.eloss*(self.nharm*self.nharm-1)
        self.flatdeltaf = semiFlat(self.cavparams['LC']['cavity']['Rs'],self.cavparams['LC']['cavity']['Q'],3,self.sring,0.97)
        if forceflat:
            if self.flatvrf and self.cscaleddetune:
                self.detune[np.argmin(np.absolute(self.detune-self.flatpot))] = self.flatpot#*0.97
            elif not self.cscaleddetune:
                self.detune[np.argmin(np.absolute(self.detune-self.flatdeltaf))] = self.flatdeltaf

        if sringsingle==None:
            self.sring_single = self.sring
        else:
            self.sring_single = sringsingle

        self.additional_resonator = additional_resonator
        self.flatform = 1

    def calcQso(self,vrf):
        return np.sqrt(self.sring.nbunch*self.sring.alphac*vrf*np.sqrt(1-(self.sring.eloss/vrf)**2)/2./np.pi/self.sring.energy)

    def getVrfDetune(self):
        """
        Obligatory function call to calculate the results. Important class member attributes after calling:
        self.chao - Prediction of Mode 0 complex coherent frequency shifts according to Chao.
        self.boschd - Prediction of complex coherent frequencies of the Robinson dipole mode according to Bosch.
        self.boschq - Prediction of complex coherent frequencies of the Robinson quadrupole mode according to Bosch.
        self.bosch1p - Prediction of the complex coherent frequencies of Mode +1 according to Bosch.
        self.bosch1m - Prediction of the complex coherent frequencies of Mode -1 according to Bosch.
        self.boschcoupled - Complex coherent frequencies of the Robinson coupled dipole mode according to Bosch.
        self.boschcoupleq - Complex coherent frequencies of the Robinson coupled quadrupole mode according to Bosch.
        self.boschcpledgrow - Growth rate of the Robinson coupled dipole mode according to Bosch.
        self.boschcpleqgrow - Growth rate of the Robinson coupled quadrupole mode according to Bosch.
        self.omegar - The Robinson angular frequency according to Bosch
        self.wrob - The angular incoherent synchrotron frequency used in the calculation of complex coherent frequencies (not Bosch).
        self.ruthd - Prediction of complex coherent frequencies of the Robinson dipole mode according to Thompson and Ruth.
        self.ruthd1 - Prediction of complex coherent frequencies of Modes+/-1  according to Thompson and Ruth.
        self.wrob_v - The angular incoherent synchrotron frequency used in the Vlasov calculations.
        self.ruthd_v - Prediction of complex coherent frequencies of the Robinson dipole mode using the Vlasov approach.
        self.ruthd1_v - Prediction of complex coherent frequencies of the +/-1 dipole modes using the Vlasov approach.
        self.ruthq_v - Prediction of complex coherent frequencies of the Robinson quadrupole mode using the Vlasov approach.
        self.ruthq1_v - Prediction of complex coherent frequencies of the +/-1 quadrupole modes using the Vlasov approach.
        self.vrf - RF voltage assumed at each beam current (will change if *flatvrf* is set to True).
        self.boschLandau - Landau damping rate according to Bosch.
        self.deltaf - The Landau detuning from the RF harmonic (equal to *detune* for all currents if  *scaleddetune* is set to False)
        self.hcfield - The Landau voltage
        self.blenbar - The average bunch length
        self.ltunebar - The average incoherent synchrotron frequency.
        self.ffactbar - The average form factor
        self.venturini - The complex coherent frequency shift close to flat potential according to the Venturini approximation
        self.tailonhe - The outcome of the Tianlong He stability predictions for Mode +1
        self.tinsts - A collection of the transients.Transient instances (or Nones if *deltinsts* is set to False)
        self.bmans - A collection of the bosch.BoschInstability instances
        """

        timeax = np.arange(-1,1,0.001)*1e-9
        gammakvart = special.gamma(0.25)
        
        def idealCurrent(vrf,target):
            qso = self.calcQso(vrf)
            sigma_tau = 1./gammakvart*(3./(self.nharm**2-1))**0.25*np.sqrt(self.sring.nbunch*self.sring.alphac*self.sring.espread/qso)/np.sqrt(np.pi)/self.sring.frf
            #formfact = 1-(self.nharm*self.sring.frf*2*np.pi*sigma_tau)**2/2.
            bdist = np.sqrt(2*np.pi*np.sqrt(8))/gammakvart/gammakvart/sigma_tau*np.exp(-2*np.pi**2*(timeax/gammakvart/sigma_tau)**4)
            formfact = np.absolute(np.trapz(bdist*np.exp(1j*self.nharm*self.sring.frf*2*np.pi*timeax),x=timeax)/np.trapz(bdist,x=timeax))
            self.flatform = 1*formfact
            #print formfact
            kfp = np.sqrt(1./self.nharm**2-1./(self.nharm**2-1)*(self.sring.eloss/vrf)**2)
            detang = np.pi/2.-np.arctan(-(self.nharm*self.sring.eloss/vrf)/np.sqrt((self.nharm**2-1)**2-(self.nharm**2*self.sring.eloss/vrf)**2))
         
            current = kfp*vrf/2./formfact/self.hcrs/np.absolute(np.cos(detang))
        
            return (current-target)**2
        
        #def calcVrf(ffactor,target):
        #    
        #    bigK = 1/(2*target*ffactor*self.hcrs)
        #    bigK = bigK*bigK
        #    n2m1 = self.nharm*self.nharm-1.
        #    #total_energy_loss = self.getGraphObject('total_energy_loss')[0]
        #    u4 = self.sring.eloss**4
        #    u2 = self.sring.eloss*self.sring.eloss
        #    n2 = self.nharm*self.nharm
        #
        #    coeffs = [bigK*n2m1**4/n2,
        #              -bigK*n2m1**3*u2+bigK*n2m1*n2m1*u2*n2-bigK*u2*n2m1**3,
        #              -bigK*self.nharm**4*n2m1+bigK*n2*n2m1*u4-bigK*n2m1*self.nharm**4*u4-n2*u2*n2m1*n2m1,
        #              -bigK*(self.nharm*self.sring.eloss)**6+(self.nharm*self.sring.eloss)**4]
        #    
        #    mainv = np.sqrt(np.amax(np.roots(coeffs)))
        #    
        #    return mainv

        #def idealCurrent(ffactor,target):
        #
        #    vrf = calcVrf(ffactor[0],target)
        #    qso = self.calcQso(vrf)
        #    sigma_tau = 1./gammakvart*(3./(self.nharm**2-1))**0.25*np.sqrt(self.sring.nbunch*self.sring.alphac*self.sring.espread/qso)/np.sqrt(np.pi)/self.sring.frf
        #    #formfact = 1-(self.nharm*self.sring.frf*2*np.pi*sigma_tau)**2/2.
        #    bdist = np.sqrt(2*np.pi*np.sqrt(8))/gammakvart/gammakvart/sigma_tau*np.exp(-2*np.pi**2*(timeax/gammakvart/sigma_tau)**4)
        #    formfact = np.absolute(np.trapz(bdist*np.exp(1j*self.nharm*self.sring.frf*2*np.pi*timeax),x=timeax)/np.trapz(bdist,x=timeax))
        #
        #    return (formfact-ffactor[0])**2

        self.chao = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschd = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschq = np.zeros((len(self.current),len(self.detune)),complex)
        self.bosch1p = np.zeros((len(self.current),len(self.detune)),complex)
        self.bosch1m = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschcoupled = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschcoupleq = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschcpledgrow = np.zeros((len(self.current),len(self.detune)),complex)
        self.boschcpleqgrow = np.zeros((len(self.current),len(self.detune)),complex)     
        self.omegar = np.zeros((len(self.current),len(self.detune)))
        self.wrob = np.zeros((len(self.current),len(self.detune)))
        self.ruthd = np.zeros((len(self.current),len(self.detune)),complex)
        self.ruthd1 = np.zeros((len(self.current),len(self.detune),2),complex)
        self.wrob_v = np.zeros((len(self.current),len(self.detune)))
        self.ruthd_v = np.zeros((len(self.current),len(self.detune)),complex)
        self.ruthd1_v = np.zeros((len(self.current),len(self.detune),2),complex)
        self.ruthq_v = np.zeros((len(self.current),len(self.detune)),complex)
        self.ruthq1_v = np.zeros((len(self.current),len(self.detune),2),complex)        
        self.vrf = np.zeros(len(self.current))
        self.boschLandau = np.zeros((len(self.current),len(self.detune)))
        self.deltaf = np.zeros((len(self.current),len(self.detune)))
        self.hcfield = np.zeros((len(self.current),len(self.detune)))
        self.blenbar = np.zeros((len(self.current),len(self.detune)))
        self.ltunebar = np.zeros((len(self.current),len(self.detune)))
        self.ffactbar = np.zeros((len(self.current),len(self.detune)),complex)
        #self.omegarq = np.zeros((len(self.current),len(self.detune)),complex)
        self.venturini = np.zeros((len(self.current),len(self.detune)),complex)
        self.venturinoff = np.zeros((len(self.current),len(self.detune)),complex)
        self.tailonhe = np.zeros((len(self.current),len(self.detune)))
        self.tinsts = []
        self.bmans = []
        self.cbms = []
        self.vbms = []
        self.maindetune = []
        self.landaudetune = []

        revfreq = self.sring.frf/self.sring.nbunch
        rs = [self.mainrs,self.hcrs]
        q = [self.mainq,self.hcq]
        for i,c in enumerate(self.current):

            tin = []
            bm = []
            cb = []
            vb = []
            md = []
            ld = []
            
            self.sring.current = c
            self.sring_single.current = c
            if self.flatvrf:
                #self.flatform = optimize.root(idealCurrent,[1.],(c,))[0]
                #self.sring.vrf = calcVrf(self.flatform,c)
                self.sring.vrf = optimize.fmin(idealCurrent,1e6,(c,))[0]*1.001
                self.sring_single.vrf = 1*self.sring.vrf*1.001
                idealCurrent(self.sring.vrf,c)
            self.vrf[i] = 1*self.sring.vrf
            maindetune = self.cavity_tuning.mainDetune(c,self.sring.vrf,self.sring.eloss)
            
            if self.cscaleddetune:
                #self.deltaf[i] = self.sring.frf*self.nharm*np.tan(np.arccos(1/np.sqrt(self.detune*c*self.flatform)))/2./self.hcq
                self.deltaf[i] = self.sring.frf*self.nharm*np.tan(np.arccos(1/np.sqrt(self.detune*c*self.flatform)))/2./self.hcq
            else: self.deltaf[i] = self.detune
            
            for j,d in enumerate(self.deltaf[i]):
                tune = self.calcQso(self.sring.vrf)
                rs_tmp = [0,self.hcrs]
                nharm_tmp = [1,self.nharm]
                detune_tmp = [maindetune,d]
                q_tmp = q[:]
                if np.all(self.additional_resonator!=None):# and self.detune[j]>=self.brentthreshold:
                    rs_tmp += self.additional_resonator[:1]
                    nharm_tmp += [0]
                    detune_tmp += self.additional_resonator[1:2]
                    q_tmp += self.additional_resonator[2:]
                #t = transient.Transient(self.sring_single,rs_tmp,[1,3],[maindetune,d],q,fill=1.0,blength=True,formcalc='full')
                if self.detune[j]<self.brentthreshold:
                    t = brentOptimise(self.sring_single,rs_tmp,nharm_tmp,detune_tmp,q_tmp,fill=1.0,**self.brentkwargs)['inst']
                else:
                    if np.all(self.transkwargs['fill']==1) and isinstance(self.transkwargs['fill'],(float,int)):
                        t = transient.Transient(self.sring_single,rs_tmp,nharm_tmp,detune_tmp,q_tmp,**self.transkwargs)
                    else:
                        t = transient.Transient(self.sring,rs_tmp,nharm_tmp,detune_tmp,q_tmp,**self.transkwargs)
                        t.time_off = 5*t.sring.blen*np.sin(np.arange(t.sring.nbunch)/t.sring.nbunch*2*np.pi)
                    t.runIterations(self.titerkwargs['niterations'],blenskip=self.titerkwargs['blenskip'])
                    isn = np.isnan(t.ltune)
                    if np.any(isn) and np.sum(isn)<5 and len(t.ltune)>10:
                        print('Starting while loop')
                        while np.any(isn):
                            mid = np.where(isn)[0][0]
                            st = mid-1
                            while np.isnan(t.ltune[st]):
                                st = st-1
                            en = (mid+1)%t.sring.nbunch
                            while np.isnan(t.ltune[en]):
                                en = (en+1)%t.sring.nbunch
                            t.ltune[mid] = np.mean(t.ltune[[st,en]])
                            isn = np.isnan(t.ltune)
                        print('Out of while loop')

                if np.isnan(t.dist).all():  self.hcfield[i,j] = np.nan
                else:  self.hcfield[i,j] = np.mean(np.absolute(t.landau_phasor[:,1]))
                blbar = np.mean(np.real(t.blen))
                ffactb = np.mean(t.formfact[:,1])
                ltuneb = np.mean(t.ltune)
                self.blenbar[i,j] = 1*blbar
                self.ltunebar[i,j] = 1*ltuneb
                self.ffactbar[i,j] = ffactb
                maindetune = self.cavity_tuning.mainDetune(c,self.sring.vrf,self.sring.eloss,np.mean(np.absolute(t.formfact[:,0])))
                rfreq = [self.sring.frf+maindetune,self.sring.frf*self.nharm+d]
                bman = bosch.BoschInstability(self.sring,rfreq,rs,q,t)
                
                #try:
                #    t.runIterations(100,blenskip=5)
                #except np.linalg.linalg.LinAlgError:
                #    print 'Skipping %.1f Hz detuning' % d
                #    continue
    
                self.chao[i,j] = haissinski.robinsonInstability(tune,revfreq,rfreq,rs,q,
                                                                energy=self.sring.energy,alpha=self.sring.alphac,ffact=np.mean(t.formfact))*c
                
                #self.boschd[i,j], self.omegar[i,j], self.boschLandau[i,j]= haissinski.boschInstability(self.sring,rfreq,rs,q,t,mode=1,**self.boschkwargs)
                #self.boschq[i,j], self.omegarq[i,j], tmp = haissinski.boschInstability(self.sring,rfreq,rs,q,t,mode=2,**self.boschkwargs)
                
                self.omegar[i,j] = bman.omegar
                bman.calcLandauDamping()
                self.boschLandau[i,j] = bman.landauD
                self.boschd[i,j] = bman.calcModeGrowth(mode=1,**self.boschkwargs)
                self.boschq[i,j] = bman.calcModeGrowth(mode=2,**self.boschkwargs)
                self.bosch1p[i,j] = bman.calcModeGrowth(mode=1,mode1=1,**self.boschkwargs)
                self.bosch1m[i,j] = bman.calcModeGrowth(mode=1,mode1=-1,**self.boschkwargs)
                self.boschcoupled[i,j], self.boschcoupleq[i,j], self.boschcpledgrow[i,j], self.boschcpleqgrow[i,j] = bman.modeCouple(self.boschd[i,j],self.boschq[i,j],consistent=True)
                ##Tracer()()

                if t.sring.nbunch==1:
                    t.changeNBunch(self.sring)
                #t.blength = False
                landaudetune = 0
                if self.activeHC:
                    rs[1] = rs[1]/(1.+self.activeHCBeta[0])
                    q[1] = q[1]/(1.+self.activeHCBeta[1])
                    if self.activeHC!=1: landaudetune = self.activeHC
                    else:
                        v = np.mean(np.absolute(t.landau_phasor[:,1]))
                        phi = np.mean(np.angle(t.landau_phasor[:,1]))
                        ffact = np.mean(np.absolute(t.formfact[:,1]))
                        #landaudetune = self.cavity_tuning.activeHCDetune(c,v,phi,ffact)                        
                        tanpsi = -2*c*ffact*rs[1]/v*np.sin(phi)
                        landaudetune = self.nharm*self.sring.frf*tanpsi/q[1]/2.
                    rfreq[1] = self.sring.frf*self.nharm+landaudetune

                rs_tmp = rs[:]
                rfreq_tmp = rfreq[:]
                q_tmp = q[:]
                if np.all(self.additional_resonator!=None):
                    rs_tmp += self.additional_resonator[:1]
                    rfreq_tmp += self.additional_resonator[1:2]
                    q_tmp += self.additional_resonator[2:]
                    
                cbm = cbm_solver.CbmTdmSolve(t,rs_tmp,rfreq_tmp,q_tmp,omegasapprox=self.omegasapprox,use_boschwr=self.use_boschwr)
                vbm = cbm_vlasov.VlasovSolve(t,rs_tmp,rfreq_tmp,q_tmp,omegasapprox=self.omegasapprox,use_boschwr=self.use_boschwr)
                #try:
                #    cbm.solveSelfConsistent(0)
                #except np.linalg.LinAlgError:
                try:
                    self.wrob[i,j] = np.mean(cbm.wrob)                    
                    if self.zerofreq:
                        if self.omegasapprox:
                            omegas = t.sring.espread*t.sring.alphac/blbar
                            cbm.wrob = np.array([omegas])
                        else:
                            cbm.wrob = np.array([0])
                    cbm.constructMatrix()#self.boschd[i,j].real)
                    cbm.solvEigen()
                    cbm.laplaceTransform()
                    cbm.constructMatrix(np.real(cbm.bigOmega[np.where(cbm.eigenmodes_lplcenum==0)[0][0]]))
                    cbm.solvEigen()
                    cbm.laplaceTransform()
                    #Tracer()()
                    self.ruthd[i,j] = cbm.bigOmega[np.where(cbm.eigenmodes_lplcenum==0)[0][0]]
                    if np.any(cbm.eigenmodes_lplcenum==1):
                        self.ruthd1[i,j,0] = cbm.bigOmega[cbm.eigenmodes_lplcenum==1][0]
                    if np.any(cbm.eigenmodes_lplcenum==175):
                        self.ruthd1[i,j,1] = cbm.bigOmega[cbm.eigenmodes_lplcenum==175][0]
                        #self.ruthd1[i,j] = cbm.bigOmega[np.where((cbm.eigenmodes_lplcenum==1) | (cbm.eigenmodes_lplcenum==175))[0][:2]]
                except np.linalg.LinAlgError:
                    pass

                try:
                    #vbm.integral()
                    if self.zerofreq:
                        if self.omegasapprox:
                            omegas = t.sring.espread*t.sring.alphac/blbar
                            vbm.wrob = np.array([omegas])
                        else:
                            vbm.wrob = np.array([0])                    
                    #if self.zerofreq: vbm.wrob = np.array([0])
                    vbm.constructMatrixKrinsky(order=1,numericint=False)#self.boschd[i,j].real)
                    vbm.solvEigen()
                    vbm.laplaceTransform()
                    wrobbar = None
                    #wrobbar = np.real(vbm.bigOmega[np.where(cbm.eigenmodes_lplcenum==0)[0][0]])#self.boschd[i,j].real)
                    niter = 0
                    for n in range(niter):
                        wrobbar = np.real(vbm.bigOmega[np.argmax(np.absolute(vbm.eigenmodes_laplace[0]))])
                        #wrobbar = self.boschd[i,j].real
                        vbm.constructMatrixKrinsky(wrobbar,order=1,numericint=False)
                        vbm.solvEigen()
                        vbm.laplaceTransform()
                        #if n==niter-2: vbm.omegasapprox = True
                    #Tracer()()
                    self.wrob_v[i,j] = np.mean(vbm.wrob)
                    self.ruthd_v[i,j] = vbm.bigOmega[np.argmax(np.absolute(vbm.eigenmodes_laplace[0]))]
                    if np.any(vbm.eigenmodes_lplcenum==1):
                        self.ruthd1_v[i,j,0] = vbm.bigOmega[vbm.eigenmodes_lplcenum==1][0]
                    if np.any(vbm.eigenmodes_lplcenum==175):
                        self.ruthd1_v[i,j,1] = vbm.bigOmega[vbm.eigenmodes_lplcenum==175][0]
                        #self.ruthd1_v[i,j] = vbm.bigOmega[[np.argmin(np.absolute(vbm.eigenmodes_lplcenum-1)),np.argmin(np.absolute(vbm.eigenmodes_lplcenum-175))]]
                    vbm.constructMatrixKrinsky(wrobbar,order=2)#self.boschd[i,j].real)
                    vbm.solvEigen()
                    vbm.laplaceTransform()
                    self.ruthq_v[i,j] = vbm.bigOmega[np.argmax(np.absolute(vbm.eigenmodes_laplace[0]))]
                    self.ruthq1_v[i,j] = vbm.bigOmega[[np.argmin(np.absolute(vbm.eigenmodes_lplcenum-1)),np.argmin(np.absolute(vbm.eigenmodes_lplcenum-175))]]   
                except np.linalg.LinAlgError, ValueError:
                    pass                
                
                self.venturini[i,j], self.venturinoff[i,j] = haissinski.venturiniInstability(revfreq,rfreq,rs,q,energy=self.sring.energy,alpha=self.sring.alphac,
                                                                                             blen=np.mean(t.blen),nbunch=self.sring.nbunch,current=c,omegas0=self.wrob[i,j])
                self.tailonhe[i,j] = haissinski.heInstability(t,t.sring)

                if self.activeHC:
                    rs[1] = rs[1]*(1+self.activeHCBeta[0])
                    q[1] = q[1]*(1+self.activeHCBeta[1])
                
                bm.append(bman)
                cb.append(cbm)
                vb.append(vbm)
                md.append(maindetune)
                ld.append(landaudetune)
                if self.deltinsts:
                    del(t)
                    tin.append(None)
                else:
                    tin.append(t)
                print(c)
                
            self.tinsts.append(tin)
            self.bmans.append(bm)
            self.cbms.append(cb)
            self.vbms.append(vb)
            self.maindetune.append(md)
            self.landaudetune.append(ld)

        #return self.sring,rfreq,rs,q,t,tune
                
def boschPlots(babr):

    from pylab import figure

    f = figure()
    ax = f.add_subplot(111)

    lt = np.array([d.ltune for d in babr.tinsts[0]])[:,0]
    ffact = np.array([np.mean(d.formfact,0) for d in babr.tinsts[0]])
    filt = ffact[:,0]==1
    lt[-np.where(np.isnan(lt)[-1::-1])[0][0]] = 0    
    lt[filt] = np.nan
    lt[np.where(np.isnan(lt))[0][0]] = 0    
    detkhz = babr.detune/1e3
    ax.plot(detkhz,lt*babr.sring.frf/babr.sring.nbunch,'-x')
    robfr = babr.omegar[0].real/2/np.pi
    robfr[filt] = np.nan
    bigfr = babr.boschd[0].real/2/np.pi
    bigfr[filt] = np.nan
    
    ax.plot(detkhz,robfr,'-x')
    ax.plot(detkhz,bigfr,'-x')

    f = figure()
    ax = f.add_subplot(111)

    growt = -1/babr.boschd[0].imag
    growt[filt] = np.nan
    ax.plot(detkhz,growt*1e3,'-x')

    f = figure()
    ax = f.add_subplot(111)
    ax.plot(detkhz,1/babr.venturini[0].real*1e3,'-x')

    f = figure()
    ax = f.add_subplot(111)
    ax.plot(detkhz,babr.venturini[0].real/1e3,'-x')    

def boschDipoleQuadrupole(babr):

    from pylab import figure

    lt = np.array([d.ltune for d in babr.tinsts[0]])[:,0]
    ffact = np.array([np.mean(d.formfact,0) for d in babr.tinsts[0]])
    filt = ffact[:,0]==1
    detkhz = babr.detune/1e3    
    
    bigfr = babr.boschd[0]
    bigfr[filt] = np.nan+1j*np.nan
    bigfrq = babr.boschq[0]
    bigfrq[filt] = np.nan+1j*np.nan
    cplefr = babr.boschcoupled[0]
    cplefr[filt] = np.nan
    cplefrq = babr.boschcoupleq[0]
    cplefrq[filt] = np.nan

    f = figure()
    ax = f.add_subplot(111)    
    
    ax.plot(detkhz,bigfr.real/2/np.pi,'-x',zorder=2)
    ax.plot(detkhz,bigfrq.real/2/np.pi,'-x',zorder=2)

    ax.plot(detkhz,cplefr.real/2/np.pi,'--b',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplefrq.real/2/np.pi,'--g',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplefr.real/2/np.pi,'xb',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplefrq.real/2/np.pi,'xg',lw=2,mew=2,ms=8,mfc='none',zorder=1)

    ax.legend(('Dipole','Quadrupole','Coupled\ndipole','Coupled\nquadrupole'),loc=1,borderaxespad=0,ncol=2)
    ax.set_xlabel('HC detuning/kHz')
    ax.set_ylabel(r'Coherent frequency/Hz')

    #Tracer()()
    
    filt = filt | (cplefr[0].imag!=0)
    cplegr = babr.boschcpledgrow[0]
    cplegr[filt] = np.nan
    cplegrq = babr.boschcpleqgrow[0]
    cplegrq[filt] = np.nan

    #Tracer()()

    f = figure()
    ax = f.add_subplot(111)
    
    ax.plot(detkhz,bigfr.imag+1/babr.sring.taue,'-x',zorder=2)
    ax.plot(detkhz,bigfrq.imag+2/babr.sring.taue,'-x',zorder=2)

    ax.plot(detkhz,cplegr.real,'--b',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplegrq.real,'--g',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplegr.real,'xb',lw=2,mew=2,ms=8,mfc='none',zorder=1)
    ax.plot(detkhz,cplegrq.real,'xg',lw=2,mew=2,ms=8,mfc='none',zorder=1)

    ax.legend(('Dipole','Quadrupole','Coupled dipole','Coupled quadrupole'),loc=4,borderaxespad=0)
    ax.set_xlabel('HC detuning/kHz')
    ax.set_ylabel(r'Growth rate/$\rm s^{-1}$')

def plotBoschModeCoupling_scaled(ginst,res,resq):

    from pylab import figure, flatten, setp

    ffact = np.array([i.ffact[1] for i in flatten(ginst.bmans)]).reshape(*ginst.boschcoupled.shape)
    phi2 = np.array([i.phi2 for i in flatten(ginst.bmans)]).reshape(*ginst.boschcoupled.shape)

    unitless = (1/np.cos(phi2.T)**2/ginst.current/1e3/ffact.T).T

    f = figure()
    ax = f.add_subplot(111)
    clr_cycle = 'bgck'
    for n,c,curr in zip(range(len(res)),clr_cycle,ginst.current):
        lab = '%d' % (curr*1e3,)
        ax.plot(unitless[n],res[n]/2./np.pi/1e3,'-'+c,label=lab)
        ax.plot(unitless[n],resq[n]/2/np.pi/1e3,'--'+c)

    ax.set_xlabel(r'$\left[I_b F \cos^2\psi\right]^{-1}$/$\rm mA^{-1}$',labelpad=15)
    ax.set_ylabel('Frequency/kHz')
    flatpot = 2*ginst.cavparams['LC']['cavity']['Rs']/ginst.sring.eloss/1e3*(ginst.nharm*ginst.nharm-1)
    ax.axvline(flatpot,ls='-',color='k')
    ax.annotate('Flat\npotential',(flatpot+0.01,np.amin(res)/4./np.pi/1e3),size=20,va='top')

    art = ax.legend(ax.lines[:2],('Robinson dipole','Robinson quadrupole'),loc=1,borderaxespad=0)
    ax.legend(loc=4,borderaxespad=0,title='Current/mA')
    setp(ax.legend_.get_title(),size=20)
    ax.add_artist(art)
    f.subplots_adjust(bottom=0.15,top=0.925)

def plotBoschModeCoupling(ginst,res,resq):

    from pylab import figure, flatten, setp

    f = figure()
    ax = f.add_subplot(111)
    clr_cycle = 'bgck'
    for n,c,curr in zip(range(len(res)),clr_cycle,ginst.current):
        lab = '%d' % (curr*1e3,)
        ax.plot(ginst.detune/1e3,res[n]/2./np.pi/1e3,'-'+c,label=lab)
        ax.plot(ginst.detune/1e3,resq[n]/2/np.pi/1e3,'--'+c)

    ax.set_xlabel(r'Harmonic cavity detuning/kHz')
    ax.set_ylabel('Frequency/kHz')

    art = ax.legend(ax.lines[:2],('Robinson dipole','Robinson quadrupole'),loc=1,borderaxespad=0)
    ax.legend(loc=4,borderaxespad=0,title='Current/mA')
    setp(ax.legend_.get_title(),size=20)
    ax.add_artist(art)


def detuneScan(*args,**kwargs):

    if 'detunes' in kwargs:
        detunes = kwargs.pop('detunes')
    else:
        detunes = np.arange(45e3,70e3,1e3)
    if 'kick' in kwargs:
        kick = kwargs.pop('kick')
    else:
        kick = 100e-12

    tinst = []
    rinst = []
    time_off = []
    time_off_r = []

    for d in detunes:
        k = list(args[:])
        k.insert(3,d)

        t = transient.Transient(*k,**kwargs)
        try:
            t.runIterations(300,blenskip=5)
        except np.linalg.linalg.LinAlgError:
            print('Error with detuning', d)
        r = transient.Transient(*k,**kwargs)
        r.time_off = kick*np.sin(np.arange(args[0].nbunch)*2*np.pi/args[0].nbunch)
        try:
            r.runIterations(300,blenskip=5)
        except np.linalg.linalg.LinAlgError:
            print('Error with kick and detuning', d)
        if kwargs['blength']==False:
            t.bunchProfile()
            r.bunchProfile()            

        tinst.append(t)
        rinst.append(r)
        time_off.append(t.time_off)
        time_off_r.append(r.time_off)
        
    return tinst,rinst,time_off,time_off_r

def formFactContour(*args,**kwargs):

    if 'delta' in kwargs:
        delta = kwargs.pop('delta')
    else:
        delta = 0.01
    if 'hcind' in kwargs:
        hcind = kwargs.pop('hcind')
    else:
        hcind = -1
        
    ffact = np.arange(0.5,1.0+delta/2,delta)
    fphi = np.arange(-1,1,delta)*np.pi/3.
    res = np.zeros((ffact.shape[0],fphi.shape[0]),complex)

    ac = 'active_cavs' in kwargs
    if not ac: kwargs.update({'active_cavs':[[],[],[]]})

    for i,f in enumerate(ffact):
        for j,p in enumerate(fphi):
            tmp_kwargs = kwargs.copy()
            active_cavs = tmp_kwargs.pop('active_cavs')
            t = transient.Transient(*args,**tmp_kwargs)
            if not ac and t.feedback:
                active_cavs = [[0],[t.vrf],[t.phi_rf]]
            t.setupActiveCavities(*active_cavs)
            #t = transient.EigenTransient(*args,**kwargs)            
            ff0 = f*np.exp(1j*p)
            t.formfact = np.ones((t.sring.nbunch,len(args[2])))*ff0
            try:
                t.runIterations(1,fract=0)
            except np.linalg.linalg.LinAlgError:
                print('Failed for amplitude %.1f and phase %.1f' % (f,p))
            res[i,j] = np.mean(t.formfact[:,hcind])-ff0

    return ffact, fphi, res

def brentOptimise(*args,**kwargs):

    from scipy.optimize import root

    old_bmean = 0
    nind = np.where(np.array(args[2])>1)[0]
    ac = False
    if 'active_cavs' in kwargs:
        ac = True
    else:
        kwargs.update({'active_cavs':[[],[],[]]})
        
    if len(args[2])>1:
        print('Multiple harmonic cavities exist, all except number %d should have shunt impedance 0.' % nind[0])
    nind = nind[0]
    
    def evalBrent(ffact):
        tmp_kwargs = kwargs.copy()
        active_cavs = tmp_kwargs.pop('active_cavs')
        t = transient.Transient(*args,**tmp_kwargs)
        if not ac and t.feedback:
            active_cavs = [[0],[t.vrf],[t.phi_rf]]
        t.setupActiveCavities(*active_cavs)
        t.time_off += np.real(old_bmean)
        ff0 = ffact[0]*np.exp(1j*ffact[1])
        t.formfact = 1*ff0
        t.runIterations(1,fract=0)
        #penalty = np.mean(t.formfact)-ff0
        penalty = (np.absolute(np.mean(t.formfact[:,nind]))-np.absolute(ff0))*np.exp(1j*(np.angle(np.mean(t.formfact[:,nind]))-np.angle(ff0)))
        return np.array([np.absolute(penalty),np.angle(penalty)])

    if 'niterations' in kwargs:
        niter = kwargs.pop('niterations')
    else:
        niter = 1
    if 'fguess' in kwargs:
        fguess = kwargs.pop('fguess')
    else:
        if not 'delta' in kwargs:
            kwargs.update({'delta':0.1})
        kwargs.update({'hcind':nind})
        fguess = formFactContour(*args,**kwargs)
        kwargs.pop('delta')
        kwargs.pop('hcind')

    bestarg = np.argmin(np.absolute(fguess[2]))
    festi = np.array([fguess[0][bestarg/fguess[1].shape[0]],
                      fguess[1][bestarg%fguess[1].shape[0]]])

    for n in range(niter):
        tmp_kwargs = kwargs.copy()
        active_cavs = tmp_kwargs.pop('active_cavs')        
        res = root(evalBrent,festi,method='lm')
        t = transient.Transient(*args,**tmp_kwargs)
        if not ac and t.feedback:
            active_cavs = [[0],[t.vrf],[t.phi_rf]]
        t.setupActiveCavities(*active_cavs)        
        t.time_off += np.real(old_bmean)
        t.formfact = res.x[0]*np.exp(1j*res.x[1])
        t.runIterations(1,fract=0)
        old_bmean = np.mean(t.bmean)
        festi[1] -= old_bmean*t.nharm[nind]*t.sring.frf*2*np.pi
    res.update({'inst':t,'fguess':fguess})

    return res

def fieldModContour(*args,**kwargs):

    if 'delta' in kwargs:
        delta1 = kwargs.pop('delta')
        delta2 = delta1*4
        delta3 = delta1*10
    else:
        delta1 = 0.01
        delta2 = 1
        delta3 = 0.1
        
    if 'delta1' in kwargs:
        delta1 = kwargs.pop('delta1')
    if 'delta2' in kwargs:
        delta2 = kwargs.pop('delta2')
    if 'delta3' in kwargs:
        delta3 = kwargs.pop('delta3')

    harmcavloc = np.where(np.array(args[1])>1)[0][0]
        
    famp = np.arange(0.3,1.0+delta1/2,delta1)*args[0].current*args[1][harmcavloc]*2*np.cos(80/180.*np.pi)**2
    fbw = np.arange(10,100,delta2)
    fphi = -np.arange(70,90,delta3)/180.*np.pi
    res = np.zeros((famp.shape[0],fbw.shape[0],fphi.shape[0]),complex)
    tot_num = len(fbw)*len(fphi)*len(famp)
    iter_count = 0

    for i,f in enumerate(famp):
        for j,p in enumerate(fbw):
            for k,q in enumerate(fphi):
                #t = transient.Transient(*args,**kwargs)
                a = transient.LorentzTransient([f],[p],[q],*args,**kwargs)
                a.bunchProfile()
                a.fullForm()
                t = transient.EigenTransient(*args,**kwargs)
                ff0 = a.formfact[:]
                t.formfact = 1*ff0 
                try:
                    t.runIterations(1,fract=0)
                except np.linalg.linalg.LinAlgError:
                    print('Failed for amplitude %.1f and phase %.1f' % (f,p))
                res[i,j,k] = np.mean(t.formfact)-np.mean(ff0)
                #res[i,j,k] = np.mean(t.formfact-ff0)
                print('Completed iteration %d of %d' % (iter_count,tot_num))
                iter_count += 1

    return famp, fbw, fphi, res

def contour(scans,cls,*args,**kwargs):

    if 'followind' in kwargs:
        followind = kwargs.pop('followind')
    else:
        followind = len(scans)-1
    shpe = [len(s) for s in scans]
    res = np.zeros(shpe,complex)
    tot_num = shpe[followind]

    if np.sum(shpe)==len(shpe):
        a = cls(*(scans+list(args)),**kwargs)
        a.bunchProfile()
        a.fullForm()
        ff0 = a.formfact[:]
        lp0 = a.landau_phasor[:]
        a.__class__ = transient.Transient
        try:
            a.runIterations(1,fract=0)
        except np.linalg.linalg.LinAlgError:
            print('Failed for the following parameters', scans)
        #res = np.mean(a.formfact-ff0)
        res = np.mean(np.absolute(a.landau_phasor)-np.absolute(lp0))*np.exp(1j*np.mean(np.angle(a.landau_phasor)-np.angle(lp0)))
        return res

    n = 0
    infill = res[:]
    while len(scans[n])==1 and n<len(scans)-1:
        infill = infill[0]
        n += 1
    #infill = np.zeros(len(scans[n]),complex)
    for i,s in enumerate(scans[n]):
        scn_tmp = scans[:]
        scn_tmp[n] = [s]
        kwargs.update({'followind':followind})
        infill[i] = contour(scn_tmp,cls,*args,**kwargs)
        if n==followind:
            print('Completed %d of %d for the tracked scanned parameter' % (i+1,tot_num))
    res = infill[:]
    for e in range(n):
        res = np.array([res],complex)
        
    return res

def treeRepeat(*res):
    shpe = np.array([len(a) for a in res])
    tst = np.zeros((np.prod(shpe),len(shpe)))
    for n in range(shpe[0]):
        steplen = np.prod(shpe[1:])
        tst[n*steplen:(n+1)*steplen,0] = res[0][n]
        if tst.shape[1]>1:
            tst[n*steplen:(n+1)*steplen,1:] = treeRepeat(*res[1:])
    return tst

def saveScanResText(res,fname):

    newshape = np.array([len(a) for a in res[:-2]])
    nvar = len(newshape)
    newlen = np.prod(newshape)
    out = res[-1].reshape(newlen,len(res[-2]))
    resarray = np.zeros(np.array(out.shape)+np.array([1,nvar]),complex)

    resarray[1:,:nvar] = treeRepeat(*res[:-2])
    resarray[0,nvar:] = res[-2]
    resarray[1:,nvar:] = out

    np.savetxt(fname,resarray,fmt='%.5e')

    return resarray

def brentModOpti(*args,**kwargs):

    from scipy.optimize import root

    def evalBrent(modphi):
        a = transient.LorentzTransient(modphi[:1],modphi[1:2],modphi[2:],*args,**kwargs)
        a.bunchProfile()
        a.fullForm()
        t = transient.Transient(*args,**kwargs)
        ff0 = a.formfact[:]
        t.formfact = 1*ff0
        try:
            t.runIterations(1,fract=0)
        except np.linalg.LinAlgError:
            pdiff = np.exp(1j*np.pi/2.)
            pen0 = 1.0
            #raise
        else:
            #penalty = np.absolute(np.mean(t.formfact-ff0))*np.exp(1j*np.angle(np.mean(t.formfact-ff0)))
            pdiff = np.mean(t.landau_phasor-a.landau_phasor)
            #pen0 = np.sum(np.absolute(t.bmean-a.bmean)/np.mean(np.absolute(t.bmean)))
            pen0 = np.mean(np.angle(t.landau_phasor))-np.mean(np.angle(a.landau_phasor))
        penalty = np.array([np.absolute(pdiff),
                            np.angle(pdiff),
                            pen0])
        return penalty

    if 'fguess' in kwargs:
        fguess = kwargs.pop('fguess')
    else:
        if not 'delta' in kwargs:
            kwargs.update({'delta':0.1})
        fguess = fieldModContour(*args,**kwargs)
        kwargs.pop('delta')
    fgm = ma.array(fguess[3],mask=np.isnan(fguess[3]))
    bestarg = np.unravel_index(ma.argmin(np.absolute(fgm)),fgm.shape)
    festi = np.array([fguess[n][b] for n,b in enumerate(bestarg)])

    res = root(evalBrent,festi,method='lm')
    a = transient.LorentzTransient(res.x[:1],res.x[1:2],res.x[2:],*args,**kwargs)
    a.bunchProfile()
    a.fullForm()
    res.update({'inst':a})

    return res

def semiFlat(rs,qfact,nharm,s,formfact=1):

    absffact = np.absolute(formfact)
    angffact = np.angle(formfact)
    evby2irs = s.vrf/(2.*s.current*rs*absffact)
    ubyirs = s.eloss/(s.current*rs*absffact)
    ubyv = s.eloss/s.vrf
    a = (1-nharm*nharm)*evby2irs
    b = nharm*nharm+ubyirs
    c = ubyv*ubyv-1

    k = np.sqrt((-b+np.sqrt(b**2-4*a*c))/2./a)
    #psi = np.arccos(k*s.vrf/(2.*s.current*rs))
    psi = np.arccos(k*evby2irs+angffact)
    detune = np.tan(psi)/2./qfact*nharm*s.frf

    return detune

def plotTransientVsDetune(detune,exinst,sring,trans_t,trans_r,kick=100e-12):

    from pylab import figure

    transients = np.array([np.amax(i)-np.amin(i) for i in trans_t])
    transient_r = np.array([np.amax(i)-np.amin(i) for i in trans_r])

    f = figure()
    ax = f.add_subplot(111)
    ax.plot(detune/1e3,transients*1e12,'-x')
    ax.plot(detune/1e3,transient_r*1e12,'-x')

    oneflat = semiFlat(exinst.rs[0],exinst.qfact[0],exinst.nharm[0],sring)/1e3
    ax.axvline(oneflat,ls='--',lw=2,color='k')
    ax.annotate('Semiflattened\npotential',(oneflat-0.5,1200),va='top',ha='right',fontsize=20)
    kicklabel = '%d ps mode 1 kick' % (kick*1e12)
    ax.legend(('No kick',kicklabel),loc=3,borderaxespad=0)
    
    ax.set_xlabel('LC Detuning/kHz')
    ax.set_ylabel('Size of transient/ps')

def plotProfiles(inst,bunches):

    from pylab import figure, setp

    f = figure()
    ax = f.add_subplot(111)
    
    for b in bunches:
        ax.plot(1e12*inst.time[:,b],inst.dist[:,b]/1e12)

    ax.legend([str(b) for b in bunches],title='Bunch number',borderaxespad=0)
    ax.set_xlabel('Time/ps')
    ax.set_ylabel(r'Normalised charge density/$\rm ps^{-1}$')
    setp(ax.legend_.get_title(),size=20)
    f.subplots_adjust(left=0.2,right=0.95)

    

    
