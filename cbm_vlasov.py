import numpy as np
import cbm_solver, transient
from scipy import special, optimize
#from IPython.core.debugger import Tracer

class VlasovSolve(cbm_solver.CbmTdmSolve):
    """
    Class for determining complex coherent frequencies based on an approach derived starting from the Vlasov equation rather
    than a differential equation based on Newton's law.
    """

    def constructMatrix(self,wrobbar=None,order=1):

        if wrobbar==None:
            wrobbar = np.mean(self.wrob)

        denomp = (1-np.exp((1j*(wrobbar-self.omegar)-self.alpha)/self.revfreq))
        denomm = (1-np.exp((1j*(wrobbar+self.omegar)-self.alpha)/self.revfreq))
        #self.wake_sump = self.alpha/2./denomp+1j*self.omegar/2./denomp
        #self.wake_summ = self.alpha/2./denomm-1j*self.omegar/2./denomm
        self.wake_sump = 1/denomp
        self.wake_summ = 1/denomm

        self.ffact = self.calcFormFact(self.omegar,magnitude=not self.tinst.full)#[:,filt]
        self.ffphi = np.angle(self.ffact)
        self.ffact = np.absolute(self.ffact)

        self.amp = self.sring.current*np.outer(self.tinst.bcurr,self.rs*self.alpha/self.omegar)*self.ffact
        self.deltaphi = np.outer(1j*self.omegar+self.alpha,self.tinst.time_off-np.arange(self.sring.nbunch)/self.sring.frf)
        self.turn_back = np.array([np.exp(-(1j*o+a)/self.revfreq)*np.tri(self.sring.nbunch)+np.tri(self.sring.nbunch,k=-1).T for o,a in zip(self.omegar,self.alpha)]).transpose(0,2,1)
        self.phase_mat = np.array([np.outer(np.exp(d),np.exp(-d))*t for d,t in zip(self.deltaphi,self.turn_back)])
        self.ffphi_mat = np.array([np.outer(np.exp(1j*f),np.exp(-1j*f)) for f in self.ffphi.T])

        self.phase_mat = self.phase_mat#*self.ffphi_mat
        self.phase_matc = self.phase_mat.conj()
        
        omegarsigmar = np.outer(self.tinst.blen,self.omegar)**2
        self.integral = special.iv(order,omegarsigmar)*np.exp(-omegarsigmar)
        const = order*self.sring.alphac/self.sring.energy*2

        self.matrix = 1j/2.*(np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.wake_sump-self.phase_matc.transpose(1,2,0)*self.amp*self.wake_summ).transpose(1,0,2)*self.integral/self.wrob/self.tinst.blen**2,2)).T*const

    def constructMatrixKrinsky(self,wrobbar=None,order=1,numericint=False,dispersion=False):

        if wrobbar==None:
            wrobbar = np.mean(self.wrob)
            
        denomp = (1-np.exp((1j*(wrobbar-self.omegar)-self.alpha)/self.revfreq))
        denomm = (1-np.exp((1j*(wrobbar+self.omegar)-self.alpha)/self.revfreq))
        self.dwake_sump = (self.alpha+1j*self.omegar)**(2*order-1)/2./denomp
        self.dwake_summ = (self.alpha-1j*self.omegar)**(2*order-1)/2./denomm

        #filt = np.zeros(len(self.omegar),bool)
        #for m in self.modeinds:
        #    filt[m] = True
        #ffact = self.tinst.formfact[:,filt]
        self.ffact = self.calcFormFact(self.omegar,magnitude=not self.tinst.full)#[:,filt]
        self.ffphi = np.angle(self.ffact)
        self.ffact = np.absolute(self.ffact)
        #ffact = np.absolute(self.tinst.formfact[:,filt])
        
        self.amp = self.sring.current*2/self.revfreq*np.outer(self.tinst.bcurr,self.rs*self.alpha)*self.ffact
        self.deltaphi = np.outer(1j*self.omegar+self.alpha,self.tinst.time_off-np.arange(self.sring.nbunch)/self.sring.frf)
        self.turn_back = np.array([np.exp(-(1j*o+a)/self.revfreq)*np.tri(self.sring.nbunch)+np.tri(self.sring.nbunch,k=-1).T for o,a in zip(self.omegar,self.alpha)]).transpose(0,2,1)
        self.phase_mat = np.array([np.outer(np.exp(d),np.exp(-d))*t for d,t in zip(self.deltaphi,self.turn_back)])
        self.ffphi_mat = np.array([np.outer(np.exp(1j*f),np.exp(-1j*f)) for f in self.ffphi.T])
        
        #deltaphi_s = np.outer(1j*wrobbar,self.tinst.time_off+np.arange(self.sring.nbunch)/self.sring.frf)
        #turn_back_s = np.exp(-(1j*wrobbar)/self.revfreq)*np.tri(self.sring.nbunch)+np.tri(self.sring.nbunch,k=-1).T
        #phase_mat_s = np.outer(np.exp(deltaphi_s),np.exp(-deltaphi_s))*turn_back_s

        ##tst = np.ones(self.phase_mat.shape)-np.eye(self.sring.nbunch)+np.exp((1j*self.omegar[0]-self.alpha[0])/self.revfreq)*np.eye(self.sring.nbunch)
        self.phase_mat = self.phase_mat#*self.ffphi_mat
        self.phase_matc = self.phase_mat.conj()#*phase_mat_s*turn_back_s
        dispfact = 1.0

        if numericint:
            #integral = self.integral(order=order)[-1][88]*self.wrob
            integral = self.numint[order-1]*self.wrob
            const = 1/self.sring.energy*self.revfreq*4*np.pi#/(2*order)
        if dispersion:
            integral = 1.0
            dispfact = 0.0
            const = self.revfreq/self.sring.energy*4*np.pi
        else:
            integral = self.tinst.blen**(2*(order-1))
            const = self.sring.alphac/self.sring.energy*self.revfreq+0*1j
                
        if self.omegasapprox:
            self.matrix = -(-1)**order*(np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sump+self.phase_matc.transpose(1,2,0)*self.amp*self.dwake_summ).transpose(1,0,2)*self.ffact,2)/self.wrob.real/(2*order)*self.tinst.blen**(2*(order-1))).T*const+dispfact*order*self.wrob*np.eye(self.tinst.nbunch,dtype=complex)+0*wrobbar*np.eye(self.tinst.nbunch,dtype=complex)
        else:
            self.matrix = -(-1)**order*(np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sump+self.phase_matc.transpose(1,2,0)*self.amp*self.dwake_summ).transpose(1,0,2)*self.ffact,2)*integral).T*const+dispfact*(order*self.wrob)**2*np.eye(self.tinst.nbunch,dtype=complex)

    def dispersionIntegralTarget(self,target):
        
        def penalty(x):
            integ = 1/self.integral(x[0])[2]
            return np.array([(np.absolute(integ)-np.absolute(target)),np.angle(integ)-np.angle(target)])
        opt = optimize.fmin(penalty,np.array([eigval]),method='lm')

        return opt[0]

    def dispersionIntegralSweep(self,imoff=0,order=2):
        
        normReOmega = np.arange(-2.5,2.6,0.5)*np.mean(self.wrob)/5.
        res = np.zeros(len(normReOmega),complex)
        imoff_const = 0.00005
        
        for i,n in enumerate(normReOmega):
            res[i] = 1/self.integral(n+1j*(imoff_const+imoff),order=order)[2]

        return res

    def integral(self,dispersion=False,samples=1000,order=2,omegasapprox=False):

        hamvals = np.arange(1,samples+0.5,1,float)/samples*(5*self.tinst.sring.espread)**2/2
        deltaE = np.sqrt(2*hamvals)
        
        area = np.zeros((self.tinst.sring.nbunch,samples),complex)
        fs = np.zeros((self.tinst.sring.nbunch,samples),complex)
        for n in range(self.tinst.sring.nbunch):
            area[n], fs[n] = transient.getFsVariableTransform(self.tinst,deltaE,n,area=True)[-2:]
            break
            if dispersion: break

        self.numint = []
        omegas = 2.*np.pi*self.tinst.sring.frf*np.real(fs)/self.tinst.sring.nbunch+1j*0
        if dispersion:
            norm = np.trapz(np.exp(-hamvals/self.tinst.sring.espread**2)/np.real(fs[0]),x=hamvals)            
            n = order
            norm = norm/self.tinst.sring.frf*self.tinst.sring.nbunch
            omegas = omegas/self.tinst.sring.frf*self.tinst.sring.nbunch/2./np.pi
            print(norm)
            universal = (self.tinst.sring.alphac*np.real(area)/omegas[0]/np.pi/4.)**n/self.tinst.sring.espread**2*np.outer(1/norm,np.exp(-hamvals/self.tinst.sring.espread**2))+1j*0
            if omegasapprox:
                integral = np.trapz(universal[0]/2/n/(dispersion-n*omegas[0]),x=hamvals)
            else:
                integral = np.trapz(universal[0]/omegas[0]/(dispersion**2/omegas[0]/omegas[0]-n*n),x=hamvals)#/self.tinst.sring.alphac
        else:
            norm = np.trapz(np.exp(-hamvals/self.tinst.sring.espread**2)/np.real(fs),x=hamvals)            
            for n in range(2):
                universal = (self.tinst.sring.alphac*np.real(area)/omegas/self.tinst.sring.nbunch/2./np.pi/np.pi/4.)**(n+1)/self.tinst.sring.espread**2*np.outer(1/norm,np.exp(-hamvals/self.tinst.sring.espread**2))                
                integral = np.trapz(universal,x=hamvals)/self.tinst.sring.alphac
                self.numint.append(integral)

        return area, fs, integral

def theIntegral(cbmv,dispersion=1000,samples=1000,order=1,omegasapprox=False):

    #deltaE = np.arange(1,samples+0.5,1,float)/samples*5*cbmv.sring.espread    
    #hamvals = deltaE**2/2.
    
    hamvals = np.arange(1,samples+0.5,1,float)/samples*(5*cbmv.sring.espread)**2/2
    deltaE = np.sqrt(2*hamvals)

    alphac = cbmv.tinst.sring.alphac
    blen = cbmv.tinst.blen[0]
    espread = cbmv.tinst.sring.espread    
        
    area, fs = transient.getFsVariableTransform(cbmv.tinst,deltaE,0,area=True)[-2:]
    area = (np.real(area)+1j*0)/2/np.pi/cbmv.tinst.sring.frf
    #fs = np.real(fs)+1j*0
    omegas = 2.*np.pi*np.real(fs)+0*1j
    #omegas = np.polyval(np.polyfit(deltaE,omegas,3),deltaE)+1j*0
    norm = np.trapz(np.exp(-hamvals/cbmv.sring.espread**2)/np.real(fs[0])/alphac,x=hamvals)

    const = 2*espread*4*np.pi/blen*3e8/blen**(2*order-2)#*(-1)**order

    if omegasapprox:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/espread**2)/espread**2/norm/2/order/omegas/(order*omegas-dispersion),x=hamvals)#*alphac/np.pi/np.pi
    else:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/espread**2)/espread**2/norm/(order*order*omegas**2-dispersion**2),x=hamvals)#*alphac/np.pi/np.pi

    return area, fs, integral

def myIntegral(cbmv,dispersion=1000,samples=1000,order=1,omegasapprox=False):
    
    alphac = cbmv.tinst.sring.alphac
    
    hamvals = np.arange(1,samples+0.5,1,float)/samples*(5*cbmv.sring.espread)**2/2*alphac
    deltaE = np.sqrt(2*hamvals/alphac)

    blen = cbmv.tinst.blen[0]
    espread = cbmv.tinst.sring.espread    
        
    area, fs = transient.getFsVariableTransform(cbmv.tinst,deltaE,0,area=True)[-2:]
    area = (np.real(area)+1j*0)/2/np.pi/cbmv.tinst.sring.frf
    #area = deltaE**2*alphac/2/np.real(fs)
    #fs = np.real(fs)+1j*0
    omegas = 2.*np.pi*np.real(fs)+0*1j
    #omegas = np.polyval(np.polyfit(deltaE,omegas,3),deltaE)+1j*0
    norm = np.trapz(np.exp(-hamvals/alphac/cbmv.sring.espread**2)/np.real(fs[0]),x=hamvals)

    const = 4*np.pi/cbmv.sring.alphac

    if omegasapprox:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/alphac/espread**2)/alphac/espread**2/norm/2/order/(order*omegas-dispersion),x=hamvals)#*alphac/np.pi/np.pi
    else:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/alphac/espread**2)/alphac/espread**2/norm*omegas/(order*order*omegas**2-dispersion**2),x=hamvals)#*alphac/np.pi/np.pi

    return area, fs, integral

def venturIntegral(cbmv,dispersion=1000,samples=1000,order=1,omegasapprox=False):
    
    alphac = cbmv.tinst.sring.alphac
    espread = cbmv.tinst.sring.espread
    blen = cbmv.tinst.blen[0]    

    deltaE = np.arange(1,samples+0.5,1,float)/samples*5*espread
        
    area, fs = transient.getFsVariableTransform(cbmv.tinst,deltaE,0,area=True)[-2:]
    area = (np.real(area)+1j*0)/2/np.pi/cbmv.tinst.sring.frf
    #area = deltaE**2*alphac/2/np.real(fs)
    #fs = np.real(fs)+1j*0
    omegas = 2.*np.pi*np.real(fs)+0*1j
    hamvals = deltaE*deltaE*alphac/2
    tauhat = deltaE*alphac/omegas
    
    #omegas = np.polyval(np.polyfit(deltaE,omegas,3),deltaE)+1j*0
    norm = np.trapz(np.exp(-hamvals/alphac/cbmv.sring.espread**2)/np.real(fs[0]),x=hamvals)

    const = 4*np.pi/cbmv.sring.alphac

    if omegasapprox:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/alphac/espread**2)/alphac/espread**2/norm/2/order/(order*omegas-dispersion)*omegas*omegas*tauhat/alphac,x=tauhat)#*alphac/np.pi/np.pi
    else:
        integral = np.trapz(const*(alphac*area/4./omegas/np.pi)**order*np.exp(-hamvals/alphac/espread**2)/alphac/espread**2/norm*omegas/(order*order*omegas**2-dispersion**2)*omegas*omegas*tauhat/alphac,x=tauhat)#*alphac/np.pi/np.pi

    return area, fs, integral
