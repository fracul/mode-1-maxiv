import numpy as np
import bunch_length
#from IPython.core.debugger import Tracer
import utility

class Transient:
    """
    Class for determining the phase slippage and bunch profiles in the presence of inhomogeneous beam loading
    """

    def __init__(self,sring,rs,nharm,detune,qfact,fill=0.75,blength=False,feedback=False,formcalc='',complex256=False):
        """
        Class initialisation. 

        Arguments:
        *sring*   - utility.StorageRing instance (loaded from file)
        *rs*      - list or array of shunt impedances for resonators loaded by the beam
        *nharm*   - list or array of the nearest RF harmonics to each resonator (can be 0)
        *detune*  - list or array of detunings from the nearest RF harmonic (if *nharm* is 0, the resonant frequency of the resonator)
        *qfact*   - list or array of quality factors of each resonator

        Optional keyword arguments:
        *fill*       - [default 0.75] fill of the machine: can either be a number less than or equal to 1 being the fraction of machine RF buckets that are full
        *blength*    - [default False] account for the finite lengths of the electron bunches using form factors
        *feedback*   - [default False] include feedback for the main RF (useful to account for beam loading of the main cavities, which is assumed to be the first resonator)
        *complex256* - [default False] extend data type to more bits (for cases where float precision is not enough)
        *formcalc*   - [default: ''] which type of form-factor calculation to use, can be either 'scalar' or 'full' for scalar or complex form factor respectively. If neither,
                       and *blength* is True, then calculate the form factors from the gradient of the RF voltage assuming Gaussian bunches. 

        After initialisation, the next step towards results is typically to run the class method 'runIterations'.
        """

        if isinstance(rs,float):
            rs = [rs]
            nharm = [nharm]
            detune = [detune]
            qfact = [qfact]
        self.rs = np.array(rs)
        self.nharm = np.array(nharm)
        detune = np.array(detune)
        qfact = np.array(qfact)
        self.iterno = 0
        
        self.sring = sring
        self.omegar = 2*np.pi*(self.nharm*self.sring.frf+detune)
        self.nharm[self.nharm==0] = detune[self.nharm==0]/self.sring.frf
        self.revfreq = self.sring.frf/self.sring.nbunch
        self.alpha = self.omegar*0.5/qfact
        self.genphase = np.arctan(qfact*(self.omegar/(self.nharm*self.sring.frf*2*np.pi)-2*np.pi*self.sring.frf*self.nharm/self.omegar))
        self.feedback = feedback      
        self.phi_0 = np.arcsin(self.sring.eloss/self.sring.vrf+np.sum(2*self.sring.current*self.rs[self.nharm!=1]*np.cos(self.genphase[self.nharm!=1])**2)/self.sring.vrf)
        self.scalar = formcalc=='scalar'
        self.full = formcalc=='full'
        self.qfact = qfact
        self.complex256 = complex256
        #else: self.phi_0 = np.arcsin(self.sring.eloss/self.sring.vrf+np.sum(2*self.sring.current*self.rs*genphase**2/self.sring.vrf))
        #self.omegar = -self.omegar
        
        if isinstance(fill,float):
            self.fill = np.zeros(self.sring.nbunch) 
            self.fill[:int(fill*len(self.fill))] = 1.
        else: self.fill = fill
        #self.fill = self.fill[-1::-1]

        #self.deltaphi = np.outer((1j*self.omegar+self.alpha),np.arange(self.sring.nbunch,dtype=complex)[self.fill!=0])/self.sring.frf
        self.deltaphi = np.outer(1j*self.omegar+self.alpha,-np.arange(self.sring.nbunch))/self.sring.frf
        self.nbunch = self.deltaphi.shape[-1]
        self.time_off = np.zeros(self.nbunch)
        #self.bcurr = self.fill[self.fill!=0]/np.sum(self.fill!=0)
        #self.bcurr = self.fill/np.sum(self.fill!=0)
        self.bcurr = self.fill/np.sum(self.fill)
        self.blength = blength
        self.vrf = 1*self.sring.vrf
        self.phi_rf = 1*self.phi_0
        
        #if self.feedback:
        #    self.phi_0 = self.phi_0-np.mean(self.genphase[self.nharm==1])
        #    vb = np.sum(2*self.sring.current*self.rs[self.nharm==1]*np.cos(self.genphase[self.nharm==1]))
        #    #vre = vb*np.cos(np.pi-np.mean(genphase[nharm==1]))+self.sring.vrf*np.sin(self.phi_0)
        #    #vim = vb*np.sin(np.pi-np.mean(genphase[nharm==1]))+self.sring.vrf*np.cos(self.phi_0)
        #    vre = vb+self.sring.vrf*np.sin(self.phi_0)
        #    vim = self.sring.vrf*np.cos(self.phi_0)            
        #    self.vrf = np.sqrt(vre**2+vim**2)#/np.cos(np.mean(genphase[nharm==1]))
        #    self.phi_rf = np.arctan2(vim,vre)
        #    #self.vrf = np.sqrt(vb**2+self.sring.vrf**2+2*vb*self.sring.vrf*np.sin(self.phi_0+np.mean(genphase[nharm==1])))/np.cos(np.mean(genphase[nharm==1]))
        #    #self.phi_rf = -self.phi_0 + np.arccos(vb/self.vrf*np.cos(self.phi_0+np.mean(genphase[nharm==1])))-np.mean(genphase[nharm==1])#/np.cos(np.mean(genphase[nharm==1])))-np.mean(genphase[nharm==1])

        self.activeCavs = []
        if self.feedback:
            self.activeCavs = [ActiveCavity(self.vrf,self.phi_rf,0,2*np.pi*self.sring.frf)]

        if self.blength:
            self.formfact = np.exp(-(self.sring.blen*self.omegar)**2/2.)
        else:
            self.formfact = 1.
        self.blen = np.ones(self.nbunch)*self.sring.blen
        self.phase_diff = 0
        
        bunch_length.init_param(self)

    def setupActiveCavities(self,indices,kfrac,phi):
        if self.feedback:
            self.activeCavs[0].index = indices[0]
            self.activeCavs[0].ph_design = phi[0]
            
        for i,k,p in zip(indices[self.feedback:],kfrac[self.feedback:],phi[self.feedback:]):
            self.addActiveCav(i,k*self.vrf,p)

    def addActiveCav(self,index,v_design,phi_design):
        self.activeCavs.append(ActiveCavity(v_design,phi_design,index,2*np.pi*self.nharm[index]*self.sring.frf))

    def changeNBunch(self,s):

        tmplte = np.ones(s.nbunch)
        self.nbunch = s.nbunch
        self.revfreq = self.revfreq/s.nbunch*self.sring.nbunch
        self.fill = tmplte[:]
        self.bcurr = self.fill/np.sum(self.fill)

        self.blen = tmplte*self.blen[0]
        self.formfact = np.outer(tmplte,self.formfact[0])
        self.landau_phasor = np.outer(tmplte,self.landau_phasor[0])
        self.mainrf_phasor = np.outer(tmplte,self.mainrf_phasor[0])
        self.hamilton = np.outer(self.hamilton[:,0],tmplte)
        self.mainrf = np.outer(self.mainrf[:,0],tmplte)
        self.dist = np.outer(self.dist[:,0],tmplte)
        self.time = np.outer(self.time[:,0],tmplte)
        self.time_off = self.time_off[0]*tmplte
        #self.ltune = np.ones(s.nbunch)

        self.sring = s

    def constructMatrix(self):

        self.turn_back = np.array([np.exp(-(1j*o+a)/self.revfreq)*np.tri(self.nbunch)+np.tri(self.nbunch,k=-1).T for o,a in zip(self.omegar,self.alpha)]).transpose(0,2,1)

        #self.dwake_sum = (-1j*self.omegar-self.alpha)/(1-np.exp(-(1j*self.omegar+self.alpha)/self.revfreq))**2
        self.dwake_sum = (-1j*self.omegar-self.alpha)/(1-np.exp(-(1j*self.omegar+self.alpha)/self.revfreq))
        self.cwake_sum = 1./(1-np.exp(-(1j*self.omegar+self.alpha)/self.revfreq))        
        self.phase_mat = np.array([np.outer(np.exp(d),np.exp(-d))*t for d,t in zip(self.deltaphi,self.turn_back)])

        self.cvolt = (not self.feedback)*self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*self.time_off)
        self.dvolt = (not self.feedback)*self.vrf*2*np.pi*self.sring.frf*np.cos(self.phi_rf+2*np.pi*self.sring.frf*self.time_off)

        self.amp = self.sring.current*2/self.revfreq*np.outer(self.bcurr,self.rs*self.alpha)*self.formfact
        self.cwake = (np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum).T
        self.dwake = (np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum*(1j*self.omegar+self.alpha)).T
        #self.dwake = np.dot(np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sum).transpose(2,0,1),0).real,self.time_off)
        for a in self.activeCavs:
            v,phi = a.calcActiveParams(self.cwake[a.index]+0.5*self.amp[:,a.index].real,self.dwake[a.index])
            self.cvolt += v*np.sin(phi+a.omega*self.time_off)#-self.phase_diff)
            #self.dvolt += a.v_design*a.omega*np.cos(a.ph_design+a.omega*self.time_off)#-self.phase_diff)            
            self.dvolt += v*a.omega*np.cos(phi+a.omega*self.time_off)#-self.phase_diff)
        
        #self.lhs = np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sum).transpose(2,0,1),0).real-self.dvolt*np.eye(self.nbunch)
        #self.rhs = -(np.sum((np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum).T,0)).real+(self.cvolt-self.sring.eloss)*np.ones(self.nbunch)-(0.5*np.sum(self.amp,1)).real

        self.lhs = np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sum).transpose(2,0,1),0).real-self.dvolt*np.eye(self.nbunch)
        self.rhs = -(np.sum((np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum).T,0)).real+(self.cvolt-self.sring.eloss)*np.ones(self.nbunch)-(0.5*np.sum(self.amp,1)).real
        
        #self.turn_back = np.exp(-(1j*self.omegar+self.alpha)/self.revfreq)*np.tri(self.nbunch)+np.tri(self.nbunch,k=-1).T        
        #self.phase_mat = np.outer(np.exp(self.deltaphi),np.exp(-self.deltaphi))*self.turn_back        
        #self.amp = self.bcurr*self.sring.current*2/self.revfreq        
        #self.lhs = np.dot(self.phase_mat,self.dwake_sum).real*self.amp-self.dvolt*np.eye(self.nbunch)        
        #self.rhs = -(np.sum(self.cwake_sum*np.sum(self.phase_mat,2),0)).real*self.amp+(self.cvolt-self.sring.eloss)*np.ones(self.nbunch)-0.5*self.amp#+self.dvolt*self.time_off
        #+self.dvolt*self.time_off
        #self.rhs = -(self.cwake_sum*np.sum(self.phase_mat,1)).real*self.amp+(self.cvolt-self.sring.eloss)*np.ones(self.nbunch)-0.5*self.amp
        #-np.dot(self.lhs,self.time_off)
        #self.lhs += -self.dvolt*np.eye(self.nbunch)

    def bunchLengthening(self):
        dvdt = self.dvolt+np.sum(np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum*(-1j*self.omegar-self.alpha),1)
        self.ltune = np.sqrt(self.sring.alphac/self.sring.energy*self.revfreq*dvdt.real)/2./np.pi/self.revfreq
        if self.blength:
            if self.scalar or self.full:
                ######if (self.blen==self.sring.blen).all(): self.blen[~np.isnan(self.ltune)] = self.sring.ltune/self.ltune[~np.isnan(self.ltune)]*self.sring.blen
                self.bunchProfile()
                if self.scalar: self.scalarForm()
                else: self.fullForm()
                #self.formFactTO()
            else:
                self.blen[~np.isnan(self.ltune)] = self.sring.ltune/self.ltune[~np.isnan(self.ltune)]*self.sring.blen
                self.formfact = np.exp(-np.outer(self.blen,self.omegar)**2/2.)
            self.formfact[np.isnan(self.formfact)] = 1.0

    def frequencyLoop(self,target):

        if hasattr(self,'landau_phasor'):
            fieldamp = self.landau_phasor[:]
        else:
            fieldamp = (np.sum(np.transpose(self.phase_mat,(1,2,0))*self.amp*self.cwake_sum,1)-0.5*self.amp.real).T[0]

        mainrffilt = self.nharm!=1
        newcos = target/np.absolute(fieldamp)*np.cos(self.genphase[mainrffilt])
        newomegar = (np.sqrt(1-newcos*newcos)/newcos+2*self.qfact[mainrffilt])*self.frf*2*np.pi*self.nharm[mainrffilt]/2*self.qfact[mainrffilt]
        self.omegar[mainrffilt] = newomegar
        self.alpha[mainrffilt] = self.omegar[mainrffilt]*0.5/self.qfact[mainrffilt]

    def feedbackRF(self):
        
        if not hasattr(self,'cwake_0'):
            self.dwake_0 =  (-1j*self.omegar[0]-self.alpha[0])/(1-np.exp(-(1j*self.omegar[0]+self.alpha[0])/self.revfreq))**2
            self.cwake_0 = 1./(1-np.exp(-(1j*self.omegar[0]+self.alpha[0])/self.revfreq))
            self.deltaphi_0 = (1j*self.omegar[0]+self.alpha[0])*np.arange(self.sring.nbunch,dtype=complex)/self.sring.frf            
            #self.deltaphi_0 = (1j*self.omegar[0]+self.alpha[0])*np.arange(self.sring.nbunch,dtype=complex)[self.fill!=0]/self.sring.frf
            self.dvolt_0 = self.sring.vrf*2*np.pi*self.sring.frf*(np.cos(self.phi_0)-1j*np.sin(self.phi_0))
            self.cvolt_0 = self.sring.vrf*(np.sin(self.phi_0)+1j*np.cos(self.phi_0))
            qfact = 2*self.omegar[0]/self.alpha[0]
            self.genphase_0 = np.arctan(qfact*(self.omegar[0]/(self.sring.frf*2*np.pi)-2*np.pi*self.sring.frf/self.omegar[0]))
            self.phase_diff = 0
        
        #dvdt = self.dvolt+(-1j*self.omegar-self.alpha)*np.sum(self.cwake_sum*np.sum(self.phase_mat*self.amp,2),0)
        #dvdt = self.dvolt+np.sum(np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum*(-1j*self.omegar-self.alpha),1)
        #dvdt = self.dvolt+np.sum(self.phasemat_0*self.amp,1).T
        #dev = np.dot((self.phasemat_0*self.dwake_0).real*self.amp,self.time_off)-self.dvolt_0
        turn_back = self.turn_back[0].copy()
        turn_back[self.time_off<0,self.time_off<0] /= self.turn_back[0,0,0]
        self.phasemat_0 = np.outer(np.exp(self.deltaphi_0),np.exp(-self.deltaphi[0]))*turn_back#

        #self.cvolt_0 = self.sring.vrf*(np.sin(self.phi_0-self.phase_diff)+1j*np.cos(self.phi_0-self.phase_diff))
        
        #dvdt = self.dvolt_0+np.sum(self.phasemat_0*self.amp[:,0],1)*self.cwake_0*(-1j*self.omegar[0]-self.alpha[0])
        #const = self.cwake_0*np.sum(self.phasemat_0*self.amp[:,0],1)+self.cvolt_0+self.dwake_0*np.dot(self.phasemat_0*self.amp[:,0],self.time_off)#+self.dvolt_0*self.time_off

        #self.const_store = const
        #self.dvdt_store = dvdt

        #self.phi_rf = np.arctan2(2*np.pi*self.sring.frf*np.mean(const.real),np.mean(dvdt.real))#+self.phase_diff
        #self.vrf = np.sqrt(np.mean(dvdt.real)**2/(2.*np.pi*self.sring.frf)**2+np.mean(const.real)**2)
        #self.phi_rf = np.angle(np.mean(const))+self.phase_diff
        #self.vrf = np.mean(np.absolute(const))

        #self.vrf = np.mean(np.absolute(const))
        #self.phi_rf = np.mean(np.angle(const))

        #hamilton = self.sring.vrf*1j*np.exp(-1j*(self.phi_rf+2*np.pi*self.sring.frf*self.time_off))+np.sum(self.phase_mat[0]*self.amp[:,0],1)*self.cwake_0
        hamilton = -self.sring.vrf*1j*np.exp(1j*(self.phi_rf))+np.sum(self.phasemat_0*self.amp[:,0],1)*self.cwake_0

        self.vrf = np.mean(np.absolute(hamilton))

    def evaluateResult(self,**kwargs):
        
        time_off = np.dot(np.linalg.pinv(self.lhs,**kwargs),self.rhs)
        
        return time_off

    def runIterations(self,niterations,fract=1,blenskip=1):
        """
        Execute iterations to determine the final outcome of the calculation of equilibrium time offsets and, if requested, bunch lengths and profiles.

        Arguments:
        *niterations* - Number of iterations to run

        Optional keyword arguments:
        *fract*    - [default: 1] Fraction of solution to apply each iteration, can be useful for 
        *blenskip* - [default: 1] Skip this number of iterations between calculating bunch lengths and profiles
        """

        for n in range(niterations):
            self.constructMatrix()
            self.time_off += fract*self.evaluateResult()
            ###self.deltaphi = np.outer(1j*self.omegar+self.alpha,self.time_off+np.arange(self.sring.nbunch)[self.fill!=0]/self.sring.frf)
            ##self.time_off = self.time_off-np.mean(self.time_off)
            #if self.feedback:
            #    self.phase_diff = 2*np.pi*self.sring.frf*np.mean(self.time_off)                
            #    self.time_off -= np.mean(self.time_off)
            self.deltaphi = np.outer(1j*self.omegar+self.alpha,self.time_off-np.arange(self.sring.nbunch)/self.sring.frf)
        
            #if self.feedback:
            #    self.feedbackRF()
            #    self.phase_diff = -2*np.pi*self.sring.frf*np.mean(self.time_off)
            #    self.phi_rf -= self.phase_diff
            #    self.time_off -= np.mean(self.time_off)
            #    self.deltaphi = np.outer(1j*self.omegar+self.alpha,self.time_off-np.arange(self.sring.nbunch)/self.sring.frf)
            if self.blength:
                if (self.full | self.scalar):
                    if (self.iterno%blenskip==0): self.bunchLengthening()
                else: self.bunchLengthening()
            self.iterno += 1
        
    def evaluateResultSympy(self,**kwargs):

        import sympy

        g = sympy.Matrix(self.lhs)
        return np.dot(np.asarray(g.inv()).astype(float),self.rhs)

    def getCavfield(self):
        return np.sum(np.transpose(self.phase_mat,(1,2,0))*self.amp*self.cwake_sum,1)

    def bunchProfile(self):

        #tmax = np.amax(self.blen)*5
        #########################################################
        #tmax = 1250e-12
        ###20190418 tmax = 1300e-12
        tmax = 1300e-12*100e6/self.sring.frf
        #trange = np.arange(-1.1,1.1,2e-3)*tmax######TO Paper
        #########################################################
        trange = np.linspace(-1,1,1001)*tmax
        #trange = np.linspace(-1,1,501)*tmax
        self.time = (np.ones((self.nbunch,trange.shape[0]))*trange).T-self.time_off

        #hamilton = self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*(self.time+self.time_off))-self.sring.eloss-0.5*np.sum(self.amp,1).real-\
        #           np.sum((np.exp(np.outer(1j*self.omegar+self.alpha,self.time).reshape(len(self.omegar),*self.time.shape)).transpose(1,2,0)*np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum).transpose(2,0,1),0).real

        #hamilton = self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*(self.time+self.time_off))-self.sring.eloss-0.5*np.sum(self.amp,1).real-\
        #           np.sum((np.exp(np.outer(1j*self.omegar+self.alpha,self.time).reshape(len(self.omegar),*self.time.shape)).transpose(1,2,0)
        #                   *np.sum(self.phase_mat.transpose(1,2,0)*self.amp,1)*self.cwake_sum).transpose(2,0,1),0).real
        
        #return hamilton

        cavfield = self.getCavfield()
        #cavfield = np.sum(np.transpose(self.phase_mat,(1,2,0))*self.amp*self.cwake_sum,1)        
        self.landau_phasor = cavfield[:]
        omegar = np.transpose(np.ones(self.time.shape+(len(self.omegar),),complex)*self.omegar,(2,0,1))
        cavfield = np.transpose(np.ones(self.time.shape+(len(self.omegar),),complex)*cavfield,(2,0,1))
        #hamilton = self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*(self.time+0*self.time_off))-self.sring.eloss-0.5*np.sum(self.amp,1).real-\
        #           np.sum(np.absolute(cavfield)*np.cos(omegar*(self.time-self.time_off)+np.angle(cavfield)),0)
        
        hamilton = 0
        for a in self.activeCavs:
            v,phi = a.calcActiveParams(self.cwake[a.index]+0.5*self.amp[:,a.index].real,self.dwake[a.index])
            hamilton += v*np.sin(phi+a.omega*(self.time_off+self.time))+0*1j
        
        hamilton += (not self.feedback)*self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*(self.time+self.time_off))-self.sring.eloss-0.5*np.sum(self.amp,1).real-\
                    np.sum(np.absolute(cavfield)*np.cos(omegar*(self.time-0*self.time_off)+np.angle(cavfield)),0)
        
        self.mainrf = self.vrf*np.sin(self.phi_rf+2*np.pi*self.sring.frf*(self.time+self.time_off))-self.sring.eloss-0.5*np.sum(self.amp,1).real
        self.landau_rf = np.sum(np.absolute(cavfield)*np.cos(omegar*self.time+np.angle(cavfield)),0)
        self.mainrf_phasor = self.vrf*np.exp(1j*(np.pi/2.-self.phi_rf+2*np.pi*self.sring.frf*self.time_off))#-self.sring.eloss-0.5*np.sum(self.amp,1)

        if self.complex256:
            dt = np.complex256
        else:
            dt = complex
        dist = np.array([-self.revfreq/self.sring.energy/self.sring.espread**2/self.sring.alphac*np.trapz(hamilton[:n+1],x=self.time[:n+1],axis=0) for n in range(self.time.shape[0])],dtype=dt)
        dist = np.exp(dist)
        ##dist = dist/np.trapz(dist[~np.isnan(dist)],self.time[~np.isnan(dist)],axis=0)
        dist = np.array([d/np.trapz(d[~np.isnan(d)],t[~np.isnan(d)]) for t,d in zip(self.time.T,dist.T)]).T
        if self.complex256: dist = np.array(dist,dtype=np.complex128)
        #Tracer()()        
        #dist[np.isnan(dist)] = 0
        self.hamilton = hamilton[:]
        self.dist = dist[:]
        
        self.bmean = np.trapz(self.dist*self.time,x=self.time,axis=0)/np.trapz(self.dist,x=self.time,axis=0)
        self.blen = np.sqrt(np.trapz(self.dist*self.time**2,x=self.time,axis=0)/np.trapz(self.dist,x=self.time,axis=0)-self.bmean**2)
        #self.bmean += self.time_off

    def formFactTO(self):
        #testing Teresia's functions
        cavfield = np.sum(np.transpose(self.phase_mat,(1,2,0))*self.amp*self.cwake_sum,1).T#-0.5*self.amp.real
        #cavfield = np.sum(self.phase_mat*self.amp*self.cwake_sum,1)
        #cavfield = np.absolute(cavfield)*np.exp(1j*(-np.angle(cavfield)-np.pi))
        #raise ValueError
        #bls_res = bunch_length.bunch_length(self.bcurr,self.vrf*np.exp(1j*(np.pi/2-self.phi_rf+2*np.pi*self.sring.frf*self.time_off)),cavfield)
        bls_res = bunch_length.bunch_length(self.bcurr,self.vrf*np.exp(1j*(np.pi/2-self.phi_rf-2*np.pi*self.sring.frf*self.time_off)),-cavfield.conj()) 
        bls = bls_res[0][1:]
        #self.blen = np.roll(bls_res[2][-1::-1],int(np.sum(self.fill-1)))
        #self.bmean = np.roll(bls_res[1][-1::-1],int(np.sum(self.fill-1)))
        #self.formfact = np.roll(np.absolute(bls.T),int(np.sum(self.fill-1)),axis=0)

        self.blen = bls_res[2]
        self.bmean = bls_res[1]
        self.formfact = np.absolute(bls.T)

        #self.bmean = np.sum(self.dist*self.time,axis=0)/np.sum(self.dist,axis=0)       
        #self.blen = np.sqrt(np.sum(self.dist*self.time**2,axis=0)/np.sum(self.dist,axis=0)-self.bmean**2)        
        #self.time_off -= self.bmean

    def scalarForm(self):
        self.formfact = (np.absolute(np.trapz(np.exp(1j*np.outer(0*self.nharm*2*np.pi*self.sring.frf+self.omegar,self.time).reshape(len(self.omegar),*self.time.shape))*self.dist,axis=1,
                                              x=self.time[:,0]))/np.trapz(self.dist,axis=0,x=self.time[:,0])).T
        
    def fullForm(self):
        #self.formfact = (np.trapz(np.exp(-1j*np.outer(self.nharm*2*np.pi*self.sring.frf,self.time-self.time_off).reshape(len(self.omegar),*self.time.shape))*self.dist,axis=1,
        #x=self.time[:,0])/np.absolute(np.trapz(self.dist,axis=0,x=self.time[:,0]))).T
        self.formfact = (np.trapz(np.exp(-1j*np.outer(0*self.nharm*2*np.pi*self.sring.frf+self.omegar,self.time-0*self.time_off).reshape(len(self.omegar),*self.time.shape))*self.dist,axis=1,
                                  x=self.time[:,0])/np.absolute(np.trapz(self.dist,axis=0,x=self.time[:,0]))).T
        #self.formfact = (np.trapz(np.exp(-1j*np.outer(self.omegar,self.time-0*self.time_off).reshape(len(self.omegar),*self.time.shape))*self.dist,axis=1,
        #                          x=self.time[:,0])/np.absolute(np.trapz(self.dist,axis=0,x=self.time[:,0]))).T

    def tuneFromTracking(self,amplitude,nturns=4096):

        if isinstance(amplitude,(list,tuple,np.ndarray)):
            coords = np.zeros((nturns,2,self.sring.nbunch,len(amplitude)))
            landau_phasor = (np.ones((len(amplitude),)+self.landau_phasor.shape,complex)*self.landau_phasor).T
        else:
            coords = np.zeros((nturns,2,self.sring.nbunch))
            
        coords[0,0] = -amplitude
        coords[0,1] = 0
        nharm = (np.ones(landau_phasor.shape).T*self.nharm).T
        for n in range(1,nturns):
            coords[n,1] = coords[n-1,1]+(self.sring.vrf*np.sin(2*np.pi*self.sring.frf*coords[n-1,0]+self.phi_rf)
                                         -np.sum(np.absolute(landau_phasor)*np.cos(nharm*2*np.pi*self.sring.frf*coords[n-1,0]+np.angle(landau_phasor)),0)-self.sring.eloss)/self.sring.energy
            coords[n,0] = coords[n-1,0]-self.sring.alphac*coords[n,1]/self.revfreq

        return coords

class ActiveCavity:

    def __init__(self,v_design,ph_design,index,omega):

        self.v_design = v_design
        self.ph_design = ph_design
        self.index = index
        self.omega = omega
        self.fixed = False

    def addFixed(self,v_fix,ph_fix):
        self.fixed = True
        self.v_fixed = v_fix
        self.ph_fixed = ph_fix

    def calcActiveParams(self,cwake,dwake):

        #wakevamp = np.mean(np.absolute(wake))
        #wakephi = np.mean(np.angle(wake))
        #wakev = wakevamp*np.exp(1j*wakephi)
        #v_new = self.v_design+np.mean(np.absolute(cwake))
        #phi_new = self.ph_design

        if self.fixed:
            return self.v_fixed, self.ph_fixed

        cwake = np.mean(np.absolute(cwake))*np.exp(1j*np.mean(np.angle(cwake)))
        dwake = np.mean(np.absolute(dwake))*np.exp(1j*np.mean(np.angle(dwake)))
        
        phi_new = np.arctan2((self.v_design*np.sin(self.ph_design)+np.mean(cwake.real))*self.omega,(self.v_design*self.omega*np.cos(self.ph_design)+np.mean(dwake.real)))
        v_new = (self.v_design*np.sin(self.ph_design)+np.mean(cwake.real))/np.sin(phi_new)

        self.v_calc = v_new
        self.ph_calc = phi_new

        return v_new, phi_new

class EigenTransient(Transient):

    def fourierTransformMatrix(self):

        self.ftrans = np.exp(1j*2*np.pi*np.outer(np.arange(self.sring.nbunch),np.arange(self.sring.nbunch))/self.sring.nbunch)
        self.iftrans = self.ftrans.conj()/self.sring.nbunch
        #self.matrix = np.dot(self.ftrans,np.dot(self.lhs,self.iftrans))
        self.matrix = self.lhs[:]

    def solvEigen(self):

        self.eigenfreqs, self.eigenmodes = np.linalg.eig(self.matrix)

class LorentzTransient(Transient):

    @staticmethod
    def lorentz(a,b,modeno):
        if b==np.inf:
            return a*(modeno==0)
        else:
            return a/(1+1j*b*modeno)

    def __init__(self,amp,bw,phi,*args,**kwargs):
        self.phi = phi
        self.bw = bw
        self.mdamp = amp
        Transient.__init__(self,*args,**kwargs)
        self.amp = self.sring.current*2/self.revfreq*np.outer(self.bcurr,self.rs*self.alpha)*self.formfact
        
    def getCavfield(self):
        cavfield = np.zeros((self.sring.nbunch,len(self.phi)),complex)
        modenums = np.arange(-self.sring.nbunch/2*0,1.5)
        #modenums = np.arange(-self.sring.nbunch/2*0,self.sring.nbunch/2+0.5)
        for i,(p,b,a) in enumerate(zip(self.phi,self.bw,self.mdamp)):
            modes = self.lorentz(a,b,modenums)
            cavfield[:,i] = np.sum(modes*np.exp(1j*np.outer(np.arange(self.sring.nbunch),modenums)*2*np.pi/self.sring.nbunch),1)*np.exp(1j*p)
            #modes[modenums==0] = 0
            #cavfield[:,i] = a*np.exp(1j*p)*(1+1j*np.sum(modes*np.exp(1j*np.outer(np.arange(self.sring.nbunch),modenums)*2*np.pi/self.sring.nbunch),1))
            #Tracer()()
        return cavfield

class Mode1Transient(Transient):

    def __init__(self,amp,phi,amp1,phi1,*args,**kwargs):
        
        self.mdamp = amp
        self.phi = phi        
        self.mdamp1 = amp1
        self.phi1 = phi1

        Transient.__init__(self,*args,**kwargs)
        self.amp = self.sring.current*2/self.revfreq*np.outer(self.bcurr,self.rs*self.alpha)*self.formfact
        
    def getCavfield(self):
        osc1 = np.exp(1j*np.arange(self.sring.nbunch)*2*np.pi/self.sring.nbunch)
        if self.iterno==0:
            cavfield = np.zeros((self.sring.nbunch,len(self.phi)),complex)
            for i,(a,a1,p,p1) in enumerate(zip(self.mdamp,self.mdamp1,self.phi,self.phi1)):
                cavfield[:,i] = (a+a1*np.exp(1j*np.arange(self.sring.nbunch)*2*np.pi/self.sring.nbunch))*np.exp(1j*(p+p1*np.sin(np.arange(self.sring.nbunch)*2*np.pi/self.sring.nbunch)))
            return cavfield
        else:
            cavfield = Transient.getCavfield(self)
            for i,(a,a1,p,p1) in enumerate(zip(self.mdamp,self.mdamp1,self.phi,self.phi1)):
                ff1 = np.sum(osc1.conj()*cavfield[:,i])/self.sring.nbunch
                cavfield[:,i] = cavfield[:,i]-ff1*osc1+a1*osc1*np.exp(1j*(p+p1*np.sin(np.arange(self.sring.nbunch)*2*np.pi/self.sring.nbunch)))
            return cavfield

class EmpiricalTransient(Transient):

    def __init__(self,amp,phi,phi1,phislope,amp1,ampslope,*args,**kwargs):

        self.mdamp = amp
        self.phi = phi        
        self.mdamp1 = amp1
        self.phi1 = phi1
        self.ampslope = ampslope
        self.phislope = phislope

        Transient.__init__(self,*args,**kwargs)
        self.amp = self.sring.current*2/self.revfreq*np.outer(self.bcurr,self.rs*self.alpha)*self.formfact

    def getCavfield(self):
        cavfield = np.zeros((self.sring.nbunch,len(self.phi)),complex)
        modenums = np.arange(1,self.sring.nbunch/2,1)
        allosc = np.exp(1j*np.outer(np.arange(self.sring.nbunch),modenums)*2*np.pi/self.sring.nbunch)
        #modenums = np.arange(-self.sring.nbunch/2*0,self.sring.nbunch/2+0.5)
        for i,(a,p,a1,ass,p1,pss) in enumerate(zip(self.mdamp,self.phi,self.mdamp1,self.ampslope,self.phi1,self.phislope)):
            modes = a1*np.exp(ass*modenums)*np.exp(1j*(p1+pss*modenums))
            cavfield[:,i] = a*np.exp(1j*p)+np.sum(modes*allosc,1)
            #modes[modenums==0] = 0
            #cavfield[:,i] = a*np.exp(1j*p)*(1+1j*np.sum(modes*np.exp(1j*np.outer(np.arange(self.sring.nbunch),modenums)*2*np.pi/self.sring.nbunch),1))
            #Tracer()()
        return cavfield

class SawTransient(Transient):

    @staticmethod
    def sawTooth(a,modeno):
        return 2*a/np.pi*(-1)**modeno/modeno

    def __init__(self,amp,amp1,phi,*args,**kwargs):
        self.mdamp = amp
        self.mdamp1 = amp1
        self.phi = phi
        Transient.__init__(self,*args,**kwargs)
        self.amp = self.sring.current*2/self.revfreq*np.outer(self.bcurr,self.rs*self.alpha)*self.formfact
        
    def getCavfield(self):
        cavfield = np.zeros((self.sring.nbunch,len(self.phi)),complex)
        #modenums = np.arange(-self.sring.nbunch/2*0,1.5)
        modenums = np.arange(-self.sring.nbunch/2*0,self.sring.nbunch/8+0.5)
        for i,(a,b,p) in enumerate(zip(self.mdamp,self.mdamp1,self.phi)):
            modes = self.sawTooth(b,modenums)
            modes[modenums==0] = a
            cavfield[:,i] = np.sum(modes*np.exp(-1j*np.outer(np.arange(self.sring.nbunch),modenums)*2*np.pi/self.sring.nbunch),1)*np.exp(1j*p)
            #Tracer()()
        return cavfield

class FlatPotentialTransient(Transient):

    def __init__(self,*args,**kwargs):

        Transient.__init__(self,*args,**kwargs)

    def bunchLengthening(self,wsfromblen=False):
        self.wsfromblen = wsfromblen
        self.bunchProfile()
        if self.scalar:
            self.scalarForm()
        else:
            self.fullForm()

    def bunchProfile(self):

        from scipy import special
        
        tmax = 1300e-12*100e6/self.sring.frf/4.
        trange = np.linspace(-1,1,1001)*tmax        
        self.time = (np.ones((self.nbunch,trange.shape[0]))*trange).T
        qso = np.sqrt(self.nbunch*self.sring.alphac*self.sring.vrf*np.sqrt(1-(self.sring.eloss/self.sring.vrf)**2)/2./np.pi/self.sring.energy)

        self.blen = 2*np.sqrt(np.pi)/special.gamma(0.25)*(3./(self.nharm[1]*self.nharm[1]-1))**0.25*np.sqrt(self.nbunch*self.sring.alphac*self.sring.espread/qso)/2./np.pi/self.sring.frf
        self.dist = 8**0.25/special.gamma(0.25)**2/self.blen*np.sqrt(2*np.pi)*np.exp(-2*np.pi**2/special.gamma(0.25)**4*(self.time/self.blen)**4)

        deltaE = np.arange(1,10001)/10000.*5*self.sring.espread
        edist = np.exp(-deltaE**2/2./self.sring.espread**2)
        nus = flatPotentialTune2(self.sring,deltaE,nharm=self.nharm[1])
        if self.wsfromblen:
            self.ltune = self.sring.espread*self.sring.alphac/self.blen/2./np.pi/self.revfreq
        else:
            self.ltune = np.trapz(nus*edist,x=deltaE)/np.trapz(edist,x=deltaE)/self.revfreq

        tmplte = np.ones(self.nbunch)
        self.ltune *= tmplte
        self.blen *= tmplte

def randomDistribution(i,bnum=0,npart=10000):

    from utility import acceptReject

    e = np.random.randn(npart)*i.sring.espread
    dist = i.dist[:,bnum]/np.amax(i.dist[:,bnum])
    distfunc = lambda x: np.interp(x,i.time[:,bnum],dist.real)
    tau = acceptReject(npart,i.time[0,bnum],i.time[-1,bnum],distfunc)

    return np.array(zip(np.arange(npart),(tau+i.time_off[bnum])*1e9,e))

def getFs(sring,delta,time,voltage):

    hamilton = np.array(delta)**2/2.
    myham = np.outer(np.array([np.trapz(voltage[:n+1],x=time[:n+1],axis=0) for n in range(time.shape[0])]),np.ones(hamilton.shape))
    invfidot = 1/np.sqrt(2*(hamilton-(myham-np.amin(myham,axis=0))/sring.alphac/sring.energy*sring.frf/sring.nbunch))

    fs = 1/(2./sring.alphac*np.trapz(invfidot,x=time,axis=0))
    
    return fs

def getFsPFT(tinst,delta,index):

    hamilton = np.array(delta)**2/2.

    time = tinst.time[:,index]
    time = np.linspace(np.amin(time),np.amax(time),10000)
    sring = tinst.sring
    n = tinst.nharm[1]
    k = np.absolute(tinst.landau_phasor[index,1])/sring.vrf
    elossnorm = sring.eloss/sring.vrf
    phi = 2*np.pi*sring.frf*time
    phis = tinst.phi_rf+2*np.pi*sring.frf*tinst.time_off[index]
    #phis = np.arcsin(elossnorm)
    phih = -np.pi/2+np.angle(tinst.landau_phasor[index,1])
    
    bigPhi = sring.vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi)+0*k/n*(np.cos(phih)-np.cos(phih+n*phi))-elossnorm*phi)
    myham = np.array(np.outer(bigPhi,np.ones(hamilton.shape)),dtype=complex)

    invfidot = 1/np.sqrt(2*(hamilton-(myham-0*np.amin(myham,axis=0))))

    fs = 1/(2./sring.alphac*np.trapz(invfidot,x=time,axis=0))

    return bigPhi, invfidot, fs

def getFsVariableTransform(tinst,delta,index=0,k=None,n=None,phih=None,area=False):
    """
    Calculate the tune at a given energy offset for an arbitrary double-RF system
    
    tinst - Either a Transient instance or a utility.StorageRing instance
    delta - Relative energy offset

    Optional keyword arguments with default values:
    index=0    - If Transient instance given, RF bucket to calculate for
    k    =None - Ratio of Landau cavity voltage to main cavity voltage
    n    =None - If StorageRing instance given, harmonic number of Landau cavity
    phih =None - Phase of the Landau cavity field at the synchronous phase of the RF bucket

    If only *n* is given, k and phih are calculated for flat potential conditions.
    """

    hamilton = np.array(delta)**2/2.

    if isinstance(tinst,utility.StorageRing):
        sring = tinst
        elossnorm = sring.eloss/sring.vrf
        vrf = 1*sring.vrf
        time = np.linspace(-1.3e-9,1.3e-9,30000)
        phis = np.arcsin(elossnorm)
        if n==None:
            k = 0
            n = 1
            phih = 0
        elif k==None and phih==None:
            phis = np.arcsin(n*n*elossnorm/(n*n-1))
            if k==None: k = np.sqrt(1./n**2-1./(n*n-1)*elossnorm**2)
            if phih==None: phih = -np.pi-np.arctan(-n*elossnorm/np.sqrt((n*n-1)**2-(n*n*elossnorm)**2))
            print 'Landau cavity voltage: %.1f' % (k*sring.vrf)
        else:
            k = 0
            phih = 0
            n = 1
    else:
        time = tinst.time[:,index]
        time = np.linspace(np.amin(time),np.amax(time),30000)
        sring = tinst.sring
        elossnorm = sring.eloss/sring.vrf
        if tinst.feedback:
            v,phi = tinst.activeCavs[0].calcActiveParams(tinst.cwake[0],tinst.dwake[0])
            main_phasor = -tinst.landau_phasor[index,0]+v*np.exp(-1j*(np.pi/2.-(phi+2*np.pi*sring.frf*tinst.time_off[index])))
            vrf = np.absolute(main_phasor)
            phis = np.pi/2.+np.angle(main_phasor)
        else:
            phis = tinst.phi_rf+2*np.pi*sring.frf*tinst.time_off[index]
            vrf = 1*sring.vrf
        n = tinst.nharm[1]
        if tinst.feedback and len(tinst.activeCavs)>1 or len(tinst.activeCavs)>0:
            v,phi = tinst.activeCavs[tinst.feedback].calcActiveParams(tinst.cwake[1],tinst.dwake[1])
            landau_phasor = -tinst.landau_phasor[index,1]+v*np.exp(-1j*(np.pi/2.-(phi+2*np.pi*n*sring.frf*tinst.time_off[index])))
            k = np.absolute(landau_phasor)/vrf
            phih = -3*np.pi/2.+np.angle(landau_phasor)
        else:
            k = np.absolute(tinst.landau_phasor[index,1])/vrf
            phih = -np.pi/2+np.angle(tinst.landau_phasor[index,1])

        print 'RF bucket number = %d' % index

    phi = 2*np.pi*sring.frf*time
    
    bigPhi = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi)+k/n*(np.cos(phih)-np.cos(phih+n*phi))-elossnorm*phi)
    myham = np.array(np.outer(bigPhi,np.ones(hamilton.shape)),dtype=complex)

    fidot = np.sqrt(2*(hamilton-(myham-np.amin(myham,axis=0))))
    #invfidot = 1/np.sqrt(2*(hamilton-(myham-np.amin(myham,axis=0))))

    midpoint = np.argmin(myham,axis=0)
    fitu1 = np.zeros(len(midpoint),int)
    fitu2 = np.zeros(len(midpoint),int)
    for i,m in enumerate(midpoint):
        fitu1[i] = np.argmin(np.absolute(fidot[:m,i]),axis=0)
        fitu2[i] = np.argmin(np.absolute(fidot[m:,i]),axis=0)+m

    imax = np.sqrt(time[midpoint]-time[fitu1])
    phip1 = np.array([np.linspace(0,i,len(time)) for i in imax]).T
    time1 = time[fitu1]+phip1**2
    phi1 = 2*np.pi*sring.frf*time1
    bigPhi1 = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi1)+k/n*(np.cos(phih)-np.cos(phih+n*phi1))-elossnorm*phi1)
    myham1 = np.array(bigPhi1,dtype=complex)
    invfidot1 = 2*phip1/np.sqrt(2*(hamilton-(myham1-np.amin(myham1,axis=0))))

    imax = np.sqrt(time[fitu2]-time[midpoint])
    phip2 = np.array([np.linspace(0,i,len(time)) for i in imax]).T
    time2 = time[fitu2]-phip2**2
    phi2 = 2*np.pi*sring.frf*time2
    bigPhi2 = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi2)+k/n*(np.cos(phih)-np.cos(phih+n*phi2))-elossnorm*phi2)
    myham2 = np.array(bigPhi2,dtype=complex)
    invfidot2 = -2*phip2/np.sqrt(2*(hamilton-(myham2-np.amin(myham2,axis=0))))

    fs = 1/(2./sring.alphac*(np.trapz(invfidot1,x=phip1,axis=0)-np.trapz(invfidot2,x=phip2,axis=0)))

    print 'vrf=%.3e, phi_s=%.5f' % (vrf,phis)    
    print 'n=%d, k=%.5f, n*phi_h=%.5f' % (n,k,phih)

    #Tracer()()
    if area:
        #fidot1 = 2*phip1/np.sqrt(2*(hamilton-(myham1-np.amin(myham1,axis=0))))    
        #fidot2 = -2*phip2*np.sqrt(2*(hamilton-(myham2-np.amin(myham2,axis=0))))
        #area = 2*(np.trapz(fidot1,x=phip1,axis=0)-np.trapz(fidot2,x=phip2,axis=0))
        area = 2*np.trapz(fidot.T,x=phi)
        return bigPhi, invfidot1, area, fs

    return bigPhi, invfidot1, fs

def getFsGeneral(tinst,delta,index=0,area=False):
    """
    Calculate the tune at a given energy offset for an arbitrary RF potential

    tinst - Transient instance
    delta - Relative energy offset

    Optional keyword arguments:
    index = 0     - If Transient instance given, RF bucket to calculate for
    area  = False - Return the phase space area as well 
    """

    hamilton = np.array(delta)**2/2.

    time = tinst.time[:,index]
    time = np.linspace(np.amin(time),np.amax(time),30000)
    sring = tinst.sring
    elossnorm = sring.eloss/sring.vrf

    
    
    if tinst.feedback:
        v,phi = tinst.activeCavs[0].calcActiveParams(tinst.cwake[0],tinst.dwake[0])
        main_phasor = -tinst.landau_phasor[index,0]+v*np.exp(-1j*(np.pi/2.-(phi+2*np.pi*sring.frf*tinst.time_off[index])))
        vrf = np.absolute(main_phasor)
        phis = np.pi/2.+np.angle(main_phasor)
    else:
        phis = tinst.phi_rf+2*np.pi*sring.frf*tinst.time_off[index]
        vrf = 1*sring.vrf
        
    ks = []
    nphihs = []
    ns = []
    #ns = tinst.nharm[tinst.feedback:]
    ind = tinst.feedback
    for a in tinst.activeCavs[tinst.feedback:]:
        while ind!=a.index and ind<len(tinst.nharm):
            ks.append(np.absolute(tinst.landau_phasor[index,ind])/vrf)
            nphihs.append(-np.pi/2+np.angle(tinst.landau_phasor[index,ind]))
            ns.append(tinst.nharm[ind])
            ind += 1
        v,phi = a.calcActiveParams(tinst.cwake[a.index],tinst.dwake[a.index])
        landau_phasor = -tinst.landau_phasor[index,a.index]+v*np.exp(-1j*(np.pi/2.-(phi+a.omega*tinst.time_off[index])))
        k = np.absolute(landau_phasor)/vrf
        phih = -tinst.nharm[a.index]*np.pi/2.+np.angle(landau_phasor)
        ks.append(k)
        nphihs.append(phih)
        ns.append(tinst.nharm[a.index])

    print 'RF bucket number = %d' % index

    phi = 2*np.pi*sring.frf*time

    bigPhi = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi)
                                                                  +np.sum(np.array([k/n*(np.cos(phih)-np.cos(phih+n*phi)) for k,n,phih in zip(ks,ns,nphihs)]),axis=0)
                                                                  -elossnorm*phi)
    myham = np.array(np.outer(bigPhi,np.ones(hamilton.shape)),dtype=complex)

    fidot = np.sqrt(2*(hamilton-(myham-np.amin(myham,axis=0))))
    #invfidot = 1/np.sqrt(2*(hamilton-(myham-np.amin(myham,axis=0))))

    midpoint = np.argmin(myham,axis=0)
    fitu1 = np.zeros(len(midpoint),int)
    fitu2 = np.zeros(len(midpoint),int)
    for i,m in enumerate(midpoint):
        fitu1[i] = np.argmin(np.absolute(fidot[:m,i]),axis=0)
        fitu2[i] = np.argmin(np.absolute(fidot[m:,i]),axis=0)+m

    imax = np.sqrt(time[midpoint]-time[fitu1])
    phip1 = np.array([np.linspace(0,i,len(time)) for i in imax]).T
    time1 = time[fitu1]+phip1**2
    phi1 = 2*np.pi*sring.frf*time1
    bigPhi1 = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi1)
                                                                   +np.sum(np.array([k/n*(np.cos(phih)-np.cos(phih+n*phi1)) for k,n,phih in zip(ks,ns,nphihs)]),axis=0)
                                                                   -elossnorm*phi1)
    myham1 = np.array(bigPhi1,dtype=complex)
    invfidot1 = 2*phip1/np.sqrt(2*(hamilton-(myham1-np.amin(myham1,axis=0))))

    imax = np.sqrt(time[fitu2]-time[midpoint])
    phip2 = np.array([np.linspace(0,i,len(time)) for i in imax]).T
    time2 = time[fitu2]-phip2**2
    phi2 = 2*np.pi*sring.frf*time2
    bigPhi2 = vrf/sring.alphac/2./np.pi/sring.energy/sring.nbunch*(np.cos(phis)-np.cos(phis+phi2)
                                                                   +np.sum(np.array([k/n*(np.cos(phih)-np.cos(phih+n*phi2)) for k,n,phih in zip(ks,ns,nphihs)]),axis=0)
                                                                   -elossnorm*phi2)
    myham2 = np.array(bigPhi2,dtype=complex)
    invfidot2 = -2*phip2/np.sqrt(2*(hamilton-(myham2-np.amin(myham2,axis=0))))

    fs = 1/(2./sring.alphac*(np.trapz(invfidot1,x=phip1,axis=0)-np.trapz(invfidot2,x=phip2,axis=0)))

    print 'vrf=%.3e, phi_s=%.5f' % (vrf,phis)    
    print 'n=%d, k=%.5f, n*phi_h=%.5f' % (n,k,phih)

    #Tracer()()
    if area:
        #fidot1 = 2*phip1/np.sqrt(2*(hamilton-(myham1-np.amin(myham1,axis=0))))    
        #fidot2 = -2*phip2*np.sqrt(2*(hamilton-(myham2-np.amin(myham2,axis=0))))
        #area = 2*(np.trapz(fidot1,x=phip1,axis=0)-np.trapz(fidot2,x=phip2,axis=0))
        area = 2*np.trapz(fidot.T,x=phi)
        return bigPhi, invfidot1, area, fs

    return bigPhi, invfidot1, fs    
    
def flatPotentialTune(sring,delta,nharm=3):
    """
    Calculate the tune vs. energy offset for the theoretical flat potential condition.

    sring - utility.StorageRing instance
    delta - Relative energy offset (can be an array)
    nharm - Harmonic number
    """

    from scipy import special

    tune0 = np.sqrt(sring.nbunch*sring.alphac*sring.vrf*np.sqrt(1-(sring.eloss/sring.vrf)**2)/2./np.pi/sring.energy)
    tunegrad = tune0*np.pi/2./special.ellipk(1/2.)*np.sqrt((nharm*nharm-1)/6.)
    bigC = special.gamma(0.25)**2/4./np.pi*np.sqrt((nharm*nharm-1)/3.)

    #return sring.frf/sring.nbunch*np.sqrt(tunegrad*sring.nbunch*sring.alphac*delta)

    return sring.frf/sring.nbunch*tunegrad*np.sqrt(sring.nbunch*sring.alphac/tune0/bigC*delta)

def flatPotentialTune2(sring,delta,nharm=3):

    from scipy import special

    cosphis0 = np.sqrt(1-(sring.eloss/sring.vrf)**2)
    tune0 = np.sqrt(sring.nbunch*sring.alphac*sring.vrf*cosphis0/2./np.pi/sring.energy)
    tunegrad = tune0*np.pi/2./special.ellipk(1/2.)*((nharm*nharm-1)/3.)**0.25
    convfact = np.sqrt(sring.energy/sring.vrf*np.pi*sring.nbunch*sring.alphac*2/cosphis0)

    return tunegrad*np.sqrt(convfact*delta)*sring.frf/sring.nbunch

def tuneSpread(tinst,*args,**kwargs):
    """
    Calculate the tune spread in a bunch
    
    *tinst* - A utility.StorageRing instance or a Transient instance

    Optional keyword arguments with their default values

    flat=False - Excecute for a flat potential if true
    samples=10000 - Number of energy offsets for which to calculate the tune
    scan=False - If Transient instance as first argument, scan all RF buckets

    *args, **kargs - Optional arguments and keyword arguments to pass to 
                     'getFsVariableTransform'
    """

    flat = False
    if 'flat' in kwargs:
        flat = kwargs.pop('flat')

    pfit = False
    if 'pfit' in kwargs:
        pfit = kwargs.pop('pfit')

    scan = False
    if isinstance(tinst,utility.StorageRing):
        sring = tinst
    else:
        if 'scan' in kwargs:
            scan = kwargs.pop('scan')
        sring = tinst.sring

    samples = 10000
    if 'samples' in kwargs:
        samples = kwargs.pop('samples')

    deltaE = np.arange(1,samples+1,1,float)/(samples+1.)*5*sring.espread
    edist = np.exp(-(deltaE/sring.espread)**2/2.)*np.sqrt(2/np.pi)/sring.espread
    edistvolume = np.trapz(edist,x=deltaE)

    rms = lambda x: np.sqrt(np.trapz(x**2*edist,x=deltaE)/edistvolume-(np.trapz(x*edist,x=deltaE)/edistvolume)**2)

    if flat:
        nus = flatPotentialTune2(sring,deltaE,*args,**kwargs)
    elif scan:
        nbuckets = sring.nbunch/scan
        nus = np.zeros((sring.nbunch/scan,len(deltaE)))
        
        for i in range(nbuckets):
            nus[i] = getFsVariableTransform(tinst,deltaE,i*scan)[-1]
            if pfit: nus[i] = np.polyval(np.polyfit(deltaE,nus[i],10),deltaE)
    else:
        nus = getFsVariableTransform(tinst,deltaE,*args,**kwargs)[-1]
        if pfit: nus = np.polyval(np.polyfit(deltaE,nus[i],10),deltaE)

    return deltaE, nus, edist, rms(nus)

def calcCField(tinst,current,rs,rfreq,qfactor):

    omegar = 2*np.pi*rfreq
    alpha = omegar*0.5/qfactor
    formfact = np.trapz(np.exp(-1j*omegar*tinst.time)*tinst.dist,axis=0,x=tinst.time[:,0])/np.absolute(np.trapz(tinst.dist,axis=0,x=tinst.time[:,0]))
    amp = current*2/tinst.revfreq*tinst.bcurr*rs*alpha*formfact

    print amp[0], np.mean(formfact)

    deltaphi = np.outer(1j*omegar+alpha,-np.arange(tinst.sring.nbunch))/tinst.sring.frf    
    dwake_sum = (-1j*omegar-alpha)/(1-np.exp(-(1j*omegar+alpha)/tinst.revfreq))
    turn_back = np.exp(-(1j*omegar+alpha)/tinst.revfreq)*np.tri(tinst.nbunch)+np.tri(tinst.nbunch,k=-1).T
    phase_mat = np.outer(np.exp(deltaphi),np.exp(-deltaphi))*turn_back
    cwake_sum = 1./(1-np.exp(-(1j*omegar+alpha)/tinst.revfreq))

    cwake = -(np.sum(phase_mat*amp,1)*cwake_sum).T

    return cwake
        
