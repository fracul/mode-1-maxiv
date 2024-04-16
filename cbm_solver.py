import numpy as np
#from IPython.core.debugger import Tracer

class CbmTdmSolve:
    """
    Class for determining complex coherent frequencies using the time-domain method of Thompson and Ruth.
    """

    def __init__(self,tinst,rs,rfreq,qfact,modeinds=[],omegasapprox=False,use_boschwr=True,use_newwr=False):
        """
        Class initialisation:
        *tinst* - transients.Transient instance 
        *rs* - List of shunt impedances
        *rfreq* - List of resonant frequencies
        *qfact* - List of quality factors
        *modeinds*=[] - Indices of the modes with the above parameters can be found in the Transient instance if appropriate.
        *omegasapprox*=False - Use the small coherent-tune shift approximation
        *use_boschwr*=True - Use the Bosch Robinson frequnecy in the calculations, otherwise use the incoherent synchrotron
                             frequency calculated from the gradient of the RF voltage at the synchronous phase. If the gradient
                             is zero or negaitve, assume an incoherent synchrotron frequency of 0.
        *use_newwr*=False - Attempt to determine an average incoherent synchrotron frequency from the average RF gradient along
                            the bunch length.
        """

        self.tinst = tinst
        self.sring = self.tinst.sring
        if isinstance(rs,float):
            rs = [rs]
            rfreq = [rfreq]
            qfact = [qfact]
        rfreq = np.array(rfreq)
        qfact = np.array(qfact)
        self.rs = np.array(rs)
        self.omegar = 2*np.pi*rfreq
        self.alpha = self.omegar*0.5/qfact
        self.revfreq = self.sring.frf/self.sring.nbunch
        self.modeinds = modeinds
        self.omegasapprox = omegasapprox
        self.use_boschwr = use_boschwr
        self.use_newwr = use_newwr
        if not self.modeinds:
            self.modeinds = []
            for o in self.omegar:
                self.modeinds.extend(np.where([self.tinst.omegar==o])[0])

        self.meanTunes()

    def calcFormFact(self,angfreq,magnitude=True):

        if not hasattr(self.tinst,'time'):
            return np.ones((self.tinst.nbunch,len(angfreq)))
        
        ffact = (np.trapz(np.exp(-1j*np.outer(angfreq,self.tinst.time).reshape(len(angfreq),*self.tinst.time.shape))*self.tinst.dist,
                          axis=1,x=self.tinst.time[:,0])/np.trapz(self.tinst.dist,axis=0,x=self.tinst.time[:,0])).T
        if magnitude:
            return np.absolute(ffact)
        return ffact

    def meanTunes(self):
        if self.tinst.blength:
            if self.use_boschwr:
            
                nharm = list(self.tinst.nharm)
                mainoff = 1
                if nharm[0]!=1:
                    nharm = [0]+nharm
                    mainoff = 0
                nharm = np.array(nharm)
                ffact = self.calcFormFact(nharm*2*np.pi*self.sring.frf)
                #ffact = (np.absolute(np.trapz(np.exp(1j*np.outer(nharm*2*np.pi*self.sring.frf,self.tinst.time).reshape(len(self.tinst.omegar),*self.tinst.time.shape))*self.tinst.dist,axis=1,
                #                              x=self.tinst.time[:,0]))/np.trapz(self.tinst.dist,axis=0,x=self.tinst.time[:,0])).T
                vphi1 = np.pi/2-self.tinst.phi_rf
                v2 = np.mean(np.absolute(self.tinst.landau_phasor[:,mainoff:]),axis=0)        
                vphi2 = -np.pi-np.angle(self.tinst.landau_phasor[:,mainoff:]/self.tinst.formfact[:,mainoff:])

                self.wrob = np.sqrt(self.sring.alphac*self.sring.frf*2*np.pi*self.revfreq/self.sring.energy*(ffact[:,0]*self.sring.vrf*np.sin(vphi1)+
                                                                                                             np.sum(nharm[mainoff:]*ffact[:,mainoff:]*v2*np.sin(vphi2),1)))
            elif self.use_newwr:
                self.wrob = np.zeros(self.tinst.hamilton.shape[1])
                self.wrobnew = np.zeros(self.tinst.hamilton.shape[1])                
                self.hamdiff = []
                self.time = []
                self.dist = []
                const = self.sring.alphac*self.revfreq/self.sring.energy
                one_period = np.linspace(0,2*np.pi,100)
                for i,(h,d,t) in enumerate(zip(self.tinst.hamilton.T,self.tinst.dist.T,self.tinst.time.T)):
                    timelim = np.absolute(min(np.amin(t),np.amax(t)))
                    time = np.linspace(-timelim,timelim,500)
                    dist = np.interp(time,t/2.,d.real)
                    ham = np.interp(time,t,h.real)
                    hamdiff = np.zeros(500)
                    hamdiff[250:] = (ham[250:]-ham[249::-1])/(time[250:]-time[249::-1])
                    hamdiff[:250] = hamdiff[-1:249:-1]
                    self.wrob[i] = np.sqrt(const*np.trapz(hamdiff*dist)/np.trapz(dist))

                    dist = np.interp(time,t,d.real)
                    cosx = np.outer(time,np.cos(one_period))
                    hamdiff = np.zeros(500)
                    hamdiff[1:-1] = (ham[2:]-ham[:-2])/(time[2]-time[0])
                    hamdiff[0] = hamdiff[1]
                    hamdiff[-1] = hamdiff[-2]
                    self.wrobnew[i] = np.sqrt(const*np.trapz(np.trapz(np.array([np.interp(c,time,hamdiff) for c in cosx]),one_period,axis=1)/2./np.pi*dist)/np.trapz(dist))

                    self.time.append(time.copy())
                    self.dist.append(dist.copy())
                    self.hamdiff.append(hamdiff.copy())
                    #self.hamdiff.append(np.trapz(np.array([np.interp(c,time,hamdiff) for c in cosx]),axis=1)/2./np.pi)
                    
            else:
                self.wrob = 2*np.pi*self.sring.frf/self.sring.nbunch*np.nanmax([self.tinst.ltune,np.zeros(self.tinst.ltune.shape[0])],axis=0)+1j*0
                #Tracer()()
                    
                #ham = self.tinst.hamilton
                #dist = self.tinst.dist
                #time = self.tinst.time
                #halflen = self.tinst.time.shape[0]/2
                #self.tstdiff = np.zeros(ham.shape,complex)
                #self.dist = np.zeros(ham.shape,complex)
                #if (halflen!=self.tinst.time.shape[0]/2.):
                #    self.tstdiff[halflen+1:] = (ham[halflen+1:]-ham[halflen-1::-1])/(time[halflen+1:]+self.tinst.time_off)
                #    self.tstdiff[:halflen] = self.tstdiff[-1:halflen:-1]
                #    self.tstdiff[halflen] = np.mean(self.tstdiff[halflen-1:halflen+2:2],axis=0)
                #    self.dist[halflen] = self.tinst.dist[halflen]
                #    tail1 = self.tinst.dist[halflen+2::2]
                #    tail0 = self.tinst.dist[:halflen:2]
                #    self.dist[halflen+1:halflen+len(tail1)+1] = tail1
                #    self.dist[halflen-len(tail0):halflen] = tail0
                #else:
                #    self.tstdiff[halflen:] = (ham[halflen:]-ham[halflen-1::-1])/(time[halflen:]+self.tinst.time_off)
                #    self.tstdiff[:halflen] = self.tstdiff[-1:halflen:-1]
                #    tail = self.tinst.dist[::2]
                #    if halflen/2==halflen/2.:
                #        self.dist[halflen-len(tail)/2:halflen+len(tail/2)] = tail
                #    else:
                #        self.dist[halflen-len(tail)/2:halflen+len(tail/2)+1] = tail                        
                #self.wrob = np.sqrt(self.sring.alphac*self.revfreq/self.sring.energy*np.trapz(self.tstdiff*self.dist,axis=0)/np.trapz(self.dist,axis=0))
                #Tracer()()
        else:
            self.tinst.bunchLengthening()
            self.wrob = 2*np.pi*self.sring.frf/self.sring.nbunch*self.tinst.ltune+1j*0
            self.tinst.formfact = np.ones((self.sring.nbunch,len(self.tinst.nharm)),complex)
        
    def constructMatrix(self,wrobbar=None):

        if wrobbar==None:
            wrobbar = np.mean(self.wrob)
            
        denomp = (1-np.exp((1j*(wrobbar-self.omegar)-self.alpha)/self.revfreq))
        denomm = (1-np.exp((1j*(wrobbar+self.omegar)-self.alpha)/self.revfreq))
        self.dwake_sump = self.alpha/2./denomp+1j*self.omegar/2./denomp
        self.dwake_summ = self.alpha/2./denomm-1j*self.omegar/2./denomm

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
        self.phase_mat = self.phase_mat*self.ffphi_mat
        self.phase_matc = self.phase_mat.conj()#*phase_mat_s*turn_back_s

        const = self.sring.alphac/self.sring.energy*self.revfreq+0*1j
        if self.omegasapprox:
            self.matrix = (np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sump+self.phase_matc.transpose(1,2,0)*self.amp*self.dwake_summ).transpose(1,0,2)*self.ffact,2)/self.wrob.real).T*const/2+self.wrob*np.eye(self.tinst.nbunch,dtype=complex)+0*wrobbar*np.eye(self.tinst.nbunch,dtype=complex)
        else:
            self.matrix = (np.sum((self.phase_mat.transpose(1,2,0)*self.amp*self.dwake_sump+self.phase_matc.transpose(1,2,0)*self.amp*self.dwake_summ).transpose(1,0,2)*self.ffact,2)).T*const*2/2+self.wrob*self.wrob*np.eye(self.tinst.nbunch,dtype=complex)+0*wrobbar*wrobbar*np.eye(self.tinst.nbunch,dtype=complex)
        #REMOVED A FACTOR OF TWO!!!! IN THE DENOMINATOR (OMEGA^2-omega^2)~=2omega(OMEGA-omega)????

    def solvEigen(self):
        self.eigenfreqs, self.eigenmodes = np.linalg.eig(self.matrix)
        if self.omegasapprox:
            self.bigOmega = self.eigenfreqs
        else:
            self.solveQuadraticEvals()
            self.bigOmega = self.ef_qsolvep

    def solveSelfConsistent(self,modenum=-1):

        from scipy import optimize
        
        if modenum>0 and not hasattr(self,'eigenmodes_laplace'):
            self.laplaceTransform()

        def getGrowthRate(x):
            self.constructMatrix(wrobbar=x[0])
            self.solvEigen()

            if modenum<0:
                modeind = np.argmax(self.bigOmega.imag)
            else:
                self.laplaceTransform()
                modeind = np.where(self.eigenmodes_lplcenum==modenum)[0][0]
                
            return (np.real(self.bigOmega[modeind])-x[0])**2
        

        res = optimize.fmin(getGrowthRate,np.array([np.mean(self.wrob)]))
        return res
            
    def laplaceTransform(self):
        self.eigenmodes_laplace = np.fft.fft(self.eigenmodes,axis=0)
        self.eigenmodes_lplcenum = (176-np.argmax(np.absolute(self.eigenmodes_laplace),axis=0))%176

    def sortEigenModes(self,sortby):
        lplce_transformed = hasattr(self,'eigenmodes_laplace')
        if lplce_transformed:
            sortarr = np.array(zip(sortby,np.arange(self.eigenfreqs.shape[0]),self.eigenfreqs,self.eigenmodes_lplcenum),
                               dtype=[('sortcol',float),('emind',int),('ef',complex),('emln',float)])
        else:
            sortarr = np.array(zip(sortby,np.arange(self.eigenfreqs.shape[0]),self.eigenfreqs),dtype=[('sortcol',float),('emind',int),('ef',complex)])
        sortarr.sort(order='sortcol')
        self.eigenfreqs = sortarr['ef']
        self.eigenmodes = self.eigenmodes.T[sortarr['emind']].T
        if lplce_transformed:
            self.eigenmodes_laplace = self.eigenmodes_laplace.T[sortarr['emind']].T
            self.eigenmodes_lplcenum = sortarr['emln']
            
        #if self.omegasapprox:
        #    self.bigOmega = self.eigenfreqs
        #else:
        #    self.solveQuadraticEvals()
        #    self.bigOmega = self.ef_qsolvep

    def solveQuadraticEvals(self):

        lmbd = 1/self.sring.taue

        b = -np.real(self.eigenfreqs)+lmbd**2+1j*0
        c = -np.imag(self.eigenfreqs)**2/4.+1j*0
        reomegap = np.sqrt(np.array(-b+np.sqrt(b*b-4*c),complex)/2)
        reomegam = np.sqrt(np.array(-b-np.sqrt(b*b-4*c),complex)/2)
        imomegap = np.imag(self.eigenfreqs)/2/reomegap#-lmbd
        imomegam = np.imag(self.eigenfreqs)/2/reomegam#-lmbd
        self.ef_qsolvep = reomegap+1j*imomegap
        self.ef_qsolvem = reomegam+1j*imomegam

    @staticmethod
    def solveQuadratic(eigfreq,taue):
        lmbd = 1/taue

        b = -np.real(eigfreq)+lmbd**2
        c = -np.imag(eigfreq)**2/4.
        reomegap = np.sqrt(np.array(-b+np.sqrt(b*b-4*c),complex)/2)
        reomegam = np.sqrt(np.array(-b-np.sqrt(b*b-4*c),complex)/2)
        imomegap = np.imag(eigfreq)/2/reomegap-lmbd
        imomegam = np.imag(eigfreq)/2/reomegam-lmbd
        ef_qsolvep = reomegap+1j*imomegap
        ef_qsolvem = reomegam+1j*imomegam

        return ef_qsolvep, ef_qsolvem
        
    def solveQuarticEvals(self):

        a = np.real(self.eigenfreqs)
        b = np.imag(self.eigenfreqs)
        lmbd = 1/self.sring.taue

        shpe4 = (len(self.eigenfreqs),)
        coeffs = np.ones(shpe4+(5,))        
        coeffs[:,1:] = np.array([3*lmbd*np.ones(shpe4),
                                 a+3*lmbd**2,
                                 2*a*lmbd+lmbd**3,
                                 a*lmbd**2-b**2/4.]).T
        out = np.zeros(shpe4+(4,),dtype=[('realimag',float),('roots',complex),('diffomega',complex)])

        for i,c in enumerate(coeffs):
            g = np.roots(c)
            realpart = b[i]/(2*(g+lmbd))
            out[i]['realimag'] = -realpart.imag
            out[i]['roots'] = -g
            out[i]['diffomega'] = realpart
            if (realpart.imag!=0).all():
                print('Warning: eigenfrequency %d has no completely real frequency shift' % i)

        out = np.sort(out,order=['realimag','roots'],axis=1)
        self.ef_qqsolve = (out['diffomega']-1j*out['roots']).T

def thresholdRs(cbinst,niter=1,fit_order=2,refreshtinst=False,quadrupole=False):

    damprate = 1/cbinst.tinst.sring.taue

    def getgrate():
        if refreshtinst:
            cbinst.tinst.rs[-1] = cbinst.rs[0]
            cbinst.tinst.runIterations(20,blenskip=5)
        if quadrupole:
            cbinst.constructMatrixKrinsky(order=2)
        else:
            cbinst.constructMatrix()
        cbinst.solvEigen()
        if cbinst.omegasapprox:
            bigOmega = 1*cbinst.eigenfreqs
        else:
            cbinst.solveQuadraticEvals()
            bigOmega = 1*cbinst.ef_qsolvep
        #cbinst.laplaceTransform()
        #cbmoff = 0.5
        #while not np.any(np.all(cbinst.eigenmodes.real<cbmoff,axis=1)):
        #    cbmoff += 0.1
        #print np.argmax(np.absolute(np.fft.fft(cbinst.eigenmodes,axis=0))[10]), np.argmax(bigOmega.imag)
        
        #return np.amax(bigOmega.imag)
        #return bigOmega.imag[np.argmax(np.absolute(np.fft.fft(cbinst.eigenmodes,axis=0))[10])]
        return np.amax(bigOmega.imag)#[np.all(cbinst.eigenmodes<cbmoff,axis=1)])
        
    rs0 = 1*cbinst.rs
    rsreduc = np.linspace(1,0.25,fit_order+1)/2.
    rs = np.outer(np.array(rsreduc),cbinst.rs)
    gr = np.zeros(len(rs))
    while (gr<500).all():
        rsreduc = np.array(rsreduc)*2
        rs = rs*2
        for i,r in enumerate(rs):
            cbinst.rs = 1*r
            gr[i] = getgrate()

    rsreduc = list(rsreduc)
    gr = list(gr)
    for n in range(niter):
        #tstfit = np.polyfit(np.array(gr)-damprate,rsreduc,fit_order)
        tstfit = np.polyfit(rsreduc,np.array(gr)-1/0.025194,fit_order)
        #tstfit = np.polyfit(rsreduc[-(fit_order+1):],np.array(gr[-(fit_order+1):])-damprate,fit_order)
        tstroots = np.roots(tstfit)
        if (tstroots.imag!=0).all():
            print('No real roots found, ending after %d iterations' % n)
            break
        tstroots = tstroots[tstroots.imag==0]
        if (tstroots<0).all():
            print('No positive roots found, ending after %d iterations' % n)
            break
        tstroots = tstroots[tstroots>0]
        cbinst.rs = np.amin(tstroots)*rs0
        #cbinst.rs = tstfit[-1]*rs0
        rsreduc.append(np.amin(tstroots))
        gr.append(getgrate())

    tol = rsreduc[-1]-rsreduc[-2]
    print('Tolerance reached: %e' % tol)

    #rsarray = np.array(zip(rsreduc,gr,np.absolute(gr)),dtype=[('rsreduc',float),('gr',float),('absgr',float)])
    #rsarray = np.sort(rsarray,order=['absgr'])
    #pfit = np.polyfit(rsarray['rsreduc'][:2],rsarray['gr'][:2]-damprate,1)
    #cbinst.rs = rsarray['rsreduc'][0]*rs0#-pfit[1]/pfit[0]*rs0
    
    return rsreduc, gr, cbinst, tol
        
def frequencyScan(freqs,*args,**kwargs):
    """
    args - transient.Transien instance ans arguments
    kwargs - scantype
             modesort
             landauOff
             omegasapprox
             refreshtinst
             fit_order
             fundfreq
             qvals
             rss
             fullform
             quadrupole
    """

    import transient
    import cbm_vlasov    

    modescan = False
    landauOff = False
    solveq = False
    solveqq = False
    refreshtinst = False
    fullform = False
    quadrupole = False
    scantype = 'growthrates'
    tmplte = np.ones(len(freqs))
    qvals = np.outer(tmplte,args[2])
    rss = np.outer(tmplte,args[1])
    fundfreqs = np.ones(len(freqs))*args[0].omegar[1]/2/np.pi

    if 'scantype' in kwargs:
        scantype = kwargs.pop('scantype')
    if 'modesort' in kwargs:
        modescan = kwargs.pop('modesort')
        #modenum = kwargs.pop('mode')
    if 'landauOff' in kwargs:
        landauOff = kwargs.pop('landauOff')
    if 'omegasapprox' in kwargs:
        solveq = not kwargs['omegasapprox']
        if solveq and 'solvequartic' in kwargs:
            solveqq = kwargs.pop('solvequartic')
            if solveqq:
                solveq = False
    if 'refreshtinst' in kwargs:
        refreshtinst = kwargs.pop('refreshtinst')
    if 'fit_order' in kwargs:
        fit_order = kwargs.pop('fit_order')
    else:
        fit_order = 2
    if 'fundfreqs' in kwargs:
        fundfreqs = kwargs.pop('fundfreqs')
    if 'qvals' in kwargs:
        qvals = kwargs.pop('qvals')
    if 'rss' in kwargs:
        rss = kwargs.pop('rss')
    if 'fullform' in kwargs:
        fullform = kwargs.pop('fullform')
    if 'quadrupole' in kwargs:
        quadrupole = kwargs.pop('quadrupole')

    if scantype=='thresholdrs':
        scanres = np.zeros((3,len(freqs)))
    else:
        scanres = np.zeros((3,len(freqs),args[0].nbunch),complex)
    hcfields = np.zeros(len(freqs))
    tunespread = np.zeros(len(freqs))

    for i,(f,r,q) in enumerate(zip(freqs,rss,qvals)):
        #Tracer()()
        t = args[0]
        if refreshtinst and scantype!='thresholdrs':
            t.omegar[1] = 2*np.pi*fundfreqs[i]
            if isinstance(f,np.ndarray):
                t.omegar[-len(f):] = 2*np.pi*f
            else:
                t.omegar[-1] = 2*np.pi*f
            #Tracer()()
            t = transient.Transient(t.sring,t.rs,t.nharm,t.omegar/2/np.pi-t.nharm*t.sring.frf,t.qfact,t.fill,blength=t.blength,formcalc='scalar')
            #t.time_off = np.zeros(len(t.time_off))
            try:
                t.runIterations(20,blenskip=5)
            except(np.linalg.linalg.LinAlgError):
                print('Static fail')
                continue
            t.runIterations(80,blenskip=5)
            if fullform:
                t.scalar = False
                t.full = True
                t.runIterations(100,blenskip=5)
        if len(freqs.shape)==1:
            f = (f,)          
        argnew = (t,)+(r,)+(f,)+(q,)+args[4:]
        if quadrupole:
            g = cbm_vlasov.VlasovSolve(*argnew,**kwargs)
        else:
            g = CbmTdmSolve(*argnew,**kwargs)
        if t.blength:
            hcfields[i] = np.mean(np.absolute(t.landau_phasor[:,1]))
        else:
            hcfields[i] = np.mean(g.wrob)
        tunespread[i] = np.std(g.wrob,ddof=1)
        
        if scantype=='thresholdrs':
            rstres = thresholdRs(g,niter=5,fit_order=fit_order,refreshtinst=refreshtinst,quadrupole=quadrupole)
            scanres[0,i] = rstres[2].rs[0]
            scanres[1,i] = rstres[1][-1]
            g.wrob = np.ones(g.wrob.shape[0],complex)*np.mean(g.wrob)
            
            if quadrupole:
                g.constructMatrixKrinsky(order=2)
            else:
                g.constructMatrix()
                
            try:
                g.solvEigen()
            except(np.linalg.linalg.LinAlgError):
                print('Dynamic fail (threshold Rs search)')
                continue
            if solveq:
                g.solveQuadraticEvals()
                bigOmega = g.ef_qsolvep
            else: bigOmega = g.eigenfreqs        
            scanres[1,i] = np.amax(bigOmega.imag)
            #scanres[2,i] = g.ef_qsolvep[np.argmax(bigOmega.imag)]
            scanres[2,i] = g.bigOmega[np.argmax(bigOmega.imag)]            
            
        else:
            if landauOff:
                g.wrob = np.ones(g.wrob.shape[0],complex)*np.mean(g.wrob)
                
            if quadrupole:
                g.constructMatrixKrinsky(order=2)
            else:
                g.constructMatrix()
                
            try:
                g.solvEigen()
            except(np.linalg.linalg.LinAlgError):
                print('Dynamic fail')
                continue
            if modescan:
                g.laplaceTransform()
                g.sortEigenModes(g.eigenmodes_lplcenum)
                #modeind = np.where(g.eigenmodes_lplcenum==modenum)[0]
                #if len(modeind)>1:
                #    print 'Warning: more than one eigenfrequency exists for mode number %d' % modenum
                #modeind = modeind[0]
            if solveq:
                g.solveQuadraticEvals()
                scanres[1,i] = g.ef_qsolvep
                scanres[2,i] = g.ef_qsolvem
            if solveqq:
                g.solveQuarticEvals()
                scanres[1:,i] = g.ef_qqsolve
            #scanres[0,i] = g.eigenfreqs#[np.argmax(g.eigenfreqs.imag)]
            scanres[0,i] = g.eigenmodes[np.argmax(g.eigenfreqs.imag)]

    return scanres, hcfields, tunespread

def emeryGrowth(sring,rfreq,rs,qfactor):

    revfreq = sring.frf/sring.nbunch
    nharm = round(np.mean(rfreq)/revfreq)
    varpi = (rfreq-nharm*revfreq)*2*qfactor/rfreq
    nu = rs*rfreq*2*np.pi*sring.blen/sring.espread*revfreq/sring.energy*np.exp(-np.square(2*np.pi*rfreq*sring.blen))
    omegas = sring.espread/sring.blen*sring.alphac

    return sring.current*nu*(1j-varpi)/(1+varpi**2)/2.

def plotResults(freqs,res,labels,polar=False,cbinst=None,index=[]):

    from pylab import figure

    f1 = figure()
    ax1 = f1.add_subplot(111)
    f2 = figure()
    ax2 = f2.add_subplot(111)
    if polar:
        f3 = figure()#figsize=(8,8))
        ax3 = f3.add_subplot(111)
    if not index:
        index = [None]*len(res)

    if cbinst!=None:
        omegas0 = np.mean(cbinst.wrob)
    else:
        omegas0 = np.mean(res[0].real)
    fr = (freqs-np.mean(freqs))/1e3
    xind = np.arange(len(fr))
    for ind,r in zip(index,res):
        if len(r.shape)==1:
            r = np.array([r]).T+omegas0
        elif ind==None:
            ind = np.argmax(r.imag,axis=1)
        
        imr = r.imag[xind,ind]
        rer = r.real[xind,ind]
        ax1.plot(fr,np.amax(r.imag,axis=1))
        ax2.plot(fr,rer/2./np.pi)
        if polar:
            ax3.plot(imr,rer,'-')
        
    ax1.legend(labels,loc=2,borderaxespad=0)
    ax1.set_xlabel('HOM detuning/kHz')
    ax1.set_ylabel(r'Growth rate/$\rm s^{-1}$')
    ax1.set_xlim(fr[0],fr[-1])
    ax2.legend(labels,loc=3,borderaxespad=0)
    ax2.set_xlabel('HOM detuning/kHz')
    ax2.set_ylabel('Synchrotron frequency shift/Hz')
    ax2.set_xlim(fr[0],fr[-1])    
    if polar:
        ax3.legend(labels,loc='center left',borderaxespad=0)
        ax3.set_xlabel(r'$\mathrm{Im}(\Omega)$',labelpad=10)
        ax3.set_ylabel(r'$\mathrm{Re}(\Omega)$')
        ax3.axis('equal')
        #xlm = ax3.get_xlim()[1]
        #ax3.set_xlim(0,xlm)
        #ylbar = np.mean(ax3.get_ylim())
        #ax3.set_ylim(ylbar-xlm/2.,ylbar+xlm/2.)

#def convertSingle(s,tinst):
#    tinst.nbunch = s.nbunch
    
    
