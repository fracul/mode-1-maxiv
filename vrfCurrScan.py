import scanDetune, haissinski, lindberg
from scipy.special import gamma
import cavity_tuning
import utility
import numpy as np
#from IPython.core.debugger import Tracer

sd = utility.StorageRing('./maxiv_delivery201901.inp')
sds = utility.StorageRing('./maxiv_delivery201901.inp')
sds.alphac *= 176
sds.nbunch = 1
sd.taue = 0.025194
sds.taue = 0.025194
gammaquart = gamma(1/4.)
lbergdat = np.loadtxt('landau_contour_store.txt')
lberg = lbergdat[:,2]+1j*lbergdat[:,3]

def boschAnalysisAtMinimumCurrent(vrfs,estcurrs=None,rs=8.25e6,nharm=3,**kwargs):
    """
    Determine the threshold currents of the Mode-1 instability for different RF voltages as predicted using
    the approach of Tianlong He. Stability is determined from a number of beam currents and Landau detunings
    and interpolation is used to find the beam current where the mode-1 amplitude crosses 1.
    *vrfs* - list of RF voltages
    *estcurrs*=None - list of estimated beam currents just above the threshold. If not given, the minimum beam
                      current at which flat potential can be achieved is used.
    *rs*=8.25e6    - shunt impedance of the Landau cavities
    *nharm*=3 - Harmonic number of the Landau cavities
    **kwargs - keyword arguments to pass to the function *boschAnalysisVrf*.
    """

    useest = False
    if estcurrs==None:
        estcurrs = np.ones(len(vrfs))
    else:
        useest = True
        
    results = []
    hcfres = []
    iths = np.zeros(len(vrfs))
    for i,(v,c) in enumerate(zip(vrfs,estcurrs)):
        sd.vrf = v
        if useest:
            current = np.arange(c-0.018,c,0.002)
        else:
            mincurr = cavity_tuning.minimumCurrentFlatPotential(rs,nharm,sd,0.94)
            current = np.arange(mincurr,mincurr+0.061,0.02)
        res = boschAnalysisVrf(v,current=current,**kwargs)
        hcfs = flatPotential(res)
        hcfs[np.isnan(hcfs)] = np.nanmean(hcfs)
        if np.all(np.isnan(hcfs)): continue
        detinds = np.array(np.floor(getOmegasBlen(res,hcfs)[0]),int)
        results.append(res)
        hcfres.append(hcfs)
        found = False
        for d,h,t,c,hf in zip(detinds,res.hcfield,res.tailonhe,res.current,hcfs):
            isless = h[d]<hf
            start = d-1+isless
            end = d+1+isless
            pfit = np.polyfit(h[start:end],t[start:end],1)
            if np.polyval(pfit,hf)>1:
                iths[i] = c
                break
            
    return results, hcfres, iths

def boschAnalysisVrf(vrf,triplelc=False,parked_main=0,**kwargs):
    """
    Stability analysis at a given RF voltage.
    *vrf* - the RF voltage
    *triplelc*=True - If true, assume 3 Landau cavities. Otherwise, assume 2.
    *parked_main*=0 - If nonzero, calculate with an inactive main cavity parked at the frequency given
                      by the value of parked_main.
    **kwargs - Keyword arguments to pass to scanDetune.BoschAnalysis
    """

    #dets = np.arange(40,2405,40)
    #dets = 1/np.arange(0.0005,0.0251,0.0005)

    kwargs.setdefault('current',np.arange(0.05,0.525,0.05))
    kwargs.setdefault('omegasapprox',False)
    kwargs.setdefault('zerofreq',False)
    kwargs.setdefault('deltinsts',True)
    kwargs.setdefault('flatvrf',False)
    if parked_main:
        kwargs.update({'additional_resonator':(620.8e3/2.,parked_main,3688.)})#99.418
        #kwargs.update({'additional_resonator':(332.45e3,parked_main,3950.)})#99.418
    
    dets = 1/np.arange(0.02,0.16,0.002)**2
    kwargs.setdefault('detune',dets)
    #dets = 1/np.arange(0.06,0.2,0.002)[::5]**2
    brentthresh = 830
    cavfile = 'cav_params_2xMAXIV_4main.json'
    if triplelc:
        dets = dets*3/2.
        brentthresh = brentthresh*3/2.
        cavfile = 'cav_params_5main.json'
    sd.vrf = vrf
    sds.vrf = vrf
    ba = scanDetune.BoschAnalysis(sd,cavfile,scaledetune=True,sringsingle=sds,brentthreshold=brentthresh,
                                  use_boschwr=False,**kwargs)
    ba.getVrfDetune()
    #teres = np.array([[haissinski.heInstability(t,sd) for t in te] for te in ba.tinsts])

    return ba

def flatPotential(ba,rsqfact=None,nharm=3):
    """
    Determine the harmonic-cavity voltage at flat potential by linearly extrapolating to where the incoherent synchrotron
    tune becomes undefined.
    *ba* - A scanDetune.BoschAnalysis instance
    *rsqfact*=None - The total shunt impedance and quality factor of the Landau cavities.
    *nharm*=3 - The operational RF harmonic of the Landau cavities.
    """

    hcfields = np.zeros(len(ba.hcfield))
    sdlocal = utility.StorageRing('/Users/fracul/Physics/Instabilities/machines/maxiv_delivery201901.inp')    
    if rsqfact==None:
        cb = ba.cbms[0][0]
        qfact = cb.omegar[1]/cb.alpha[1]/2.
        rs = cb.rs[1]
    else:
        rs = rsqfact[0]
        qfact = rsqfact[1]
    for i,(lt,hcf,fts) in enumerate(zip(ba.ltunebar,ba.hcfield,ba.ffactbar)):
        #lt = np.array([t.ltune for t in b])
        #fts = np.array([np.absolute(t.formfact[0,1]) for t in b])
        isnan_lt = np.isnan(lt)
        if not np.any(isnan_lt):
            hcfields[i] = np.nan
            continue
        flatind = np.amin(np.where(isnan_lt)[0])
        pfit = np.polyfit(hcf[[flatind-1,flatind-2]],lt[[flatind-1,flatind-2]],1)
        ffact = np.absolute(fts[flatind-1])
        sdlocal.vrf = 1*ba.vrf[0]
        k = cavity_tuning.semiFlat(rs,qfact,nharm,sdlocal,ffact,k_out=True)[1]
        hcfields[i] = k*ba.vrf[0]
        #hcfields[i] = hcf[flatind-1]

    return hcfields

def getOmegasBlen(inst,hcflat,interp=False):
    """
    Determine the index in a series of Landau-cavity detunings evaluated in a scanDetune.BoschAnalysis
    instance at which the RF is closest to flat potential. From these indices, calculate the bunch length
    and the incoherent synchrotron frequency (from the ratio of the bunch length and the energy spread as
    well as the momentum compaction.
    *inst* - scanDetune.BoschAnalysis instance
    *hcflat* - HC voltage corresponding to the flat potential
    *interp* - If True, interpolate between detunings to arrive at a non-integer index (bunch length and
               incoherent synchrotron frequency correspond to integer index).
    """

    omegas = np.zeros(len(hcflat))
    blen = np.zeros(len(hcflat))
    detinds = np.zeros(len(hcflat),float)
    for i,(h,f) in enumerate(zip(hcflat,inst.hcfield)):
        if interp:
            try:
                detind0 = np.where(f-h<0)[0][np.nanargmin(np.absolute((f-h)[f-h<0]))]
                detind1 = np.where(f-h>0)[0][np.nanargmin(np.absolute((f-h)[f-h>0]))]
            except(ValueError, IndexError):
                detind = np.nanargmin(np.absolute(f-h))
            else:
                detind = detind0+(h-f[detind0])/(f[detind1]-f[detind0])
                print(detind0, detind1, detind)
        else:
            detind = np.nanargmin(np.absolute(f-h))
        omegas[i] = inst.sring.espread*inst.sring.alphac/inst.blenbar[i,int(detind)]
        blen[i] = inst.blenbar[i,int(detind)]
        detinds[i] = 1*detind
    return detinds, omegas, blen

def flatPotentialBunchLength(vrf,nharm=3):
    """
    Calculate the buunch length in a quartic potential for a given RF voltage.
    *vrf* - RF voltage
    *nharm*=3 - The operational RF harmonic of the Landau cavities.
    """
    qso = np.sqrt(sd.alphac*sd.nbunch*sd.vrf*np.sqrt(1-sd.eloss**2/sd.vrf**2)/2./np.pi/sd.energy)
    return 1/gammaquart*(3./(nharm*nharm-1))**(1/4.)*np.sqrt(sd.alphac*sd.nbunch*sd.espread/qso)/np.sqrt(np.pi)/sd.frf

def getLindbergThreshold2(inst,hcflat,startind=0,controlplot=False,saveOutput=None,fitparam=np.imag,ax=None,ax2=None,fitorder=1,twoCross=False,lb=None,use_vlasov=False,highlightpoint=False,interp=False):
    """
    Determine the threshold current at which the mode-1 instability is predicted to cross the LLandau contour.
    *inst* - A scanDetune.BoschAnalysis instance.
    *hcflat* - The harmonic cavity voltage corresponding to flat potential (should not contain NaNs)
    *startind*=0 - Ignore harmonic cavity detunings and indexes below this integer number.
    *controlplot*=False - If True, draw two control plots displaying the calculated growth rates and the Landua contour
                          and the polynomial fit used to determine at what beam current they cross. Can also be set to 2
                          for  a slightly different looking plot.
    *saveOutput*=False - If True, save the output coplex coherent tune shifts.
    *fitparam*=np.imag - Choose the fit parameter used to determine at what current the complex coherent tune shift
                         crosses the Landau stability contour.
    *ax*=None - pylab.Axis instance to add the first control plot to an existing figure if desired.
    *ax2*=None - pylab.Axis instance to add the second control plot to an existing figure if desired.
    *fitorder*=1 - Order of the polynomial fit
    *twoCross*=False - If True, only use last point within the Landau contour and the first point outside in the polynomial
                       fit (fitorder must be equal to 1)
    *lb*=None - Can be used to provide a Landau contour as input.
    *use_vlasov*=False - Calculate complex coherent tune shifts using the Vlasov theory intead of Thompson/Ruth.
    *highlightpoint*=False - Draw a ring around one point in the control plot by setting this value to the index of the point to be
                             highlighted.
    *interp*=False - pass this boolean value to the flatPotential function.
    """
    
    from pylab import figure, cm

    #Determine whether loading from BoschAnalysis instance or array
    if isinstance(inst,np.ndarray):
        curr = inst[:,0]
        hcflat = inst[:,1]
        mode1 = inst[startind:,2:5:2]+1j*inst[startind:,3:6:2]
        omegasf = sd.alphac*sd.espread/flatPotentialBunchLength(1e6)
        saveOutput = None
    else:
        detinds, omegas, bl = getOmegasBlen(inst,hcflat,interp)
        #bl[detinds<7] = inst.blenbar[detinds<7,detinds]
        #omegas[detinds<7] = inst.sring.espread*inst.sring.alphac/bl
        #detinds[detinds<7] = 7
        print(detinds)
        omegasf = sd.alphac*sd.espread/flatPotentialBunchLength(inst.vrf[0])
        if interp:
            floordetind = np.array(np.floor(detinds),int)
            ceildetind = np.array(np.floor(detinds),int)
            omegas0 = inst.sring.espread*inst.sring.alphac/inst.blenbar[np.arange(len(hcflat)),floordetind]
            omegas1 = inst.sring.espread*inst.sring.alphac/inst.blenbar[np.arange(len(hcflat)),ceildetind]
            fracdiff = detinds%1
        else:
            detinds = np.array(detinds,int)
        if use_vlasov:
            if interp:
                mode1 = ((inst.ruthd1_v[np.arange(startind,len(hcflat)),floordetind[startind:],:].T-omegas0[startind:])*(1-fracdiff)
                         +(inst.ruthd1_v[np.arange(startind,len(hcflat)),ceildetind[startind:],:].T-omegas1[startind:])*fracdiff).T
            else: mode1 = (inst.ruthd1_v[np.arange(startind,len(hcflat)),detinds[startind:],:].T-omegas[startind:]-0*omegasf).T
        else:
            if interp:
                mode1 = ((inst.ruthd1[np.arange(startind,len(hcflat)),floordetind[startind:],:].T-omegas0[startind:])*(1-fracdiff)
                         +(inst.ruthd1[np.arange(startind,len(hcflat)),ceildetind[startind:],:].T-omegas1[startind:])*fracdiff).T
            else: mode1 = (inst.ruthd1[np.arange(startind,len(hcflat)),detinds[startind:],:].T-omegas[startind:]-0*omegasf).T
        curr = inst.current

    if np.all(lb==None): scaledlberg = lindberg.lindbergIntegral(1/sd.taue/omegasf,delta=0.01)*omegasf
    else: scaledlberg = lb*omegasf
    lbergfreqs = np.arange(-2.5,2.5,0.01)

    #Save output data that can later be used to calculate thresholds
    if saveOutput!=None:
        mode1_v = (inst.ruthd1_v[np.arange(startind,len(hcflat)),detinds[startind:],:].T-omegas[startind:]-0*omegasf).T
        outres = np.zeros((len(inst.current),10))
        outres[:,0] = inst.current
        outres[:,1] = hcflat
        outres[:,2:5:2] = mode1.real
        outres[:,3:6:2] = mode1.imag
        outres[:,6:9:2] = mode1_v.real
        outres[:,7:10:2] = mode1_v.imag
        np.savetxt(saveOutput,outres)

    #Determine the point where the mode 1 growth rate intercepts the Lindberg contour
    mode1 = mode1[np.arange(mode1.shape[0]),np.argmax(mode1.imag,axis=1)]
    #scaledlberg = lberg*omegasf        
    posgrow = (scaledlberg.imag>0)
    sclberg = scaledlberg[posgrow]
    lbgfreqs = lbergfreqs[posgrow]
    
    if twoCross:
        #Determine where the mode 1 growth rate crosses the Lindberg contour
        iswithin = lambda x: x.imag-sclberg.imag[np.argmin(np.absolute(np.real(x)-sclberg.real))]
        withinarr = np.zeros(len(curr),float)
        for i,m in enumerate(mode1):
            withinarr[i] = iswithin(m)
        doescross = np.where(np.sign(withinarr[:-1])!=np.sign(withinarr[1:]))[0]
        if len(doescross)>0:
            stind = doescross[0]
            eind = doescross[0]+2
            #posgrow &= (scaledlberg.real>min(mode1[stind].real,mode1[eind-1].real)) & (scaledlberg.real<max(mode1[stind].real,mode1[eind-1].real))
        elif np.all(withinarr>0):
            stind = 0
            eind = 2
        else:
            stind = len(mode1)-2
            eind = len(mode1)
    else:
        stind = 0
        eind = len(mode1)


    mode1fit = mode1[stind:eind]
    posgrow &= ((scaledlberg.real-np.mean(mode1fit.real))/np.std(mode1fit.real)<5)
    ##if twoCross and len(doescross>0):
    ##    posgrow &= (scaledlberg.imag>np.amin(mode1fit.imag)) & (scaledlberg.imag<np.amax(mode1fit.imag)) \
    ##               & (scaledlberg.real>np.amin(mode1fit.real)) & (scaledlberg.real<np.amax(mode1fit.real))
    sclberg = scaledlberg[posgrow]
    lbgfreqs = lbergfreqs[posgrow]
    filt = np.ones(len(mode1fit),bool)
    isdumb = np.where(mode1fit.imag[1:]-mode1fit.imag[:-1]<0)[0]
    if len(isdumb)>0 and isdumb[-1]>0:
        filt[isdumb[-1]+1:] = False
    #filt[1:] = mode1fit.imag[1:]-mode1fit.imag[:-1]>0
    #print filt
    pfit = np.polyfit(mode1fit.real[filt],mode1fit.imag[filt],fitorder)
    stabdistind = np.argmin(np.absolute(sclberg.imag-np.polyval(pfit,sclberg.real)))
    print(stabdistind)
    omega_coherent = lbgfreqs[stabdistind]*2**(5/4.)*gamma(0.75)**2/gamma(0.25)*omegasf

    pfit2 = np.polyfit(curr[startind+stind:startind+eind][filt],fitparam(mode1fit[filt])-fitparam(sclberg[stabdistind]),fitorder)
    thresh = np.roots(pfit2)
    if np.any(np.isreal(thresh)): thresh = thresh[np.isreal(thresh)]
    thresh = thresh[np.argmin(np.absolute(thresh-np.mean(curr)))]

    if controlplot:
        if ax==None:
            f = figure()
            ax = f.add_subplot(111)
            ax.plot(scaledlberg.real/2./np.pi,scaledlberg.imag,'-k')
        ax.plot(mode1.real/2./np.pi,mode1.imag,':',color='k',zorder=1)
        vmax = np.amax(curr[startind+stind:startind+eind])*1e3
        cd = ax.scatter(mode1.real/2./np.pi,mode1.imag,marker='x',
                        c=curr[startind:]*1e3,
                        cmap='jet',vmin=0,vmax=vmax,linewidth=2,s=48.,zorder=2)
        if controlplot<2:
            ax.plot(mode1fit.real/2./np.pi,np.polyval(pfit,mode1fit.real),'-')
            ax.plot(sclberg[stabdistind].real/2./np.pi,sclberg[stabdistind].imag,'o',ms=10)
        else:
            if highlightpoint:
                color = cm.jet(curr[startind+stind:startind+eind][highlightpoint]*1e3/vmax)
                ax.plot(mode1.real[highlightpoint]/2./np.pi,mode1.imag[highlightpoint],'o',ms=12,color='k',mfc='none')#,color=color)
                #ax.annotate("200 mA",(mode1.real[highlightpoint]/2./np.pi,mode1.imag[highlightpoint]+7),fontsize=20)
            ax.set_ylim(0,300)
            ax.set_xlim(-200,300)

        ax.set_xlabel(r'$\rm Re(\lambda_0)$ (Hz)')
        ax.set_ylabel(r'$\rm Im(\lambda_0)$ (Hz)')
        cbar = f.colorbar(cd)
        cbar.set_label('Beam current (mA)',rotation=270,va='bottom')
        ax.legend(ax.lines[:1]+[cd],('Landau\ncontour','$\lambda_0$'),loc=1)

        if ax2==None:
            f = figure()
            ax2 = f.add_subplot(111)
        ax2.plot(curr[startind:],fitparam(mode1),'x')
        ax2.plot(curr[startind+stind:startind+eind],np.polyval(pfit2,curr[startind+stind:startind+eind])+fitparam(sclberg[stabdistind]),'-')
        ax2.axhline(fitparam(sclberg[stabdistind]))
        ax2.axvline(thresh)
        ax2.set_xlabel('Beam current (A)')
        if fitparam==np.absolute:
            ax2.set_ylabel('Coherent mode amplitude (Hz)')
        elif fitparam==np.real:
            ax2.set_ylabel('Coherent frequency shift (Hz)')
        elif fitparam==np.imag:
            ax2.set_ylabel('Growth rate (Hz)') 
            
    if np.all(lb==None):
        return thresh, omega_coherent, scaledlberg/omegasf
    return thresh, omega_coherent

def getTailongThreshold(inst,hcflat,controlplot=False):
    """
    Estimate the threshold current using the Tianlong He method by interpolating the results of a
    scanDetune.BoschAnalysis instance. Similar to the boschAnalysisAtMinimumCurrent function but
    takes a scanDetune.BoschAnalysis instance as input rather than creating one.
    *inst* - a scanDetune.BoschAnalysis instance.
    *hcflat* - Landau voltage at flat potential (for example, using the flatPotential function)
    *controlplot*=False - If True, draw a plot illustrating the linear fit used to determine the threshold current.
    """    

    from pylab import figure

    if controlplot:
        f = figure()
        ax = f.add_subplot(111)
    
    heres = np.zeros(len(inst.hcfield))
    for i,(h,c,f) in enumerate(zip(inst.hcfield,inst.tailonhe,hcflat)):
        hcind = np.nanargmin(np.absolute(h-f))
        pfit = np.polyfit(h[hcind-1:hcind+2],c[hcind-1:hcind+2],1)
        heres[i] = np.polyval(pfit,f)
        if controlplot:
            ax.plot(h[hcind-1:hcind+2],c[hcind-1:hcind+2],'xC'+str(i))
            ax.plot(h[hcind-1:hcind+2],np.polyval(pfit,h[hcind-1:hcind+2]),'-C'+str(i))

    pfitc = np.polyfit(inst.current,heres,1)
    thresh = (1-pfitc[1])/pfitc[0]

    if controlplot:
        f = figure()
        ax = f.add_subplot(111)
        ax.plot(inst.current,heres,'x')
        xaxe = np.zeros(inst.current.shape[0]+1)
        xaxe[1:] = inst.current
        ax.plot(xaxe,np.polyval(pfitc,xaxe),'-')

    return thresh

def getLindbergThreshold(inst,hcflat,startind=0,niter=0,**kwargs):
    """
    Deprecated version of getLindbergThreshold2
    """

    detinds, omegas, bl = getOmegasBlen(inst,hcflat)
    detinds = np.array(detinds,int)
    omegasf = sd.alphac*sd.espread/flatPotentialBunchLength(inst.vrf[0])
    mode1 = (inst.ruthd1[np.arange(startind,len(hcflat)),detinds[startind:],:].T-omegas[startind:]-0*omegasf).T
    stabdist = mode1[np.arange(mode1.shape[0]),np.argmax(mode1.imag,axis=1)]-np.outer(lberg*omegasf,np.ones(mode1.shape[0]))#np.mean(np.outer(lberg,omegas[startind:]),axis=1)
    stabdist = stabdist[np.argmin(np.absolute(stabdist),axis=0),np.arange(stabdist.shape[1])]
    #Tracer()()

    threshes = np.zeros(2)
    for n in range(2):
        if n==0:
            stb = np.real(stabdist)
        else:
            stb = np.imag(stabdist)
        mindistind = np.argmin(np.absolute(stb))
        if len(inst.current)==2:
            currinds = range(2)
        elif (-1)**n*stb[mindistind]<0 or mindistind==len(stb)-1:
            currinds = np.array([mindistind-1,mindistind],int)
        else:
            currinds = np.array([mindistind,mindistind+1],int)                
        pfit = np.polyfit(inst.current[currinds+startind],stb[currinds],1)
        threshes[n] = -pfit[1]/pfit[0]
        #return inst.current, stb, pfit
        
    thresh = np.mean(threshes)
    #return inst.current, stabdist, pfit

    if niter>0:
        currdiff = (isnt.current[1]-inst.current[0])/2.
        ba = boschAnalysisVrf(inst.vrf[0],current=inst.current[currinds],**kwargs)
        hcflat = flatPotential(ba)
        return getLindbergThreshold(ba,hcflat,niter=niter-1,**kwargs)
    
    return thresh

def thresholdSolve(vrfs,*args,**kwargs):
    """
    Deprecated
    """

    tailongres = np.zeros(len(vrfs))
    lbergres = np.zeros(len(vrfs))
    
    for i,v in enumerate(vrfs):
        inst = boschAnalysisVrf(v,*args,**kwargs)
        hcf = flatPotential(inst)
        lbergres[i] = getLindbergThreshold2(inst,lberg,hcf)
        tailongres[i] = getTailongThreshold(inst,hcf)
        del(inst)

    return lbergres, tailongres
        
def fieldThreshold(inst):
    """
    Deprecated
    """
    
    currs = inst.current[:]
    threshes = np.zeros(len(currs))
    thresh_the = np.zeros(len(currs))
    for i,(h,r,t) in enumerate(zip(inst.hcfield,inst.ruthd1.imag,inst.tailonhe)):
        r = np.amax(r,axis=1)
        fnd = np.where(r>40.)[0]
        if len(fnd)>0:
            firstind = fnd[0]
            threshes[i] = (h[firstind]*(r[firstind]-40.)+h[firstind-1]*(40.-r[firstind-1]))/(r[firstind]-r[firstind-1])

        fnd_he = np.where(t>1)[0]
        if len(fnd_he)>0:
            firstind = fnd_he[0]
            thresh_the[i] = (h[firstind]*(t[firstind]-1.)+h[firstind-1]*(1.-t[firstind-1]))/(t[firstind]-t[firstind-1])

    return currs, threshes, thresh_the

def mode1Impedance(inst):
    """
    Calculate the Cavity Impedances at the +/-1 revolution harmonics for all the points covered by a scanDetune.BoschAnalysis
    instance.
    *inst* - scanDetune.BoschAnalysis instance
    """
    #freqp1 = inst.sring.frf*(3+1./inst.sring.nbunch)
    #freqm1 = inst.sring.frf*(3-1./inst.sring.nbunch)

    imped = np.zeros((2,4)+np.shape(inst.cbms),complex)
    #impm = np.zeros((4,)+np.shape(inst.cbms),complex)
    for i,ce in enumerate(inst.cbms):
        for j,c in enumerate(ce):
            for n,(r,a,w) in enumerate(zip(c.rs,c.alpha,c.omegar)):
                qf = w*0.5/a
                nharm = np.round(w/2./np.pi/inst.sring.frf)
                freqp1 = inst.sring.frf*(nharm+1./inst.sring.nbunch)
                freqm1 = inst.sring.frf*(nharm-1./inst.sring.nbunch)
                im = haissinski.ImpedanceModel(rs=r,rfreq=w/2./np.pi,qfact=qf)
                imped[0,n,i,j] = im.calcImpedance(freqp1)
                imped[1,n,i,j] = im.calcImpedance(freqm1)
            imped[0,-1,i,j] = np.sum(imped[0,:-1,i,j])
            imped[1,-1,i,j] = np.sum(imped[1,:-1,i,j])

    return imped

def plotImpedance(inst,imped,currind=0):
    """
    Plot the results of the mode1Impedance function
    *inst* - scanDetune.BoschAnalysis instance
    *imped* - The impedance returned by the mode1Impedance function
    *currint* - The index of the currents in the scanDetune.BoschAnalysis instance
    """

    from pylab import figure, setp

    f = figure()
    ax = f.add_subplot(111)
    xaxvals = inst.hcfield[currind]/1e3

    ax.plot(xaxvals,imped[0,1,currind].real/1e3,'-C0')
    ax.plot(xaxvals,imped[1,1,currind].real/1e3,'--C0')
    ax.plot(xaxvals,imped[0,2,currind].real/1e3,'-C1')
    ax.plot(xaxvals,imped[1,2,currind].real/1e3,'--C1')

    ax.set_xlabel('Landau voltage (kV)')
    ax.set_ylabel(r'Real impedance ($\rm k\Omega$)')

    art = ax.legend(('Mode +1','Mode -1'),loc=2,bbox_to_anchor=(0.0,0.5))
    ax.legend(ax.lines[::2],('Landau','Parked main'),loc=1,bbox_to_anchor=(1.0,0.9),title='Cavity')
    setp(ax.legend_.get_title(),size=20)
    ax.add_artist(art)

    f = figure()
    ax = f.add_subplot(111)

    ax.plot(xaxvals,imped[0,1,currind].imag/1e3,'-C0')
    ax.plot(xaxvals,imped[1,1,currind].imag/1e3,'--C0')
    ax.plot(xaxvals,imped[0,2,currind].imag/1e3,'-C1')
    ax.plot(xaxvals,imped[1,2,currind].imag/1e3,'--C1')

    ax.set_xlabel('Landau voltage (kV)')
    ax.set_ylabel(r'Reactive impedance ($\rm k\Omega$)')    

    art = ax.legend(('Mode +1','Mode -1'),loc=4,bbox_to_anchor=(1.0,0.5))
    ax.legend(ax.lines[::2],('Landau','Parked main'),loc=3,bbox_to_anchor=(0.0,0.5),title='Cavity')
    setp(ax.legend_.get_title(),size=20)
    ax.add_artist(art)

def fullImpedance(bainst,currind,detind):
    """
    Calculate not just the impedances at the revolution harmonics but across a frequency band of 200 kHz
    """

    revfreq = bainst.sring.frf/bainst.sring.nbunch
    freqax = np.linspace(-revfreq-100e3,revfreq+100e3,10000)
    c = bainst.cbms[currind][detind]
    imped = np.zeros((5,len(freqax)),complex)
    for n,(r,a,w) in enumerate(zip(c.rs,c.alpha,c.omegar)):
        qf = w*0.5/a
        nharm = np.round(w/2./np.pi/bainst.sring.frf)
        im = haissinski.ImpedanceModel(rs=r,rfreq=w/2./np.pi,qfact=qf)
        imped[n+1] = im. calcImpedance(freqax+nharm*bainst.sring.frf)
    imped[-1] = np.sum(imped[1:-1],axis=0)
    imped[0] = freqax

    return imped

def plotCoherence(inst,currind=0,ax=None,isreal=False,ind=0):
    """
    Deprecated
    """

    from pylab import figure, setp
    
    if ax==None:
        f = figure()
        ax = f.add_subplot(111)
        
    if isreal:
        conv = lambda x: np.real(x)/2./np.pi
    else:
        conv = np.imag
        
    xaxvals = inst.hcfield[currind]/1e3
    numlines = len(ax.lines)/2

    ax.plot(xaxvals,conv(inst.ruthd1_v[currind][:,0]),'-C'+str(numlines))
    ax.plot(xaxvals,conv(inst.ruthd1_v[currind][:,1]),'--C'+str(numlines))
    ax.set_xlabel('Landau voltage (kV)')

    if isreal:
        ax.set_ylabel(r'Coherent frequency (Hz)')
    else:
        ax.set_ylabel(r'Growth rate ($\rm s^{-1}$)')

class Empty:
    """
    Empty class
    """
    pass

def loadCOSMOSDataAsInst(basedir):
    """
    Load data obtained from a cluster computation and put it into a class that resembles a
    scanDetune.BoschAnalysis instance.
    """

    dirs = utility.getOutput('ls '+basedir).split('\n')[:-1]
    currs = []
    vrfs = []
    rs = []
    maxlen = 0    
    for d in dirs:
        metadata = np.loadtxt(basedir+d+'/README')
        print('Parked detuning=%.1f' % metadata[0,0])
        tmpvrfs = metadata[1:,0]        
        tmpcurrs = metadata[1:,1]
        #if isinstance(tmpcurrs,float):
        #    tmpvrfs = [tmpvrfs]
        #    tmpcurrs = [tmpcurrs]
        currs.extend(list(tmpcurrs))
        vrfs.extend(list(tmpvrfs))
        for n in range(len(tmpcurrs)):
            tmpdat = np.loadtxt(basedir+d+'/mode01freqgrate_cstep'+str(n)+'.dat')
            if len(tmpdat)>maxlen:
                maxlen = len(tmpdat)
            rs.append(tmpdat)

    for i,r in enumerate(rs):
        if len(r)!=maxlen:
            rs[i] = np.ones((maxlen,r.shape[1]))*np.nan
            rs[i][:r.shape[0]] = r[:]

    rs = np.array(rs)
    currs = np.array(currs)
    vrfs = np.array(vrfs)

    bainst = Empty()

    bainst.vrf = vrfs
    bainst.current = currs
    bainst.deltaf = rs[:,:,0]
    bainst.hcfield = rs[:,:,1]
    bainst.blenbar = rs[:,:,2]
    bainst.ltunebar = rs[:,:,3]
    bainst.ffactbar = rs[:,:,4]
    bainst.ruthd = rs[:,:,5]+1j*rs[:,:,6]
    bainst.ruthq = rs[:,:,7]+1j*rs[:,:,8]
    bainst.ruthd1 = np.zeros(rs[:,:,9].shape+(2,),complex)
    bainst.ruthd1[:,:,0] = rs[:,:,9]+1j*rs[:,:,10]
    bainst.ruthd1[:,:,1] = rs[:,:,11]+1j*rs[:,:,12]
    if rs.shape[-1]>13:
        bainst.ruthd_v = rs[:,:,13]+1j*rs[:,:,14]
        bainst.ruthd1_v = np.zeros(rs[:,:,15].shape+(2,),complex)
        bainst.ruthd1_v[:,:,0] = rs[:,:,15]+1j*rs[:,:,16]
        bainst.ruthd1_v[:,:,1] = rs[:,:,17]+1j*rs[:,:,18]
    bainst.sring = sd

    return bainst
    


    

    


    
