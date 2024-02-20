from pylab import figure, setp
import lindberg
import utility
import numpy as np
import matplotlib as mpl
import vrfCurrScan

#with open('driveDampCavity01_wError.dat','r') as f:
#    fittxt = f.readline()
#ddampfit = np.array(fittxt[fittxt.find('=')+1:].strip('\n').split(','),float)
#ddampfit[1] = 
#potmid = ddampfit[2]
#freqgrad = 99.931e6/3688.*ddampfit[3]/2.
#freqgrad = 0.1862*1e6
#pot2freq = lambda x: freqgrad*(x-potmid)

def grateImage(gres,hcflat,inst):
    """
    Plot function not included in manuscript
    """

    f = figure()
    ax = f.add_subplot(111)

    filt = np.ones(np.shape(gres[2]),bool)
    for i in range(gres[0].shape[0]):
        ffacts = np.array([np.mean(np.absolute(t.formfact[:,1])) for t in inst.tinsts[i]])
        filtstart = np.where((ffacts[1:]-ffacts[:-1]>0)[35:])[0]
        if len(filtstart)>0:
            filt = np.where(gres[1]>inst.hcfield[i][filtstart[0]+34])[0][0]
            gres[2][i][filt:] = np.nan
    
    currdiff = (gres[0][1]-gres[0][0])/2.
    fielddiff = (gres[1][1]-gres[1][0])/2.
    gres[2][gres[2]<0.001] = 0.0005
    im = ax.imshow(np.log10(gres[2][-1::-1]),aspect='auto',cmap='jet',
                   extent=((gres[1][0]-fielddiff)/1e3,(gres[1][-1]+fielddiff)/1e3,
                           (gres[0][0]-currdiff)*1e3,(gres[0][-1]+currdiff)*1e3),
                   vmin=-3)
    ax.set_xlabel('Landau voltage (kV)')
    ax.set_ylabel('Beam current (mA)')

    ax.plot(hcflat/1e3,gres[0]*1e3,'-wx',lw=2)
    ax.set_xlim(49,370)
    leg = ax.legend(('Semiflat potential',),loc=4,bbox_to_anchor=(1,1),facecolor='grey',framealpha=0.5)

    f.subplots_adjust(left=0.15,right=0.92)
    cbar = f.colorbar(im,extend='min',extendfrac=0.03)
    cbar.set_label(r'Growth rate ($\rm s^{-1}$)',rotation=270,va='bottom')
    cbar.set_ticks(cbar.get_ticks())
    cbar.set_ticklabels(['$10^{%d}$' % c for c in cbar.get_ticks()])

def theoryPlot(inst,currind=5):
    """
    Plot function not used in manuscript
    """

    f = figure()
    ax = f.add_subplot(111)
    f.subplots_adjust(right=0.88)

    #ffacts = np.array([np.mean(np.absolute(t.formfact[:,1])) for t in inst.tinsts[currind]])
    ffacts = np.absolute(inst.ffactbar)
    filt = np.ones(len(inst.hcfield[currind]),bool)
    filtstart = np.where((ffacts[1:]-ffacts[:-1]>0)[35:])[0]
    if len(filtstart)>0: filt[filtstart[0]+34:] = False

    ax.semilogy(inst.hcfield[currind][filt]/1e3,np.amax(inst.ruthd1_v[currind].imag,axis=1)[filt],'-')
    ax.semilogy(inst.hcfield[currind][filt]/1e3,inst.venturini[currind].imag[filt],'-')
    ax2 = ax.twinx()
    ax2.plot(inst.hcfield[currind][filt]/1e3,inst.tailonhe[currind][filt],'-C2')

    ax.set_xlabel('Landau voltage (kV)')
    ax.set_ylabel(r'Growth rate ($\rm s^{-1}$)')
    ax2.set_ylabel('T. He prediction',rotation=270,va='bottom')

    ax.set_ylim(0.3,1500)
    ax2.set_ylim(0,1)

    ax.legend(f.axes[0].lines+f.axes[1].lines,('Cullinan et al.','Venturini approx.','T. He prediction'),loc=3)

def lindbergPlot(inst,lbergres=[],taue=0.025194,wsfromblen=False,detind=4,startind=0,ax=None,hcflat=None,krinsky=False):
    """
    Plot function not used in manuscript
    """

    if ax==None:
        f = figure()
        ax = f.add_subplot(111)

    if lbergres:
        tlzerodamp, tlraddamp = lbergres
    else:
        tlzerodamp = lindberg.lindbergIntegral(0)

    arrsize = len(inst.ruthd1_v)-startind
    if not np.all(hcflat==None):
        hcflat[np.isnan(hcflat)] = 0
        detinds = np.array(np.nanargmin(np.absolute(inst.hcfield.T-hcflat),axis=0),int)[startind:]
    else:
        detinds = np.ones(arrsize,int)*detind
        
    lineres = np.zeros(arrsize,complex)
    if inst.omegasapprox:
        if krinsky:
            mode1iter = inst.ruthd1_v[startind:,:,:]
        else:
            mode1iter = inst.ruthd1[startind:,:,:]            
    else:
        if krinsky:
            mode1iter = inst.vbms[startind:]
        else:
            mode1iter = inst.cbms[startind:]

    cmap = mpl.cm.jet
    omegasf = inst.sring.espread*inst.sring.alphac/vrfCurrScan.flatPotentialBunchLength(inst.vrf[0])
    for i,(t,r,d) in enumerate(zip(inst.blenbar[startind:],mode1iter,detinds)):
        omegas = inst.sring.espread*inst.sring.alphac/t[d]
        print d, omegas        
        if wsfromblen: omegasf = 1*omegas
        if i==0 and not lbergres: tlraddamp = lindberg.lindbergIntegral(1/taue/omegasf)
        ax.plot(omegasf*(tlzerodamp.real+0*1)/2/np.pi,omegasf*tlraddamp.imag,'-',c=cmap(inst.current[i+startind]/np.amax(inst.current)))
        if not inst.omegasapprox:
            #lineres[i] = r[d].eigenfreqs[np.argmax(r[d].eigenfreqs.imag)]
            #if krinsky:
            mode1 = r[d].eigenfreqs[[np.argmin(np.absolute(r[d].eigenmodes_lplcenum-1)),np.argmin(np.absolute(r[d].eigenmodes_lplcenum-175))]]
            lineres[i] = mode1[np.argmax(mode1.imag)]
            #else:
            #    lineres[i] = r[d].eigenfreqs[np.where((r[d].eigenmodes_lplcenum==1) | (r[d].eigenmodes_lplcenum==175))[0][:2]]
            lineres[i] /= 2*omegas
        else:
            lineres[i] = r[d][np.argmax(r[d].imag)]
            lineres[i] -= omegas

    ax.plot((lineres.real+0*omegas)/2./np.pi,lineres.imag,':',lw=0.5,color='k')
    sc = ax.scatter((lineres.real+0*omegas)/2./np.pi,lineres.imag,marker='x',
                    c=inst.current[startind:]*1e3,cmap='jet',vmin=0,vmax=np.amax(inst.current)*1e3,s=48.,linewidths=2)
    ax.legend(ax.lines[:1]+[sc],('Landau\ncontour','Mode 1'),loc=2)

    cbar = f.colorbar(sc)
    cbar.set_label('Beam current (mA)',rotation=270,va='bottom')

    ax.set_xlabel('$\mathrm{Re}(\lambda)$ (Hz)')
    ax.set_ylabel('$\mathrm{Im}(\lambda)$ ($\mathrm{s}^{-1}$)')

    ax.set_ylim(0,250)
    ax.set_xlim(-275,200)
    
    return tlzerodamp, tlraddamp

def plotThresholds(dirname,ax=None):
    """
    Plot function not used in manuscript
    """

    if not dirname.endswith('/'):
        dirname += '/'
    lberg = np.loadtxt('landau_contour_store.txt')
    lb = lberg[:,2]+1j*lberg[:,3]
    if ax==None:
        f = figure()
        ax = f.add_subplot(111)
    colorstr = 'C'+str(len(ax.lines)/2)
    fles = utility.getOutput('ls '+dirname).split('\n')[:-1]
    threshes = np.zeros(len(fles))
    omegar = np.zeros(len(fles))
    pfs = np.zeros(len(fles))
    lb = None
    for i,f in enumerate(fles):
        tst = np.loadtxt(dirname+f)
        if i==0:
            threshes[i], omegar[i], lb = vrfCurrScan.getLindbergThreshold2(tst,lb,controlplot=False,startind=0,fitparam=np.absolute,fitorder=1,twoCross=True)
        else:
            threshes[i], omegar[i]  = vrfCurrScan.getLindbergThreshold2(tst,lb,controlplot=False,startind=0,fitparam=np.absolute,fitorder=1,twoCross=True,lb=lb)            
        if i!=len(fles)-1:
            pfs[i] = float(f[-12:-4])
        
    pfs = 99.931e6-99.931e6/176.+np.arange(-280e3,281e3,10e3)
    xaxe = (pfs-99.931e6+99.931e6/176)/1e3    
    ax.plot(xaxe,np.array(threshes[:-1])*1e3,'-'+colorstr)
    ax.set_xlabel('Parked cavity detuning (kHz)')
    ax.set_ylabel('Threshold current (mA)')
    ax.axhline(threshes[-1]*1e3,ls='--',lw=2,color=colorstr)
    ax.annotate('No parked main cavity',(xaxe[-1]-10,threshes[-1]*1e3-10),va='top',ha='right',size=20)
    ax.set_xlim(0,290)
    ax.set_ylim(0,700)
    ax.set_yticks(np.arange(0,610,200))

def prepVrfIthPlot(dirname,lb=None,rs=8.25e6,**kwargs):
    """
    For figure6: prepVrfIthPlot('COSMOS_results/RFVoltageThreeCavities/noparking_fine')
    For figure10: prepVrfIthPlot('COSMOS_results/noparking_4main_2landaus_currzoom/',rs=5.5e6)
    """

    dirs = utility.getOutput('ls '+dirname).split('\n')[:-1]
    dets = []
    res = []
    reswr = []
    for d in dirs:
        print d
        bae = vrfCurrScan.loadCOSMOSDataAsInst(dirname+'/'+d+'/')
        hcf = vrfCurrScan.flatPotential(bae,(rs,20800))
        #hcf = bae.hcfield[:,7]
        hcf[np.isnan(hcf)] = np.mean(hcf[~np.isnan(hcf)])
        if np.all(lb==None):
            ith, omegar, b_trash = vrfCurrScan.getLindbergThreshold2(bae,hcf,controlplot=False,startind=0,use_vlasov=True,interp=False,**kwargs)
        else:
            ith, omegar = vrfCurrScan.getLindbergThreshold2(bae,hcf,controlplot=False,startind=0,use_vlasov=True,lb=lb,**kwargs)
        dets.append(float(d[3:-2]))
        res.append(ith)
        reswr.append(omegar)

        del(bae)

    resarray = np.array(zip(dets,res,reswr),dtype=[('dets',float),('iths',float),('omegar',float)])
    resarray = np.sort(resarray,order=['dets'])
        
    return resarray

def getBunchLengths(dirname,vrf,rs=8.25e6):
    dirname += '/Vrf'+str(vrf)+'kV/'
    bae = vrfCurrScan.loadCOSMOSDataAsInst(dirname)
    hcf = vrfCurrScan.flatPotential(bae,(rs,20800))
    hcf[np.isnan(hcf)] = np.mean(hcf[~np.isnan(hcf)])
    #detinds = np.argmin(np.absollute(bae.hcfield.T-hcf),axis=0)
    #detindm1 = detinds+(-1)**(bae.hcfields[np.arange(len(detinds)),detinds]-hcf>0)
    blens = np.zeros(len(bae.hcfield))
    for i,(bl,hc,hf) in enumerate(zip(bae.blenbar,bae.hcfield,hcf)):
        blens[i] = np.interp(hf,hc,bl)
    revangfreq = 2*np.pi*vrfCurrScan.sd.frf/vrfCurrScan.sd.nbunch
    qso = np.sqrt(vrfCurrScan.sd.nbunch*vrfCurrScan.sd.alphac/2./np.pi/vrfCurrScan.sd.energy*bae.vrf[0]*np.sqrt(1-vrfCurrScan.sd.eloss**2/bae.vrf[0]**2))
    blen0 = vrfCurrScan.sd.alphac*vrfCurrScan.sd.espread/qso/revangfreq
    return hcf, blens, blen0

def figure6VrfIthVsExperiment(resarray,includeTiHe=False):
    """
    Takes single input, the output of prepVrfIthPlot
    """

    f = figure()
    ax = f.add_subplot(111)

    a = np.loadtxt('flatThresh_exp.dat',delimiter=',')

    ax.errorbar(a[:,0],a[:,1],ms=8,mew=1,capsize=5,yerr=5,label='Data',ls='none')

    xaxe = [800,1050]
    #filt = resarray['dets']>820
    filt = np.ones(len(resarray['dets']),bool)
    filt &= resarray['dets']<=1050
    #filt &= (resarray['iths']>np.percentile(resarray['iths'],2)) & (resarray['iths']<np.percentile(resarray['iths'],98))
    niter = 0
    print '%d of %d points remaining' % (np.sum(filt),len(filt))    
    for n in range(niter):
        pfit = np.polyfit(resarray['iths'][filt]*1e3,resarray['dets'][filt],1)
        ydiff = resarray['dets']-np.polyval(pfit,resarray['iths']*1e3)
        filt &= np.absolute(ydiff)<np.percentile(np.absolute(ydiff),83)
        print '%d of %d points remaining' % (np.sum(filt),len(filt))
    #filt = np.ones(len(resarray['dets']),bool)

    print resarray['dets'][~filt]
    pfit = np.polyfit(resarray['dets'][filt],resarray['iths'][filt]*1e3,1)
    ax.plot(resarray['iths'][filt]*1e3,resarray['dets'][filt],'.',mew=2,alpha=0.5,label='Theory')
    #ax.plot(resarray['iths'][~filt]*1e3,resarray['dets'][~filt],'.',mew=2)
    ax.plot(np.polyval(pfit,xaxe),xaxe,'-',label='Linear fit')

    hl,lb = ax.get_legend_handles_labels()
    ax.legend(hl[2:]+hl[:2],('Data (mode -1)','Theory (mode +1)','Linear fit'))
    #ax.legend()

    #ax.set_ylim(860,1100)
    #ax.set_xlim(280,350)
    ax.set_xticks(np.arange(280,350,20))
    ax.set_ylim(0,1075)
    
    ax.set_xlabel('Beam current (mA)')
    ax.set_ylabel('Threshold RF voltage (kV)')

def figure6IthVsVrfExperiment(resarray,includeTiHe=False):
    """
    Takes single input, the output of prepVrfIthPlot
    """

    f = figure()
    ax = f.add_subplot(111)

    a = np.loadtxt('flatThresh_exp.dat',delimiter=',')

    if includeTiHe:
        tihedat = np.loadtxt('noLandauTheoryIths_twoLandaus.txt')
        filt = ithedat[:,1]!=0
        pfitth = np.polyfit(tihedat[filt,0],tihedat[filt,1],1)

    ax.errorbar(a[:,1],a[:,0],ms=6,mew=1,capsize=3,xerr=2.5,yerr=5,label='Data',ls='none')

    xaxe = [800,1050]
    #filt = resarray['dets']>820
    filt = np.ones(len(resarray['dets']),bool)
    filt &= resarray['dets']<=1050
    #filt &= (resarray['iths']>np.percentile(resarray['iths'],2)) & (resarray['iths']<np.percentile(resarray['iths'],98))
    niter = 0
    print '%d of %d points remaining' % (np.sum(filt),len(filt))    
    for n in range(niter):
        pfit = np.polyfit(resarray['iths'][filt]*1e3,resarray['dets'][filt],1)
        ydiff = resarray['dets']-np.polyval(pfit,resarray['iths']*1e3)
        filt &= np.absolute(ydiff)<np.percentile(np.absolute(ydiff),83)
        print '%d of %d points remaining' % (np.sum(filt),len(filt))
    #filt = np.ones(len(resarray['dets']),bool)

    print resarray['dets'][~filt]
    pfit = np.polyfit(resarray['dets'][filt],resarray['iths'][filt]*1e3,1)
    ax.plot(resarray['dets'][filt],resarray['iths'][filt]*1e3,'.',mew=2,alpha=0.5,label='Theory')
    #ax.plot(resarray['iths'][~filt]*1e3,resarray['dets'][~filt],'.',mew=2)
    if includeTiHe:
        ax.plot(xaxe,np.polyval(pfitth,xaxe),'-',label='Tianlong He\nprediction')
        ax.plot(xaxe,np.polyval(pfit,xaxe),'-',label='Krinsky theory')
    else:
        ax.plot(xaxe,np.polyval(pfit,xaxe),'-',label='Linear fit')
        
    hl,lb = ax.get_legend_handles_labels()
    ax.legend(hl[2:]+hl[:2],('Data (mode -1)','Theory (mode +1)','Linear fit'))
    #ax.legend()

    #ax.set_ylim(860,1100)
    #ax.set_xlim(280,350)
    #ax.set_xticks(np.arange(280,350,20))
    ax.set_ylim(0,360)
    ax.set_xlim(875,1050)
    ax.set_xticks(np.arange(900,1075,50))
    
    ax.set_ylabel('Beam current (mA)')
    ax.set_xlabel('Threshold RF voltage (kV)')
    
def figure10IthVsVrfExperiment(resarray,includeTiHe=False):
    """
    Takes single input, the output of prepVrfIthPlot
    """

    f = figure()
    ax = f.add_subplot(111)

    a = np.loadtxt('flatThresh_exp_2landaus.dat')

    if includeTiHe:
        tihedat = np.loadtxt('noLandauTheoryIths_twoLandaus.txt')
        filt = tihedat[:,1]!=0
        pfitth = np.polyfit(tihedat[filt,0],tihedat[filt,1],1)    

    ax.errorbar(a[:,1],a[:,0],mew=1,capsize=3,xerr=2.5,yerr=5,label='Data',ls='none')

    xaxe = [800,1200]
    pfit = np.polyfit(resarray['dets'],resarray['iths']*1e3,1)
    ax.plot(resarray['dets'],resarray['iths']*1e3,'.',mew=2,alpha=0.5,label='Theory')
    #ax.plot(resarray['iths'][~filt]*1e3,resarray['dets'][~filt],'.',mew=2)
    ax.plot(xaxe,np.polyval(pfit,xaxe),'-',label='Linear fit')
    if includeTiHe:
        ax.plot(xaxe,np.polyval(pfitth,xaxe)*1e3,'-',label='Tianlong He prediction')
    
    hl,lb = ax.get_legend_handles_labels()
    if includeTiHe:
        ax.legend(hl[3:]+hl[:3],('Data','Krinsky theory','Linear fit','Tianlong He prediction'),loc=2,bbox_to_anchor=(0,0.6))
    else:
        ax.legend(hl[2:]+hl[:2],('Data','Theory','Linear fit'))
    #ax.legend()

    ax.set_xlim(900,1100)
    ax.set_ylim(0,560)
    
    ax.set_ylabel('Threshold current (mA)')
    ax.set_xlabel('RF voltage (kV)')
    
def prepParkedMainPlot(dirnames=['COSMOS_results/parkedmain_4main_2Landaus','COSMOS_results/parkedmain_4main_2landaus_fine'],lb=None):
    """
    """

    dets = []
    res = []    
    for dirname in dirnames:
        dirs=utility.getOutput('ls '+dirname).split('\n')[:-1]
        for d in dirs:
            print d
            bae = vrfCurrScan.loadCOSMOSDataAsInst(dirname+'/'+d+'/')
            hcf = vrfCurrScan.flatPotential(bae,(5.5e6,20800))
            hcf[np.isnan(hcf)] = np.mean(hcf[~np.isnan(hcf)])
            if np.all(lb==None):
                ith, lb = vrfCurrScan.getLindbergThreshold2(bae,hcf,controlplot=False,startind=0,twoCross=True,use_vlasov=True)
            else:
                ith = vrfCurrScan.getLindbergThreshold2(bae,hcf,controlplot=False,startind=0,twoCross=True,lb=lb,use_vlasov=True)
            dets.append(float(d[6:].replace('k','000')))
            res.append(ith)

    resarray = np.array(zip(dets,res),dtype=[('dets',float),('iths',float)])
    resarray = np.sort(resarray,order=['dets'])
        
    return resarray

def prepFigure1TheoryCurves():
    """
    Prepare results for Figure 1 - the predictions of theories that neglect Landau damping
    """
    binst = vrfCurrScan.boschAnalysisVrf(1e6,True,current=np.array([0.2]))
    hcf = np.nanmean(vrfCurrScan.flatPotential(binst,(8.25e6,20800)))

    return binst, hcf

def figure1TheoryCurves(binst,hcfflat):
    """
    binst = vrfCurrScan.boschAnalysisVrf(1e6,True,current=array([0.2]))
    """

    f = figure()
    f.subplots_adjust(left=0.15,right=0.85)
    ax = f.add_subplot(111)

    ax.semilogy(binst.hcfield[0]/1e3,binst.ruthd1[0,:,0].imag,'-')
    ax.semilogy(binst.hcfield[0]/1e3,binst.venturini[0].imag,'-')
    ax2 = ax.twinx()
    ax2.plot(binst.hcfield[0]/1e3,binst.tailonhe[0],'-C2')
    #ax2.set_ylim(0,1.5)

    ax.set_xlabel('Landau voltage (kV)')
    ax.set_ylabel(r'Growth rate ($\rm s^{-1}$)')
    ax.set_xlim(140,365)
    ax2.set_ylabel('Tianlong He prediction',rotation=270,va='bottom')
    ax2.axhline(1,ls=':',color='k')
    ax.axvline(hcfflat/1e3,color='k',ls='--')

    ax.set_ylim(0.45,1300)
    xl = ax.get_ylim()
    x2p = 1./np.log10(40.0/xl[0])*np.log10(xl[1]/xl[0])
    ax2.set_ylim(0,x2p)
    #ax.set_ylim(xl[0],xl[1])
    ax2.annotate('Stability threshold',(148,1.01),ha='left',va='bottom',fontsize=20)
    ax.annotate('Flat\npotential',(hcfflat/1e3+2,6),ha='left',va='bottom',fontsize=20)
    ax2.set_yticks(np.arange(0,x2p,0.5))
    
    ax.legend(ax.lines[:2]+ax2.lines,('Cullinan et al., 2020','Venturini approximation','Tianlong He prediction'),loc=3)

def prepImpedanceFigure(parked_main=True):
    bainst = vrfCurrScan.boschAnalysisVrf(1e6,True,current=np.array([0.2]),zerofreq=True,deltinsts=True,omegasapprox=True,parked_main=99.931e6*(1-1/176.)+50e3)        
    tst = vrfCurrScan.fullImpedance(bainst,0,18)
    return tst

def prepImpedanceFigureDebugTest(parked_main=True):
    bainst = vrfCurrScan.boschAnalysisVrf(1e6,True,current=np.array([0.2]),detune=np.array([200]),forceflat=True,zerofreq=True,deltinsts=True,omegasapprox=True,parked_main=99.931e6*(1-1/176.)+50e3)
    tst = vrfCurrScan.fullImpedance(bainst,0,0)
    return tst

def prepLindbergPlots():
    """
    Prepare the example plot including complex coherent tune shifts and Landau contours for quartiv potential.
    """
    bae = vrfCurrScan.boschAnalysisVrf(1e6,True,current=np.arange(0.05,0.51,0.05),zerofreq=True,deltinsts=True,omegasapprox=True)
    #bae.getVrfDetune()
    hcf = vrfCurrScan.flatPotential(bae,(8.25e6,20800))
    hcf[np.isnan(hcf)] = np.nanmean(hcf)
    return bae, hcf

def figure2LContour(bae,hcf):
    vrfCurrScan.getLindbergThreshold2(bae,hcf,controlplot=2,startind=0,twoCross=False,use_vlasov=True,highlightpoint=False)

def impedanceFigure(imped,real=True,parked_main=True):
    f = figure()
    ax = f.add_subplot(111)
    #ax2 = ax.twinx()

    freqax = imped[0]/1e3

    for n,i in enumerate(imped[1:-1-(not parked_main)]):
        if real: ax.semilogy(freqax,i.real/1e3,'-C'+str(n))
        else: ax.plot(freqax,-i.imag/1e3,'-C'+str(n))

    ax.set_xlabel('Frequency offset (kHz)')
    if real:
        ax.set_ylabel(r'Real impedance ($\rm k\Omega$)')
    else:
        ax.set_ylabel(r'Reactive impedance ($\rm k\Omega$)')
        ax.set_ylim(-400,400)

    #art = ax.legend(ax.lines[:2],('Real','Reactive'),loc=1)
    if parked_main:
        ax.legend(ax.lines,('Active main','Landau','Parked main'),title='Cavity',loc=3-real)
    else:
        ax.legend(ax.lines,('Active main','Landau'),title='Cavity',loc=3-real)        
    setp(ax.legend_.get_title(),size=20)
    #ax.add_artist(art)

    revfreqkHz = 99.931e6/176./1e3
    ax.axvline(revfreqkHz,ls='--',color='k')
    ax.axvline(0,ls='--',color='k')
    ax.axvline(-revfreqkHz,ls='--',color='k')

    ax.set_xlim(-revfreqkHz-100,revfreqkHz+100)
    ax.set_xticks([-revfreqkHz,0,revfreqkHz])
    ax.set_xticklabels([r'$nf_{\rm rf}-f_0$',r'$nf_{\rm rf}$',r'$nf_{\rm rf}+f_0$'])
    f.subplots_adjust(bottom=0.15,top=0.905)

    if real:
        ax.set_ylim(0.04,110000.0)
    else:
        ax.set_ylim(-510,400)
    
