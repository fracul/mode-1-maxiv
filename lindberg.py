import numpy as np
from scipy import special, optimize

ampnorm_p2by3 = np.arange(0,32,0.00006)
gamma_quarter = special.gamma(0.25)
gamma_3quarter = special.gamma(0.75)
dimfreq = gamma_quarter/2**(5/4.)/gamma_3quarter**2
const = 128*np.pi*np.exp(-np.pi)/gamma_quarter/(1+np.exp(-np.pi))**2
intvar = np.pi**(1/3.)/2**(1/3.)*gamma_3quarter/gamma_quarter*ampnorm_p2by3**(2/3.)    
imoff_const_mag = 0.00005    

def lbergIntegrate(reomeganorm,imoff=0,positive=False):

    imoff_const = imoff_const_mag
    if positive and np.any(reomeganorm<0):
        imoff_const = np.outer(imoff_const_mag*(-1)**(reomeganorm<0),np.ones(len(intvar)))
            
    ksi = np.outer(dimfreq*reomeganorm,np.ones(len(intvar)))
    return 1/(const*np.trapz(intvar**2.5*np.exp(-intvar**2)/(ksi**2-intvar+2*1j*ksi*(imoff_const+dimfreq*imoff)-imoff_const**2-dimfreq*dimfreq*imoff*imoff),
                                      x=intvar))

def getGrowth(normphase,grate=0):

    def chi2(x):
        lbi = lbergIntegrate(x[0],imoff=grate,positive=True)
        return (lbi.real/lbi.imag-normphase)**2

    res = optimize.fmin(chi2,np.array([0]))

    return lbergIntegrate(res[0],imoff=grate,positive=True)
    

def lindbergIntegral(imoff=0,delta=0.01):
    """
    Calculate the dispersion integral from the Krinsky theory for the quartic potential. The units are all normalised and so the result is very general.
    *imoff*=0 - The normalised growth rate to assume. The default zero value can be used to neglect radiation damping. To include radiation damping, this should be the radiation damping rate normalised by the incoherent synchrotron frequency.
    *delta*=0.01 - The step size in the normalised coherent frequency parameter at which to calculate the dispersion integral. A larger value will of course lead to a faster but coarser solutio.
    """

    ampnorm_p2by3 = np.arange(0,32,0.00006)
    normReOmega = np.arange(-2.5,2.5,delta)
    gamma_quarter = special.gamma(0.25)
    gamma_3quarter = special.gamma(0.75)
    intvar = np.pi**(1/3.)/2**(1/3.)*gamma_3quarter/gamma_quarter*ampnorm_p2by3**(2/3.)
    dimfreq = gamma_quarter/2**(5/4.)/gamma_3quarter**2
    const = 128*np.pi*np.exp(-np.pi)/gamma_quarter/(1+np.exp(-np.pi))**2
    res = np.zeros(len(normReOmega),complex)
    imoff_const_mag = 0.00005

    for i,n in enumerate(normReOmega):
        ksi = dimfreq*n
        #res[i] = 1/(const*(np.trapz(ampnorm_p2by3**(2.5)*np.exp(-ampnorm_p2by3**2)/(ksi**2-ampnorm_p2by3),x=ampnorm_p2by3)-1j*np.pi*ksi**5*np.exp(-ksi**4)))
        if ksi>0:
            imoff_const = imoff_const_mag
        else:
            imoff_const = imoff_const_mag
        res[i] = 1/(const*np.trapz(intvar**2.5*np.exp(-intvar**2)/(ksi**2-intvar+2*1j*ksi*(imoff_const+dimfreq*imoff)-(imoff_const+dimfreq*imoff)**2),x=intvar))
        #func = lambda x: x**2.5*np.exp(-x**2)/(ksi**2-x+2*1j*ksi*(imoff_const+dimfreq*imoff)-(imoff_const+dimfreq*imoff)**2)
        #intgrt = integrate.quadratur(func,0,10)
        #intgrt = integrate.trapz(func(intvar),x=intvar)
        #res[i] = 1/(const*intgrt)
        
    return res

def lindbergIntCoords(normRe,normIm):

    ampnorm_p2by3 = np.arange(0,32,0.00006)
    gamma_quarter = special.gamma(0.25)
    gamma_3quarter = special.gamma(0.75)
    intvar = np.pi**(1/3.)/2**(1/3.)*gamma_3quarter/gamma_quarter*ampnorm_p2by3**(2/3.)
    dimfreq = gamma_quarter/2**(5/4.)/gamma_3quarter**2
    const = 128*np.pi*np.exp(-np.pi)/gamma_quarter/(1+np.exp(-np.pi))**2
    imoff_const = 0.00005
    ksi = dimfreq*normRe
    
    return 1/(const*np.trapz(intvar**2.5*np.exp(-intvar**2)/(ksi**2-intvar+2*1j*ksi*(imoff_const+dimfreq*normIm)-(imoff_const+dimfreq*normIm)**2),x=intvar))



        
        

    

    

    
