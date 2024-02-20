#!/opt/local/bin/python
import vrfCurrScan
import lindberg
import numpy as np

def main():
    print("Testing Brent optimisation and coupling matrix construction>>>>>>>>>>>>>>>>>>>")
    bainst0 = vrfCurrScan.boschAnalysisVrf(1e6,current=np.array([0.2]),detune=np.array([1/0.02**2]),zerofreq=True,deltinsts=True,omegasapprox=True,forceflat=True)
    print(">>>>>>>>>>>Brent optimisation and coupling-matrix construction successful\m\m")
    
    print("Testing Transient class and full calculation of complex coherent frequencies>>>>>>>>>>>>>>>>>>>>>>>>")
    bainst1 = vrfCurrScan.boschAnalysisVrf(1e6,current=np.array([0.2]),detune=np.array([900]))
    print(">>>>>>>>>>>Transient class and full calculation of complex coherent frequencies successful\n\n")
    
    print("Testing Lindberg numerical integral>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    lbg = lindberg.lindbergIntegral(delta=0.1)
    print(">>>>>>>>>>>>>Lindberg integrall successful\n\n")

    print("Testing plotting of Lindberg Integral>>>>>>>>>>>>>>.")
    plottingTest(lbg)

def plottingTest(lbg):

    import paper_plots
    from matplotlib import pyplot

    #f = paper_plots.figure()
    f = pyplot.figure()
    ax = f.add_subplot(111)
    ax.plot(lbg.real,lbg.imag)
    ax.set_xlabel('Normalized coherent frequncy shift')
    ax.set_ylabel('Normalised growth rate')
    print(">>>>>>>>>>>>>>>Testing of plotting of Lindberg Integral successful.")
    
    print("Testing calculation of impedance and plotting>>>>>>>>>>>>>>>>>>>")
    imped = paper_plots.prepImpedanceFigureDebugTest()
    paper_plots.impedanceFigure(imped)
    print(">>>>>>>>>>>>>Testing of impedance calculation and plotting successful")

    pyplot.show()

if __name__=="__main__":
    main()
