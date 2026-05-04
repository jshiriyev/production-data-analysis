import numpy

def wichert_aziz(ppc,tpc,h2s=0.0,co2=0.0):
    """It corrects pseudo-critical temperature and pressure based on
    the non-hydrocarbon mole fraction."""

    A,B = h2s+co2,h2s
    
    epsilon = 120*(A**0.9-A**1.6)+15*(B**0.5-B**4.0)

    tpc_corr = tpc-epsilon
    ppc_corr = ppc*tpc_corr/(tpc+B*(1-B)*epsilon)

    return ppc_corr,tpc_corr

def carr_kobayashi_burrows(tpc,ppc,h2s=0.0,co2=0.0,n2=0.0):
    """It corrects pseudo-critical temperature and pressure based on
    the non-hydrocarbon mole fraction."""

    ppc_corr = ppc+440*co2+600*h2s-170*n2
    tpc_corr = tpc-80*co2+130*h2s-250*n2

    return ppc_corr,tpc_corr

# High Molecular Weigth Correction

def sutton(mfracs,pcrits,tcrits,c7plus=-1):

    term1,term2,term3 = [],[],[]

    for yi,Pci,Tci in zip(mfracs,pcrits,tcrits):

        term1.append(yi*(Tci/Pci))
        term2.append(yi*(Tci/Pci)**0.5)
        term3.append(yi*(Tci/(Pci)**0.5))

    J = 1/3*sum(term1)+2/3*(sum(term2))**2
    K = sum(term3)

    yc7 = mfracs[c7plus]
    
    t1c7 = term1[c7plus]
    t2c7 = term2[c7plus]
    t3c7 = term3[c7plus]

    Fj = 1/3*t1c7+2/3*t2c7**2
    Ej = 0.6081*Fj+1.1325*Fj**2-14.004*Fj*yc7+64.434*Fj*yc7**2
    Ek = t3c7/yc7*(0.3129*yc7-4.8156*yc7**2+27.3751*yc7**3)

    J_corr,K_corr = J-Ej,K-Ek

    tpc_corr = K_corr**2/J_corr
    ppc_corr = tpc_corr/J_corr

    return ppc_corr,tpc_corr