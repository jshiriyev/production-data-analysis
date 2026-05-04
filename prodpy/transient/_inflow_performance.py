import numpy

class IPR():

    def __init__(self,**kwargs):
        """oil field units"""
        self.re     = kwargs.get("re")
        self.height = kwargs.get("height")

        self.poro   = kwargs.get("poro")
        self.perm   = kwargs.get("perm")
        
        self.Bo     = kwargs.get("Bo")
        self.muo    = kwargs.get("muo")

        self.ct     = kwargs.get("ct")
        
        self.rw     = kwargs.get("rw")
        self.skin   = kwargs.get("skin")

    def PI(self,regime="pseudo",**kwargs):

        return getattr(self,f"PI_{regime}")(**kwargs)

    def PI_transient(self,time=1):
        """time in days"""

        upper = (self.perm*self.height)

        term = self.perm/(self.poro*self.muo*self.ct*self.rw**2)
        
        lower = 162.6*self.Bo*self.muo*(numpy.log10(term*time*24)-3.23+0.87*self.skin)

        return upper/lower

    def PI_steady(self):

        upper = (self.perm*self.height)

        lower = 141.2*self.Bo*self.muo*(numpy.log(self.re/self.rw)+self.skin)

        return upper/lower

    def PI_pseudo(self):

        upper = (self.perm*self.height)

        lower = 141.2*self.Bo*self.muo*(numpy.log(self.re/self.rw)-0.75+self.skin)

        return upper/lower

    def undersaturated(self,pres,rate=None,pwf=None,regime="pseudo",**kwargs):

        if kwargs.get("PI") is None:
            PI = getattr(self,f"PI_{regime}")(**kwargs)
        else:
            PI = kwargs.get("PI")

        if rate is None:
            return PI*(pres-pwf)
        
        return pres-rate/PI

    def vogel(self,PI,pres,rate=None,pwf=None):

        qmax = PI*pres/1.8

        if rate is None:
            return qmax*(1-0.2*(pwf/pres)-0.8*(pwf/pres)**2)

        values = 81-80*(rate/qmax)
        
        values[values<0] = numpy.nan
        
        return 0.125*pres*(numpy.sqrt(values)-1)

    def fetkovich(self,PI,pres,rate=None,pwf=None,n=None):

        qmax = PI*pres/1.8

        if rate is None:
            return qmax*(1-(pwf/pres)**2)**n

        values = 1-(rate/qmax)**(1/n)

        values[values<0] = numpy.nan

        return pres*numpy.sqrt(values)

    def saturated(self,pres,rate=None,pwf=None,model="vogel",n=None,regime="pseudo",**kwargs):

        if kwargs.get("PI") is None:
            PI = getattr(self,f"PI_{regime}")(**kwargs)
        else:
            PI = kwargs.get("PI")

        if model == "vogel":
            return self.vogel(PI,pres,rate,pwf)
        elif model == "fetkovich":
            return self.fetkovich(PI,pres,rate,pwf,n)
        
    def partial(self,pb,pres,pwf,model="vogel",n=None,regime="pseudo",**kwargs):

        rate = numpy.zeros(pwf.shape)

        if kwargs.get("PI") is None:
            PI = getattr(self,f"PI_{regime}")(**kwargs)
        else:
            PI = kwargs.get("PI")

        rate[pwf>pb] = self.undersaturated(pres,None,pwf[pwf>pb],regime,**kwargs)

        rate[pwf<=pb] = self.undersaturated(pres,None,pb,regime,**kwargs)

        if model == "vogel":
            rate[pwf<=pb] += self.vogel(PI,pb,None,pwf[pwf<=pb])
        elif model == "fetkovich":
            rate[pwf<=pb] += self.fetkovich(PI,pb,None,pwf[pwf<=pb],n)

        rate[pwf<0] = numpy.nan

        return rate

    def PI_vogel(self,pb,pres,rate1:float,pwf1:float):

        if pwf1>pb:
            return rate1/(pres-pwf1)

        dp1 = (pres-pb)+self.vogel(1,pb,None,pwf1)

        return rate1/dp1

    def PI_fetkovich(self,pb,pres,rate1:float,rate2:float,pwf1:float,pwf2:float):

        upper = numpy.log10(rate1/rate2)
        lower = numpy.log10((pres**2-pwf1**2)/(pres**2-pwf2**2))

        n = upper/lower

        dp1 = (pres-pb)+self.fetkovich(1,pb,None,pwf1,n)

        return rate1/dp1,n

import psapy.FluidProps as FluidProps
  
def Darcy_IPR(k,h,visc, re,rw, s, P, OilFVF, nPoints):
    """Function to calculate IPR using Darcy's Equation.  It returns a list with a pair of Pressure and rates"""
    PwfList=[]
    QList=[]
    QList.append(0)
    PwfList.append(P)

    mStep=P/nPoints
    i=1

    while (i<=nPoints):
        
        Pwf=PwfList[i-1]-mStep
        Q= (k*h/visc)*(P-Pwf)/(141.2*OilFVF*visc*(math.log(re/rw)-0.75+s))
        
        QList.append(Q)
        PwfList.append(Pwf)

        i=i+1

    DarcyList=[QList,PwfList]

    return DarcyList

def VogelIPR(P, Pb, Pwf, Qo, nPoints):
    """Function to calculate IPR using Vogel's Equation.  It returns a list with a pair of Pressure and rates"""
    
    PwfList=[]
    QList=[]
    QList.append(0)
    PwfList.append(P)
    VogelList=[]
    mStep=P/nPoints
    i=1

    if Pwf>=Pb:
        J=Qo/(P-Pwf)
       
    else:
        J=Qo/((P-Pb)+((Pb/1.8)*(1-0.2*(Pwf/Pb)-0.8*(Pwf/Pb)**2)))

    while (i<=nPoints):
                     
        Pwfs=PwfList[i-1]-mStep
        
        if Pwfs>=Pb:
           
            Q=J*(P-Pwfs)
        else:
            
            Qb=J*(P-Pb)
            Q=Qb+(J*Pb/1.8)*(1-0.2*(Pwfs/Pb)-0.8*(Pwfs/Pb)**2)
       

        QList.append(Q)
        PwfList.append(Pwfs)

        i=i+1

    VogelList=[QList,PwfList]
    #print(VogelList)
    return VogelList

def Vogel_DarcyIPR(P, k,h,visc, re,rw, s, OilFVF,Temp, Pb, nPoints):
    """Function to calculate IPR using Vogel's Equation.  It returns a list with a pair of Pressure and rates"""
    
    PwfList=[]
    QList=[]
    QList.append(0)
    PwfList.append(P)
    VogelList=[]
    mStep=P/nPoints
    i=1

    
    J= (k*h/visc)/(141.2*OilFVF*visc*(math.log(re/rw)-0.75+s))
              

    while (i<=nPoints):
                     
        Pwfs=PwfList[i-1]-mStep
        print(Pwfs)
        
        if Pwfs>=Pb:
            Q=J*(P-Pwfs)
      
        else:
            
            Qb=J*(P-Pb)
            Q=Qb+(J*Pb/1.8)*(1-0.2*(Pwfs/Pb)-0.8*(Pwfs/Pb)**2)
       

        QList.append(Q)
        PwfList.append(Pwfs)

        i=i+1

    VogelList=[QList,PwfList]
    return VogelList

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    poro = 0.19
    perm = 8.2      # mD
    height = 53     # ft
    Pres = 5651     # psi
    Bo = 1.1
    muo = 1.7       # cp
    ct = 1.29e-5    # 1/psi
    darea = 640     # acres
    rw = 0.328      # ft
    skin = 0

    re = numpy.sqrt((43560*darea)/numpy.pi)

    Pb = 3000    # psi

    inflow = IPR(
        poro = poro,
        perm = perm,
        height = height,
        Bo = Bo,
        muo = muo,
        ct = ct,
        re = re,
        rw = rw,
        skin = skin
        )
    
    qrange = numpy.linspace(0,1200)
    prange = numpy.linspace(0,5651)
    # prange = numpy.array((5651,5000,4500,4000,3500,3000,2500,2000,1500,1000,500,0))
    # prange = numpy.array((0,565,1130,1695,2260,2826,3000,5651))

    p_transt = inflow.undersaturated(Pres,rate=qrange,regime="transient",time=30)
    p_steady = inflow.undersaturated(Pres,rate=qrange,regime="steady")
    p_pseudo = inflow.undersaturated(Pres,rate=qrange,regime="pseudo")

    q_transt = inflow.undersaturated(Pres,pwf=prange,regime="transient",time=30)
    q_steady = inflow.undersaturated(Pres,pwf=prange,regime="steady")
    q_pseudo = inflow.undersaturated(Pres,pwf=prange,regime="pseudo")
    
    # p_vogel = inflow.vogel(Pres,rate=qrange,regime="transient",time=30)
    # p_vogel = inflow.vogel(Pres,rate=qrange,regime="steady")
    p_vogel = inflow.saturated(Pres,rate=qrange,model="vogel")

    # q_vogel = inflow.vogel(Pres,pwf=prange,regime="transient",time=30)
    # q_vogel = inflow.vogel(Pres,pwf=prange,regime="steady")
    q_vogel1 = inflow.saturated(Pres,pwf=prange,model="vogel",regime="pseudo")

    q_fetkovich = inflow.saturated(Pres,pwf=prange,model="fetkovich",regime="pseudo",n=2)

    q_vogel2 = inflow.saturated(Pres,pwf=prange,regime="pseudo")

    # plt.plot(qrange,pwf_transt,label='Transient')
    # plt.plot(qrange,pwf_steady,label='Steady-State')
    # plt.plot(qrange,pwf_pseudo,label='Pseudo-Steady-State')

    # plt.plot(q_pseudo,prange,label='Pseudo')
    plt.plot(q_vogel1,prange,label='Vogel')
    # plt.plot(q_vogel2,prange,label='3000')
    # plt.plot(q_fetkovich,prange,label='Fetkovich')

    plt.legend()

    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

    plt.show()
