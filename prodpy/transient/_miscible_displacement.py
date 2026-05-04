class MiscDispment():

    def obj(XX,v,L):

        data = numpy.loadtxt('Lecture_08_dispersion.txt',skiprows=1)

        time = data[:,0]*3600   # seconds

        cc0_c = concentration(XX,v,L,time)

        cc0 = data[:,1]         # dimensionless

        return numpy.sum((cc0-cc0_c)**2)

    def concentration(XX,v,L,time):

        DL = 10**(-XX)

        term1 = erfc((L-v*time)/(2*numpy.sqrt(DL*time)))
        term2 = erfc((L+v*time)/(2*numpy.sqrt(DL*time)))

        if not numpy.any(term2):
            return 1/2*(term1)
        else:
            term3 = numpy.exp(v*L/DL)
            return 1/2*(term1+term2*term3)