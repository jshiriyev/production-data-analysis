class WaterInfluxModel():

    def __init__(self, Pi, cw, cf):
        
        self.Pi = Pi
        self.cw = cw
        self.cf = cf

    def potmodel(self,P,K):

        return (self.cw+self.cf)*K*(self.Pi-P)

	# # Van Everdingen-Hurst water influx model (simplified)
	# def water_influx(t, k, theta):
	#     # k: aquifer influx coefficient (fit parameter)
	#     # theta: aquifer size ratio (assumed)
	#     # t: time
	#     # W_e = k * sqrt(t) * theta
	#     return k * np.sqrt(t) * theta

if __name__ == "__main__":

	influx = WaterInflux(3000,4e-6,3e-6)
	influx.potmodel(2800,156.5e6*80/360)