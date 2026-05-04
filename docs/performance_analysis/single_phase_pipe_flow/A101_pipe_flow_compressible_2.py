""" Class Exercise 2 """

Pg = 101325         # pressure in Pa
Tg = 288            # Kelvin
Mg = 0.016          # mole mass in kg/mole

Qg = 50             # m3/second

Rc = 8.314          # universal gas constant in SI units

mu = 0.01e-3        # Pascal-second

P2 = 170e3          # Pascal

L = 3*1000          # meters
D = 0.6             # meters

Tamb = 293          # ambient temperature, Kelvin

rhog = (Pg*Mg)/(Rc*Tg)

G = Qg*rhog         # kilogram per seconds

A = np.pi*D**2/4

Re = (G*D)/(mu*A)

phi = optimize.newton(objective1,8/Re,args=(Re,))

P1 = optimize.minimize(objective2,P2,
                       args=(P2,L,D,phi,Mg,Tamb,G),
                       bounds=[(P2,None)],
                       method='Powell').x[0]

wc = optimize.minimize_scalar(objective3,args=(L,D,phi),bounds=(1e-5,1),method='bounded').x
