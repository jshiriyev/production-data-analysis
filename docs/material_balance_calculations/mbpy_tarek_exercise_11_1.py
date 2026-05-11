import mbpy

tank = mbpy.Tank(
	M = 0.25,
	P = 3000,
	N = 10_000_000,
	Sw = 0.2,
	Bo = 1.58,
	Bw = 1.0,
	Bg = 0.0008,
	Rs = 1040,
	cw = 1.5e-6,
	cf = 1e-6,
	)

print(tank.initial.M)
print(tank.initial.N)
print(tank.initial.G)

tank = tank(P=2800,Bo=1.48,Rs=850,Bg=0.00092,
	Np=1_000_000,Gp=1_100_000_000,Wp=50_000)

print(tank.initial.operation.Rp)
print(tank.current.operation.Rp)

print()
print(tank.drive_index("DDI"))
print(tank.drive_index("SDI"))
print(tank.drive_index("WDI"))
print(tank.drive_index("EDI"))
print()
print(tank.total_drive_index())
print()

sol = tank.minimize(We=100,optimizer=dict(method="Powell"))

if sol.success:
	tank.current.reservoir.We = sol.x[0]

print(tank.drive_index("DDI"))
print(tank.drive_index("SDI"))
print(tank.drive_index("WDI"))
print(tank.drive_index("EDI"))
print()
print(tank.total_drive_index())
print()

