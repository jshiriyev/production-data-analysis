class Cruncher:

	@staticmethod
	def N(model,M:float):
		"""
		Volume of oil in place, STB, calculated based on:

		M : the ratio of gas-cap-gas volume to oil volume, bbl/bbl.
		"""
		return (model.reservoir.G*model.phase.Bg)/(M*model.phase.Bo)

	@staticmethod
	def G(model,M:float):
		"""
		Volume of gas-cap-gas, scf, calculated based on:

		M : the ratio of gas-cap-gas volume to oil volume, bbl/bbl.
		"""
		return (model.reservoir.N*model.phase.Bo)*M/model.phase.Bg

	@staticmethod
	def M(model):
		"""Ratio of gas-cap-gas reservoir volume to reservoir oil volume, bbl/bbl"""
		return (model.reservoir.G*model.phase.Bg)/(model.reservoir.N*model.phase.Bo)

	@staticmethod
	def PV(model):
		"""Total pore volume, bbl"""
		return (model.reservoir.N*model.phase.Bo)*(1+Cruncher.M(model))/(1-model.reservoir.Sw)

	@staticmethod
	def Btotal(initial,model):
		"""Two-phase formation volume factor"""
		return model.phase.Bo+(initial.phase.Rs-model.phase.Rs)*model.phase.Bg

	@staticmethod
	def Ntotal(model):
		"""Two-phase total production"""
		return model.operation.Np*(model.phase.Bo+(model.operation.Rp-model.phase.Rs)*model.phase.Bg)

	@staticmethod
	def drive_index(initial,model,which:str="DDI",safe:bool=False):
		"""Calculates the drive index for the mbpy model with respect to initial state.
		
		'which' options are:

		DDI : depletion drive index
		SDI : segregation drive index
		WDI : water drive index
		EDI : expansion drive index

		'safe' defines what to return when drive index can not be calculated due
		to None parameters. If safe is True, it returns 0, None for the vice versa.
		"""
		try:
			return getattr(Cruncher,which)(initial,model)
		except TypeError:
			return 0 if safe else None

	@staticmethod
	def DDI(initial,model):
		"""Calculates depletion drive index."""
		delta_factor = (Cruncher.Btotal(initial,model)-initial.phase.Bo)
		return initial.reservoir.N*delta_factor/Cruncher.Ntotal(model)

	@staticmethod
	def SDI(initial,model):
		"""Calculates segregation (gas-cap) drive index."""
		relative_factor = initial.phase.Bo*(model.phase.Bg-initial.phase.Bg)/initial.phase.Bg
		return Cruncher.M(initial)*initial.reservoir.N/Cruncher.Ntotal(model)*relative_factor

	@staticmethod
	def WDI(initial,model):
		"""Water drive index"""
		return (model.reservoir.We-model.operation.Wp*model.phase.Bw)/Cruncher.Ntotal(model)

	@staticmethod
	def EDI(initial,model):
		"""Expansion (rock and liquid) depletion drive"""
		ctotal = (model.phase.cf+model.phase.cw*initial.reservoir.Sw)
		return Cruncher.PV(initial)*ctotal*(initial.reservoir.P-model.reservoir.P)/Cruncher.Ntotal(model)

	@staticmethod
	def total_drive_index(initial,model):
		"""Returns the total drive index for the mbpy model with respect to initial state."""
		ddi = Cruncher.drive_index(initial,model,"DDI",safe=True)
		sdi = Cruncher.drive_index(initial,model,"SDI",safe=True)
		wdi = Cruncher.drive_index(initial,model,"WDI",safe=True)
		edi = Cruncher.drive_index(initial,model,"EDI",safe=True)
		return ddi+sdi+wdi+edi
