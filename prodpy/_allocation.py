import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from lasio import LASFile

class Allocation():

	def __init__(self,lasfile=None):

		self.lasfile = lasfile

	@staticmethod
	def production(prod:float,tops:tuple,perf:tuple):
		"""
		prod 	: floating value to be allocated along formations
		tops 	: tuple of (formation tops,), top to deeper
		perf 	: tuple of (perforation top, perforation bottom), top to deeper

		output	: tuple of production allocated along formations,
				  the length of tuple will be len(tops)-1
		"""

		heights = Allocation.heights(tops,perf)

		perftop,perflow = perf

		pdepth = perflow-perftop

		shares = [prod*height/pdepth for height in heights]

		return tuple(shares)

	@staticmethod
	def heights(tops:tuple,perf:tuple,**kwargs):
		"""
		tops 	: tuple of (formation tops,), top to deeper
		perf 	: tuple of (perforation top, perforation bottom), top to deeper

		output 	: tuple of thicknesses distributed along perforation interval,
				  the length of tuple will be len(tops)-1

		"""

		shares = []

		perftop,perflow = perf

		for index in range(len(tops)-1):

			formtop,formlow = tops[index:index+2]

			top = perftop if formtop<perftop else formtop
			low = formlow if formlow<perflow else perflow

			if low<top:
				interval = 0
			elif len(kwargs)>0:
				interval = Allocation.nets((top,low),**kwargs)
			else:
				interval = low-top

			shares.append(interval)
		
		return shares

	@staticmethod
	def nets(zone:tuple,lithid:int=1,lasfile:"LASFile"=None,key:str="LID"):
		"""
		The method requires the use of las file where the lithology identifier curve is available.
		It creates heights for each las depth node and returns summed heights with specified lithology identifier.

		zone 	: tuple of (top,bottom), positive values and top<bottom
		lithid 	: lithology identifier to use for thickness calculations
		lasfile : lasfile containing the lithology identifier curve
		key 	: lasfile key for lithology identifier curve

		output 	: thickness of lithology type in the zone
		"""

		top,bottom = zone

		depth = lasfile[0]

		bools = np.logical_and(depth>=top,depth<=bottom)

		depth = depth[bools]
		# print("depths selected",depth)

		curve = lasfile[key][bools]

		if depth.size==1:
			return np.sum(curve==lithid)*(bottom-top)

		heights = (depth[2:]-depth[:-2])/2
		# print("heigths before correction",heights)

		if depth[0]-top<depth[1]-depth[0]:
			heightI = (depth[1]+depth[0]-2*top)/2
		else:
			heightI = depth[1]-depth[0]

		if bottom-depth[-1]<depth[-1]-depth[-2]:
			heightL = (2*bottom-depth[-1]-depth[-2])/2
		else:
			heightL = depth[-1]-depth[-2]

		heights = np.insert(heights,0,heightI)
		heights = np.insert(heights,heights.size,heightL)
		# print("heigths after correction",heights)

		return np.sum(heights[curve==lithid])

if __name__ == "__main__":

	import matplotlib.pyplot as plt

	class lasfile:

		def __init__(self,depth,lithid,lithkey="lid"):

			self.depth = depth
			self.lithid = lithid
			self.lithkey = lithkey

		def __getitem__(self,key):

			if key==0:
				return self.depth
			elif key==1:
				return self.lithid
			elif key==self.lithkey:
				return self.lithid

	N = 6

	dtop,dlow = 5,20

	zone = (0.,17.5)

	D = np.linspace(dtop,dlow,N)
	L = np.random.randint(0,2,N)

	plt.scatter(L,D)
	plt.hlines(zone,0,1)
	plt.ylim((max(dlow,zone[1])+(dlow-dtop)/10,min(dtop,zone[0])-(dlow-dtop)/10))

	file = lasfile(D,L)

	H = Allocation.nets(zone,1,file,"lid")

	print(H)

	plt.show()
