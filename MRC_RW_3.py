import mrcfile as mrc
import numpy as np
from scipy import fftpack
import pylab as pl

class MRC:

	def __init__(self):

		self.data = np.array

	def read(self, FileName):

		FileName = FileName

		micrograph = mrc.open(FileName)

		self.data = micrograph.data

		self.header = micrograph.header

	def show(self):

		pl.figure(1)
		pl.clf()
		pl.imshow(self.data, cmap=pl.cm.Greys)
		pl.show()
		
	def write(self, Data, FileName):
		
		self.data = Data
		
		with mrc.new(FileName) as new:
			new.set_data(np.float32(self.data))
