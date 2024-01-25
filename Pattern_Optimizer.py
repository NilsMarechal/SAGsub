import numpy as np
from scipy import signal
from GridTools import *

def Rotation_Search(Pattern, PS, Center, SearchStart = 0, Search_Range = 180, step = 1, Patch_Size = 20):


	Intensities = []
	Angles = []
	
	#print(range(SearchStart, int((SearchStart+Search_Range)/step), 1))


	for i in range(SearchStart, int((SearchStart+Search_Range)/step), 1):
				
		Pattern_Rot = Rotate(Pattern, Center, i*step)
				
		patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
			spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]
				
		Intensity = np.mean(patches)
		Intensities.append(Intensity)
		Angles.append(i*step)

	Intensities = np.array(Intensities)

	Angles = np.array(Angles)

	Intensities = (Intensities-np.min(Intensities))/(np.max(Intensities)-np.min(Intensities))
			
	return Angles, Intensities


def Limit_Pattern_Resolution(Pattern, Resolution_Matrix, Resolution_limit):

	Pattern_New = []
	for i in range(len(Resolution_Matrix)):
				res = Resolution_Matrix[i]
				if res >= Resolution_limit:Pattern_New.append(Pattern[i])

	return np.array(Pattern_New)

def Flank_Pattern_Resolution(Pattern, Resolution_Matrix, Upper_limit, Lower_limit):

	Pattern_New = []
	for i in range(len(Resolution_Matrix)):
				res = Resolution_Matrix[i]
				if Upper_limit >= res >= Lower_limit :Pattern_New.append(Pattern[i])

	return np.array(Pattern_New)


	
