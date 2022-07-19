import numpy as np
import matplotlib.pyplot as plt

def Ang2Pix():

	pass

def Rotate(array, origin=(0,0), degrees=0):

	Array = np.copy(array)

	angle = np.deg2rad(degrees)

	RotMatrix = np.array([[np.cos(angle), -np.sin(angle)],
			      [np.sin(angle),  np.cos(angle)]])

	origin = np.atleast_2d(origin)
	Array = np.atleast_2d(Array)

	return np.rint(np.squeeze((RotMatrix @ (Array.T-origin.T)+origin.T).T)).astype(int)

def Rotate_and_Stigmate(array, a=1, b=1 ,origin=(0,0), degrees=0):

	angle = np.deg2rad(degrees)

	RotMatrix = np.array([[a*np.cos(angle), -b*np.sin(angle)],
			      [a*np.sin(angle),  b*np.cos(angle)]])

	print(RotMatrix)

	origin = np.atleast_2d(origin)
	array = np.atleast_2d(array)

	return np.rint(np.squeeze((RotMatrix @ (array.T-origin.T)+origin.T).T)).astype(int)

def StigmateX(array, originX=0, step=1):

	Array = np.copy(array)

	########## doing X-shift to 0 ##########

	Array[:,0] = Array[:,0]-originX

	########################################

	Array[:,0] = Array[:,0]*step+originX

	return Array

def StigmateY(array, originY=0, step=1):

	Array = np.copy(array)

	########## doing Y-shift to 0 ##########

	Array[:,1] = Array[:,1]-originY

	########################################

	Array[:,1] = Array[:,1]*step+originY

	return Array

def StigmateXY(array, origin=[0,0], step=(1,1)):

	Array = np.copy(array)

	########## doing XY-shift to 0 ##########

	Array[:,0] = Array[:,0]-origin[0]
	Array[:,1] = Array[:,1]-origin[1]

	########################################

	Array[:,0] = Array[:,0]*step[0]+origin[0]
	Array[:,1] = Array[:,1]*step[1]+origin[1]

	return Array

def SquareGrid(box, plane_spacing):

	center = box//2

	#points_halfbox = np.around(box-center/plane_spacing,-1) ; #print(points_halfbox)

	Pattern_half1 = []
	Pattern_half2 = []
	Pattern_half3 = []
	Pattern_half4 = []

	for i in range(center[0], box[0], plane_spacing):
		for j in range(center[1],box[1], plane_spacing):
			Pattern_half1.append([i,j])

	for i in range(center[0]-plane_spacing, -1, -plane_spacing):
		for j in range(center[1]-plane_spacing, -1, -plane_spacing):
			Pattern_half2.append([i,j])

	for i in range(center[0], box[0], plane_spacing):
		for j in range(center[1]-plane_spacing, -1, -plane_spacing):
			Pattern_half3.append([i,j])

	for i in range(center[0]-plane_spacing, -1, -plane_spacing):
		for j in range(center[1],box[1], plane_spacing):
			Pattern_half4.append([i,j])

	Pattern = Pattern_half2 + Pattern_half1 + Pattern_half3 + Pattern_half4

	return np.array(Pattern).astype(int)
	
def MakePattern(box, plane_spacing):

	center = box//2

	points_box = np.around(box/plane_spacing,-1).astype(np.int)
	
	if points_box[0]%2 == 0: points_box[0]=points_box[0]+1
	if points_box[1]%2 == 0: points_box[1]=points_box[1]+1

	Pattern = []
	
	for i in range(0, points_box[0], 1):
		for j in range(0, points_box[1], 1):
			Pattern.append([i*plane_spacing,j*plane_spacing])
			
	Pattern = np.array(Pattern)
	
	CoM = np.mean(Pattern)
	Diff = center - CoM ; #print(Diff)
	
	return Pattern+Diff#-[0,1]
			


#box = np.array([3200,3200])
#box = np.array([300,300])

#out = SquareGrid(box, 55)

#out = Rotate(out, box//2, degrees=56)
#backup = SquareGrid(box, 55)
#backup = Rotate(backup, box//2, degrees=56)
#out = Rotate_and_Stigmate(out, a=1, b=2, origin=box//2, degrees=56)

#out = StigmateX(out, (box//2)[0], 1.1)
#out = StigmateY(out, (box//2)[1], 1.1)

#plt.scatter(out[:,0],out[:,1])
#plt.scatter(backup[:,0],backup[:,1])
#plt.show()
