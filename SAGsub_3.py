import os
import numpy as np
from scipy import fftpack, fft, ndimage, signal
import matplotlib.pyplot as plt
from MRC_RW_3 import *
from GridTools import *
from Pattern_Optimizer import *

if __name__ == '__main__':

	######### set up graphical interface ##########

	init = np.random.randint(9, size=(1000,1000))
	#PSinit = np.log10(init)**2
	plt.ion()
	fig, ax = pl.subplots(1,2)
	micrograph = ax[0]
	spectrum = ax[1]
	#micrograph_sub = ax[1,0]
	#spectrum_sub = ax[1,1]
	#micrograph.imshow(init, cmap=pl.cm.Greys)
	#spectrum.imshow(PSinit, cmap=pl.cm.Greys)
	#micrograph_sub.imshow(init, cmap=pl.cm.Greys)
	#spectrum_sub.imshow(PSinit, cmap=pl.cm.Greys)
	#
	#plt.pause(0.5)

	###############################################

	#Glacios 45k
	#apix = 0.901
	#Planes = 66
	#Profile_threshold = 0.25 # minimum ratio of signal to consider for peak profiling
	#Sigma_threshold = 1.5
	#Patch_Size = 20 # for global peak search and peak operations
	#Patch_Fine = 5  # for parameters refinement only
	#Pad = 200
	#Resolution_limit = 10
	
	#Glacios 73k
	#apix = 0.562
	#Planes = 41
	#Profile_threshold = 0.25 # minimum ratio of signal to consider for peak profiling
	#Sigma_threshold = 1.5
	#Patch_Size = 20 # for global peak search and peak operations
	#Patch_Fine = 5  # for parameters refinement only
	#Pad = 200
	#Resolution_limit = 10
	
	#Krios2 165kk
	apix = 0.729
	Planes = 56.6
	Profile_threshold = 0.1 # minimum ratio of signal to consider for peak profiling
	Sigma_threshold = 1.5
	Patch_Size = 11 # for global peak search and peak operations
	Patch_Size_Operation = 21 # not used yet !
	height = 0.3 # peak height (ratio of 1) for mosaicit search
	width = 3 # peak width (in step) for mosaicit search
	Patch_Fine = 3  # for parameters refinement only
	Pad = 200
	Resolution_limit = 10

	AllFiles = os.listdir(os.getcwd())
	MRCFiles = [element for element in AllFiles if element[-4:] == '.mrc']

	mrc = MRC()

	log = open('log.txt','w')

	for mic in MRCFiles:

		mrc.read(mic)

		box = np.array(mrc.data.shape)					# Get micrograph dimensions
		Squarizer = max(box)-box						# Calculate pixels to add to squarize micrograph
		box_padded = box+Squarizer+2*Pad				# Determine dimensions of the squarized, padded micrograph
		Pads = (box_padded-box)//2
		Center = box_padded//2	; #print(Center)						# Coordinates of the resized micrograp center
		Center = Center+0.5		; #print(Center, box_padded) 						#to center on the real PS center.
		
		x = np.arange(0,box_padded[0],1)
		y = np.arange(0,box_padded[1],1)

		Padder = np.ones(box_padded)*np.mean(mrc.data)				# Padding array
		Padder[Pads[0]:box[0]+Pads[0],Pads[1]:box[1]+Pads[1]] = mrc.data ; #print(Padder.shape)# Padding the micrograph
		
		#mrc.write(Padder, 'padded.mrc')

		micrograph.imshow(1-mrc.data, cmap=pl.cm.Greys)

		Fourier = fftpack.fftshift(fftpack.fft2(Padder)) ;			# Compute fourier transform of the padded micrograph
		PS = np.abs(Fourier)							# Compute Power Spectrum from fourier transform
		
		spectrum.imshow(1-np.log10(PS)**2, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])


		### Building Refinement Pattern ###

		Pattern = MakePattern(box_padded, Planes)+0.5

		# define a resolution-limited pattern for lattice refinement #

		DistX = Pattern[:,0]-Center[0]
		DistY = Pattern[:,1]-Center[1]
		Dist2D = np.sqrt(DistX**2+DistY**2)
		Pattern_ResMatrix = apix/Dist2D*np.max(box_padded)
		
		Resolution_distance_test = (np.max(box_padded)/53)/(2*apix) ; print("proposed resolution:", Resolution_distance_test)

		LowRes_Pattern = Flank_Pattern_Resolution(Pattern, Pattern_ResMatrix, 100, 40) ; #print(LowRes_Pattern.shape)

		### Rotate matrix for coarse peak search ###
		
		Search_Range = 90
		Search_Step = 1

		Angles, Intensities = Rotation_Search(LowRes_Pattern, PS, Center, SearchStart = 0, Search_Range = Search_Range, step = Search_Step, Patch_Size = Patch_Size)
		
		######################################################################
		
		#for i in range(len(Angles)): log.write(str(Angles[i])+'\t'+str(Intensities[i])+'\n')

		BestAngle = Angles[np.argmax(Intensities)+1] ; #print(BestAngle)
		
		AngleLoc = signal.find_peaks(Intensities, height = height, width = width)
		MainLattices = AngleLoc[0] ; print(MainLattices)
		Mosaicity = len(MainLattices) ; #print(Mosaicity)

		Pattern_Rot = Rotate(LowRes_Pattern, Center, BestAngle)

		#############  Determine XY shifts to apply to recenter on spots  ###############

		patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
			spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]

		Shifts = []

		for spot in patches:

			bright = np.array(np.where(spot==spot.max())) ; #print(bright)

			Shifts.append([bright[0][0]-Patch_Size//2,bright[1][0]-Patch_Size//2])

		Shifts = np.array(Shifts) ; #print("distance error:", np.round(np.sqrt(Shifts[:,0]**2+Shifts[:,1]**2),2))

		Pattern_Rot = Pattern_Rot+Shifts

		################## For Display Only ############################################

		patches_corrected = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
			spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]

		Nspot = len(patches)
		Columns = 4
		Rows = Nspot // Columns

		if Nspot % Columns !=0: Rows +=1

		Position = range(1, Nspot + 1)

		fig2 = plt.figure(2)

		for k in range(Nspot):
			ax = fig2.add_subplot(Rows,Columns,Position[k])
			ax.imshow(1-np.log10(patches[k])**2, cmap=pl.cm.Greys)

		plt.pause(3)

		for k in range(Nspot):
			ax = fig2.add_subplot(Rows,Columns,Position[k])
			ax.imshow(1-np.log10(patches_corrected[k])**2, cmap=pl.cm.Greys)

		plt.show()

		################################################################################

		#############################  Refining Planes spacing  ######################################################

		DistXY = np.array([Pattern_Rot-Pattern_Rot[i] for i in range(len(Pattern_Rot))]) ; #print(DistXY)
		Dist2D = np.array([np.sqrt(DistXY[i][:,0]**2+DistXY[i][:,1]**2) for i in range(len(DistXY))]) ; #print(Dist2D)
		min_Planes = [np.min(Dist2D[i][np.nonzero(Dist2D[i])]) for i in range(len(Dist2D))] ; #print(min_Planes)
		Optimized_Planes = np.mean(min_Planes) ; print('Optimized planes distance:', Optimized_Planes)
		
		Pattern_Opt = MakePattern(box_padded, Optimized_Planes)+0.5 ; #print(Pattern_Opt)
		
		Pattern_Opt = Flank_Pattern_Resolution(Pattern_Opt, Pattern_ResMatrix, 100, 40) ; #print(Pattern_Opt)
		
		if Mosaicity > 1:
		
			Pattern_Pool = []

			Patter_Pool_Final = []
		
			for i in range(Mosaicity):
		
					Pattern_Opt_oriented = Rotate(Pattern_Opt, Center, MainLattices[i]-10)
					
					Angles_Opt, Intensities_Opt = Rotation_Search(Pattern_Opt_oriented, PS, Center, SearchStart = 0, Search_Range = 11, step = 0.25, Patch_Size = Patch_Fine)
					
					BestAngle_Opt = Angles_Opt[np.argmax(Intensities_Opt)]+(MainLattices[i]-10) ; #print(BestAngle_Opt)
					
					Pattern_Final = MakePattern(box_padded, Optimized_Planes)+0.5

					Pattern_Final = Rotate(Pattern_Final, Center, BestAngle_Opt)
					
					Pattern_Pool.append(Pattern_Final)

			#print(len(Pattern_Pool))
					
			for lattice in Pattern_Pool:

				for spot in lattice:

					Patter_Pool_Final.append(spot)
				
			Pattern_Final = np.array(Patter_Pool_Final) ; #print(Pattern_Final)
		
		
		if Mosaicity == 1:
		
			Pattern_Opt_oriented = Rotate(Pattern_Opt, Center, BestAngle-10)

			Angles_Opt, Intensities_Opt = Rotation_Search(Pattern_Opt_oriented, PS, Center, SearchStart = 0, Search_Range = 11, step = 0.25, Patch_Size = Patch_Fine)
		
			#for i in range(len(Angles_Opt)): log.write(str(Angles_Opt[i]).replace('.',',')+'\t'+str(Intensities_Opt[i]).replace('.',',')+'\n')

			BestAngle_Opt = Angles_Opt[np.argmax(Intensities_Opt)]+(BestAngle-10)

			#Pattern_Rot_Opt = Rotate(Pattern_Opt, Center, BestAngle_Opt)

			Pattern_Final = MakePattern(box_padded, Optimized_Planes)+0.5

			Pattern_Final = Rotate(Pattern_Final, Center, BestAngle_Opt)


		#for i in range(len(Pattern_Final)): log.write((str(Pattern_Final[i][0])+'\t'+(str(Pattern_Final[i][1])+'\n')))

		############################# Apply optimized parameters to a large pattern

		DistX = Pattern_Final[:,0]-Center[0]
		DistY = Pattern_Final[:,1]-Center[1]
		Dist2D = np.sqrt(DistX**2+DistY**2)
		Pattern_ResMatrix = apix/Dist2D*np.max(box_padded)

		Pattern_Final_HR = Flank_Pattern_Resolution(Pattern_Final, Pattern_ResMatrix, 100, Resolution_limit)
		
		for i in range(len(Pattern_Final_HR)): log.write((str(Pattern_Final_HR[i][0])+'\t'+(str(Pattern_Final_HR[i][1])+'\n')))

		#############  Determine XY shifts to apply to recenter on spots  ###############

		patches_Opt = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
			spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Final_HR]

		Shifts = []

		for spot in patches_Opt:

			bright = np.array(np.where(spot==spot.max()))

			Shifts.append([bright[0][0]-Patch_Size//2,bright[1][0]-Patch_Size//2])

		Shifts = np.array(Shifts)

		Pattern_Final_HR = Pattern_Final_HR+Shifts

		patches_Opt = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
			spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Final_HR]

		############################ Building Peak Mask  ################################

		Summed_Peak = np.log10(np.sum(np.array(patches_Opt), axis=0)**2)

		Normalized_Peak = (Summed_Peak-np.min(Summed_Peak))/(np.max(Summed_Peak)-np.min(Summed_Peak))

		Peak_Mask = Normalized_Peak >= Profile_threshold

		Bg_Mask = Normalized_Peak < Profile_threshold

		#spectrum.imshow(Bg_Mask, cmap=pl.cm.Greys)

		################## For Display Only ############################################

		#Nspot = len(patches_Opt)
		#Columns = 4
		#Rows = Nspot // Columns

		#if Nspot % Columns !=0: Rows +=1

		#Position = range(1, Nspot + 1)

		#fig2 = plt.figure(2)

		#for k in range(Nspot):
		#	ax = fig2.add_subplot(Rows,Columns,Position[k])
		#	ax.imshow(1-np.log10(patches_Opt[k])**2, cmap=pl.cm.Greys)

		####################### Iterate through spots ################################

		Spot_MetaData = []

		for s in range(len(patches_Opt)):

			spot = patches_Opt[s]

			coordinate = Pattern_Final_HR[s]

			#resolution = 

			DistXY = coordinate-Center
			Dist2D = np.sqrt(DistXY[0]**2+DistXY[1]**2)
			Spot_Resolution = round(apix/Dist2D*np.max(box_padded),3)

			Peak_value = np.mean(spot[Peak_Mask])
			Bg_value = np.mean(spot[Bg_Mask])

			Spot_MetaData.append([Peak_value/Bg_value, Spot_Resolution])

		Spot_MetaData = np.array(Spot_MetaData) ; #print(Spot_MetaData)

		fig3 = plt.figure(3)
		ax3 = fig3.add_subplot(1,1,1)
		ax3.invert_xaxis()
		ax3.scatter(Spot_MetaData[:,1], Spot_MetaData[:,0], c ="blue")
		#fig3.plot()

		################## For Display Only ############################################

#		Nspot = len(patches_Opt)
#		Columns = 4
#		Rows = Nspot // Columns
#
#		if Nspot % Columns !=0: Rows +=1
#
#		Position = range(1, Nspot + 1)
#
#		fig2 = plt.figure(2)
#
#		for k in range(Nspot):
#
#			sigma, resolution = Spot_MetaData[k][0], Spot_MetaData[k][1] ; #print(sigma, resolution)
#
#			if sigma >= Sigma_threshold:
#
#				ax = fig2.add_subplot(Rows,Columns,Position[k])
#				ax.imshow(1-np.log10(patches_Opt[k])**2, cmap=pl.cm.Greys)
#
#		spectrum.scatter(Pattern_Final_HR[:,0], Pattern_Final_HR[:,1], marker = "s", edgecolor = "red", s = 50, facecolors='none')

                ############################################# Building finalized pattern ##################################################
                
		Pattern_Final_HR_cleaned = []
		
		for k in range(len(Pattern_Final_HR)):

			spot_coord = Pattern_Final_HR[k]
			spot_sigma = Spot_MetaData[k][0]
			spot_res   = Spot_MetaData[k][1]

			#print(spot_coord, spot_res, spot_sigma)

			if spot_sigma >= Sigma_threshold: Pattern_Final_HR_cleaned.append(spot_coord)

		Pattern_Final_HR_cleaned = np.array(Pattern_Final_HR_cleaned)

		#print(Bg_Mask.shape)

		#spectrum.scatter(Pattern_Final_HR_cleaned[:,0], Pattern_Final_HR_cleaned[:,1], marker = "s", edgecolor = "red", s = 50, facecolors='none')

		############################################ SUBTRACTING SIGNAL ##########################################################

		PS_sub = np.copy(Fourier)

		for spot in Pattern_Final_HR_cleaned:
			PS_sub[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] = PS_sub[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1]*Bg_Mask

		
		#spectrum.imshow(1-np.log10(np.abs(PS_sub[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1]))**2, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])

		data_sub = np.real(fftpack.ifft2(fftpack.fftshift(PS_sub)))

		data_sub = data_sub[Pads[0]:box[0]+Pads[0],Pads[1]:box[1]+Pads[1]] #unpadding

		spectrum.imshow(1-data_sub, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])

		


	#mrc.write(1-np.log10(PS)**2,'PS.mrc')
	mrc.write(data_sub, mic)

	plt.pause(10)
	log.close()


