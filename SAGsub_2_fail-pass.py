import os
import numpy as np
from scipy import fftpack, fft, ndimage, signal
#import matplotlib.pyplot as plt
from MRC_RW_3 import *
from GridTools import *
from Pattern_Optimizer import *
import shutil as sh

#fig, ax = pl.subplots(2,2)
#micrograph = ax[0,0]
#spectrum = ax[0,1]
#micrograph_sub = ax[1,0]
#spectrum_sub = ax[1,1]

AllFiles = os.listdir(os.getcwd())

MRCFiles = []
    
for element in AllFiles:

    if element[-4:] == '.mrc' and 'sub' not in element: MRCFiles.append(element)


for mic in MRCFiles:

    #if 'FILTERED_'+mic not in AllFiles:
    if mic[:-4]+'_sub.mrc' not in AllFiles:

        print('############################### processing ###############################\n',mic,'\n##########################################################################')

        # Krios 130k
        #apix = 0.526
        #Planes = 52.5
        #Patch_Size = 20
        #Pix_Per_Peak = 17
        #PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
        #Resolution_limit = 20

        # Krios 81k
        apix = 0.862
        Planes = 86.5
        Patch_Size = 20
        Pix_Per_Peak = 27
        PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
        Resolution_limit = 20

        # Glacios 45k
        #apix = 0.901
        #Planes = 62.2
        #Patch_Size = 20
        #Pix_Per_Peak = 17
        #PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
        #Resolution_limit = 20

        # Glacios 45k GOLD
        #apix = 0.901
        #Planes = 62.2
        #Patch_Size = 30
        #Pix_Per_Peak = 30
        #PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
        #Resolution_limit = 20
        
        # Glacios 73k
        #apix = 0.562
        #Planes = 39.0
        #Patch_Size = 20
        #Pix_Per_Peak = 17
        #PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
        #Resolution_limit = 10

        # F20 29k    
#        apix = 3.6
#        Planes = 125
#        Pix_Per_Peak = 17
#        PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
#        Patch_Size = 20

        # F20 50k    
#        apix = 2.12
#        Planes = 73.25
#        Pix_Per_Peak = 31
#        Patch_Size = 20
#        PERCENT = 100-Pix_Per_Peak/((Patch_Size+1)**2)*100
#        Resolution_limit = 20

        try:
            mrc = MRC()
            
            mrc.read(mic)

            box = np.array(mrc.data.shape)
            Max = np.max(box)
            Center = box//2

            ### Pad the Image array to a square to prevent PS distortion ###

            Square = np.ones([Max, Max])*np.mean(mrc.data)
            Square_box = np.array(Square.shape)
            Square_center = Square_box//2

            Pad_Indexes = (Square_box-box)//2
            Square[Pad_Indexes[0]:Square_box[0]-Pad_Indexes[0],Pad_Indexes[1]:Square_box[1]-Pad_Indexes[1]] = mrc.data

            Fourier = fftpack.fftshift(fftpack.fft2(Square))
            PS = np.abs(Fourier)

            ### Building Refinement Pattern ###

            Pattern = MakePattern(Square_box, Planes)

            # define a resolution-limited pattern for lattice refinement #

            DistX = Pattern[:,0]-Square_center[0]
            DistY = Pattern[:,1]-Square_center[1]
            Dist2D = np.sqrt(DistX**2+DistY**2)
            Pattern_ResMatrix = apix/Dist2D*np.max(Square_box)

            # Correct for Pattern Center (not need for now)
            Pattern_Center_Index = Dist2D.tolist().index(min(Dist2D))
            Pattern_Center_Coord = Pattern[Pattern_Center_Index].astype(int)
            Central_Spot = PS[Pattern_Center_Coord[0]-Patch_Size//2:Pattern_Center_Coord[0]+Patch_Size//2+1,Pattern_Center_Coord[1]-Patch_Size//2:Pattern_Center_Coord[1]+Patch_Size//2+1]
                        
            Ref_Pattern = Limit_Pattern_Resolution(Pattern, Pattern_ResMatrix, Resolution_limit)

            ### Refining Pattern ###

            Search_Start_Rot = 0 ; Search_Range_Rot = 180 ; Patch_Size_Rot = 20 ; Step_Rot = 0.05 ; Rot_Total = 0
            Search_Offset_Plane = [-2,2] ; Search_Step_Plane = 0.1 #0.05

            angle, intensity = Solve_Orientation_New(Ref_Pattern, PS, Square_center, Search_Start_Rot, Search_Range_Rot, step = Step_Rot, Patch_Size = Patch_Size_Rot) ; Raw_Angle = angle ; #print(Raw_Angle)
            Oriented_Pattern = Rotate(Ref_Pattern, Square_center, angle) ; #print(Oriented_Pattern.shape)

            Spots = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1, spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Oriented_Pattern]

            Shifts = []
            for spot in Spots:
                Shifts.append(Get_Shift2Center(spot))

            Oriented_Pattern = Oriented_Pattern-np.array(Shifts)

            DistList = []

            for coord in Oriented_Pattern:
                DistX = coord[0]-Oriented_Pattern[:,0]
                DistY = coord[1]-Oriented_Pattern[:,1]
                Dist2D = np.sqrt(DistX**2+DistY**2)
                Dist2D = Dist2D[Dist2D<1.25*Planes]
                Dist2D = Dist2D[0<Dist2D]
                DistList.append(np.mean(Dist2D))

            Planes_Opt = np.mean(DistList) ; print('Optimized Plane Distance:',Planes_Opt)

            Pattern = MakePattern(Square_box, Planes_Opt)
            DistX = Pattern[:,0]-Square_center[0]
            DistY = Pattern[:,1]-Square_center[1]
            Dist2D = np.sqrt(DistX**2+DistY**2)
            Pattern_ResMatrix = apix/Dist2D*np.max(Square_box)
            Final_Pattern = Limit_Pattern_Resolution(Pattern, Pattern_ResMatrix, 10).astype(int)
            angle, intensity = Solve_Orientation_New(Final_Pattern, PS, Square_center, Search_Start_Rot, Search_Range_Rot, step = Step_Rot, Patch_Size = 10)
            Final_Pattern = Rotate(Final_Pattern, Square_center, angle)

            Spots = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1, spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Final_Pattern]
            Final_Shifts = []
            for spot in Spots:
                Final_Shifts.append(Get_Shift2Center(spot))

            Final_Pattern = Final_Pattern-np.array(Final_Shifts)

            Spots = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1, spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Final_Pattern]


            ### Calculating Peak Profile ###
            Spot_Profile    = np.mean(np.array(Spots),0)
            Spot_Mask       = Spot_Profile > np.percentile(Spot_Profile, PERCENT)
            Noise_Mask      = np.invert(Spot_Mask)

            ### PERFORMING SUBTRACTION ###
            Subtracted = []
            Signal_Mask_List = []

            New_Array = []
            for i in range(len(Spots)):
                Noise_lvl = np.mean(Spots[i][Noise_Mask])
                Signal_lvl = np.mean(Spots[i][Spot_Mask])
                SNR = Signal_lvl/Noise_lvl
                if 1.1 < SNR < 100:
                    Average_array = np.ones(Spot_Profile.shape)*Noise_lvl
                    Signal_Mask = Average_array*Spot_Mask.astype(int)
                    Signal_Mask_List.append(Noise_Mask.astype(int))
                    #Around_Mask = Spots[i]*Noise_Mask.astype(int) ; Around_Mask = np.where(Around_Mask == 0, 1, Around_Mask)
                    #Final_Mask = Signal_Mask+Noise_Mask.astype(int)
                    #Subtracted_Spot = Final_Mask+Around_Mask
                    #Subtracted.append(Subtracted_Spot)
                else:
                    #Subtracted.append(Spots[i])
                    Signal_Mask_List.append(np.ones(Spot_Profile.shape))

            Subtracted = np.array(Subtracted)
            PS_sub = np.copy(Fourier)

            for i in range(len(Final_Pattern)):
                PS_sub[Final_Pattern[i][0]-Patch_Size//2:Final_Pattern[i][0]+Patch_Size//2+1,
                       Final_Pattern[i][1]-Patch_Size//2:Final_Pattern[i][1]+Patch_Size//2+1] = PS_sub[Final_Pattern[i][0]-Patch_Size//2:Final_Pattern[i][0]+Patch_Size//2+1,
                                                                                                       Final_Pattern[i][1]-Patch_Size//2:Final_Pattern[i][1]+Patch_Size//2+1]*Signal_Mask_List[i]

            PS_sub_inv = fftpack.ifftshift(PS_sub)
            data_sub = np.real(fftpack.ifft2(PS_sub_inv))

            ### Display ###
            #x = np.arange(0,Square_box[0],1)
            #y = np.arange(0,Square_box[1],1)
            #micrograph.imshow(1-mrc.data, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])
            #spectrum.imshow(1-np.log10(PS)**2, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])
            #spectrum.plot(Final_Pattern[:,0],Final_Pattern[:,1], 'bo', fillstyle='none')
            ##for spot in Subtracted: spectrum_sub.imshow(1-np.log10(spot)**2, cmap=pl.cm.Greys) ; plt.pause(0.1)
            #micrograph_sub.imshow(1-data_sub, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])
            #spectrum_sub.imshow(1-np.log10(np.abs(PS_sub))**2, cmap=pl.cm.Greys, extent=[x.min(), x.max(), y.min(), y.max()])
            #plt.pause(2)

            ### REDIMENSION AND WRITE MRC ###
            data_sub = data_sub[Pad_Indexes[0]:Square_box[0]-Pad_Indexes[0],Pad_Indexes[1]:Square_box[1]-Pad_Indexes[1]]

            mrc.write(data_sub, mic[:-4]+'_sub.mrc')
            
        except:
            sh.move(mic,mic+'.failed')
            print('failed to process '+mic)

