#!/usr/bin/env python3
import os, sys, glob, mrcfile
import shutil as sh
import numpy as np
from scipy import fftpack, fft, ndimage, signal
import matplotlib.pyplot as plt
from MRC_RW_3 import *
from GridTools import *
from Pattern_Optimizer import *


 
def glacios_45k():
    apix = 0.901
    Planes = 66
    Profile_threshold = 0.2 # minimum ratio of signal to consider for peak profiling
    Sigma_threshold = 1.3
    Patch_Size = 11 # for global peak search and peak operations
    Patch_Size_Operation = 21 # 
    height = 0.3 # peak height (ratio of 1) for mosaicit search
    width = 3 # peak width (in step) for mosaicit search
    Patch_Fine = 3  # for parameters refinement only
    Pad = 200
    Resolution_limit = 10
    return (apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit)
    
def glacios_73k():
    apix = 0.562
    Planes = 41
    Profile_threshold = 0.25 # minimum ratio of signal to consider for peak profiling
    Sigma_threshold = 1.5
    Patch_Size = 20 # for global peak search and peak operations
    Patch_Size_Operation = 21 # 
    height = 0.3 # peak height (ratio of 1) for mosaicit search
    width = 3 # peak width (in step) for mosaicit search
    Patch_Fine = 5  # for parameters refinement only
    Pad = 200
    Resolution_limit = 10
    return (apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit)
    
def krios2_165k():
    apix = 0.729
    Planes = 56.6
    Profile_threshold = 0.1 # minimum ratio of signal to consider for peak profiling
    Sigma_threshold = 1.3
    Patch_Size = 11 # for global peak search and peak operations
    Patch_Size_Operation = 21 # 
    height = 0.3 # peak height (ratio of 1) for mosaicit search
    width = 3 # peak width (in step) for mosaicit search
    Patch_Fine = 3  # for parameters refinement only
    Pad = 200
    Resolution_limit = 10
    return (apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit)

if __name__ == '__main__':
    
    for f in glob.glob('**/*.mrc', recursive=True):
        if 'aligned_doseweighted' in f:
            preprocess_type = 'cryosparc'
        else:
            preprocess_type = 'warp'
        firstMrc = mrcfile.open(f)
        pixSize = firstMrc.header.field(10).field(2)
        break
        
    if '0.729' in str(pixSize):
        apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit = krios2_165k()
    elif '0.562' in str(pixSize):
        apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit = glacios_73k()
    elif '0.901' in str(pixSize):
        apix, Planes, Profile_threshold,Sigma_threshold, Patch_Size, Patch_Size_Operation, height, width, Patch_Fine, Pad, Resolution_limit = glacios_45k()
    else:
        print('This mrc was acquired in a condition, which was not yet prepared for lattice subtraction')
        print('Please contact the developer!')
        print('Exiting...')
        sys.exit()
    
    if preprocess_type == 'cryosparc':
        AllFiles = glob.glob('**/*.mrc', recursive=True)
        MRCFiles = glob.glob('**/*doseweighted.mrc', recursive=True)
    elif preprocess_type == 'warp':
        AllFiles = os.listdir(os.getcwd())
        MRCFiles = [element for element in AllFiles if element[-4:] == '.mrc' and 'sub' not in element]

    mrc = MRC()

    log = open('SAGsub_3.star','w')
    log.write('_loop\n\n_rlnMicrographName #1\n_rlnMosaicity #2\n_rlnSpotDistError #3\n_rlnStatus #4\n')

    for mic in MRCFiles:
    
        LOG = [None,None,None,None]
    
        try:
        
            if mic.replace('.mrc','_sub.mrc') not in AllFiles:
        
                print('####################################################################\nprocessing', mic)

                mrc.read(mic) ; LOG[0] = mic 

                box = np.array(mrc.data.shape)                  # Get micrograph dimensions
                Squarizer = max(box)-box                        # Calculate pixels to add to squarize micrograph
                box_padded = box+Squarizer+2*Pad                # Determine dimensions of the squarized, padded micrograph
                Pads = (box_padded-box)//2
                Center = box_padded//2      # Coordinates of the resized micrograp center
                Center = Center+0.5         #to center on the real PS center.
                
                x = np.arange(0,box_padded[0],1)
                y = np.arange(0,box_padded[1],1)

                Padder = np.ones(box_padded)*np.mean(mrc.data)              # Padding array
                Padder[Pads[0]:box[0]+Pads[0],Pads[1]:box[1]+Pads[1]] = mrc.data # Padding the micrograph
                
                #mrc.write(Padder, 'padded.mrc')

                Fourier = fftpack.fftshift(fftpack.fft2(Padder))   # Compute fourier transform of the padded micrograph
                PS = np.abs(Fourier)                            # Compute Power Spectrum from fourier transform


                ### Building Refinement Pattern ###

                Pattern = MakePattern(box_padded, Planes)+0.5

                # define a resolution-limited pattern for lattice refinement #

                DistX = Pattern[:,0]-Center[0]
                DistY = Pattern[:,1]-Center[1]
                Dist2D = np.sqrt(DistX**2+DistY**2)
                Pattern_ResMatrix = apix/Dist2D*np.max(box_padded)
                
                Resolution_distance_test = (np.max(box_padded)/53)/(2*apix) 

                LowRes_Pattern = Flank_Pattern_Resolution(Pattern, Pattern_ResMatrix, 100, 40) 

                ### Rotate matrix for coarse peak search ###
                
                Search_Range = 100
                Search_Step = 1

                Angles, Intensities = Rotation_Search(LowRes_Pattern, PS, Center, SearchStart = 0, Search_Range = Search_Range, step = Search_Step, Patch_Size = Patch_Size)
                
                ######################################################################

                BestAngle = Angles[np.argmax(Intensities)+1] 
                
                AngleLoc = signal.find_peaks(Intensities, height = height, width = width)
                MainLattices = AngleLoc[0] ; print(MainLattices)
                Mosaicity = len(MainLattices) 
                LOG[1] = Mosaicity

                Pattern_Rot = Rotate(LowRes_Pattern, Center, BestAngle)

                #############  Determine XY shifts to apply to recenter on spots  ###############

                patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
                    spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]

                Shifts = []

                for spot in patches:

                    bright = np.array(np.where(spot==spot.max())) 

                    Shifts.append([bright[0][0]-Patch_Size//2,bright[1][0]-Patch_Size//2])

                Shifts = np.array(Shifts) #; print("distance error:", np.round(np.sqrt(Shifts[:,0]**2+Shifts[:,1]**2),2))

                Pattern_Rot = Pattern_Rot+Shifts

                #############################  Refining Planes spacing  ######################################################

                DistXY = np.array([Pattern_Rot-Pattern_Rot[i] for i in range(len(Pattern_Rot))]) 
                Dist2D = np.array([np.sqrt(DistXY[i][:,0]**2+DistXY[i][:,1]**2) for i in range(len(DistXY))]) 
                min_Planes = [np.min(Dist2D[i][np.nonzero(Dist2D[i])]) for i in range(len(Dist2D))] 
                Optimized_Planes = np.mean(min_Planes) ; print('Optimized planes distance:', round(Optimized_Planes,2))
                LOG[2] = round(abs(Optimized_Planes-Planes),2)
                
                Pattern_Opt = MakePattern(box_padded, Optimized_Planes)+0.5 
                
                Pattern_Opt = Flank_Pattern_Resolution(Pattern_Opt, Pattern_ResMatrix, 100, 40) 
                
                if Mosaicity > 1:
                
                    Pattern_Pool = []

                    Patter_Pool_Final = []
                
                    for i in range(Mosaicity):
                
                            Pattern_Opt_oriented = Rotate(Pattern_Opt, Center, MainLattices[i]-10)
                            
                            Angles_Opt, Intensities_Opt = Rotation_Search(Pattern_Opt_oriented, PS, Center, SearchStart = 0, Search_Range = 11, step = 0.25, Patch_Size = Patch_Fine)
                            
                            BestAngle_Opt = Angles_Opt[np.argmax(Intensities_Opt)]+(MainLattices[i]-10) 
                            
                            Pattern_Final = MakePattern(box_padded, Optimized_Planes)+0.5

                            Pattern_Final = Rotate(Pattern_Final, Center, BestAngle_Opt)
                            
                            Pattern_Pool.append(Pattern_Final)

                    #print(len(Pattern_Pool))
                            
                    for lattice in Pattern_Pool:

                        for spot in lattice:

                            Patter_Pool_Final.append(spot)
                        
                    Pattern_Final = np.array(Patter_Pool_Final) 
                
                
                if Mosaicity == 1:
                
                    Pattern_Opt_oriented = Rotate(Pattern_Opt, Center, BestAngle-10)

                    Angles_Opt, Intensities_Opt = Rotation_Search(Pattern_Opt_oriented, PS, Center, SearchStart = 0, Search_Range = 11, step = 0.25, Patch_Size = Patch_Fine)

                    BestAngle_Opt = Angles_Opt[np.argmax(Intensities_Opt)]+(BestAngle-10)

                    #Pattern_Rot_Opt = Rotate(Pattern_Opt, Center, BestAngle_Opt)

                    Pattern_Final = MakePattern(box_padded, Optimized_Planes)+0.5

                    Pattern_Final = Rotate(Pattern_Final, Center, BestAngle_Opt)

                ############################# Apply optimized parameters to a large pattern

                DistX = Pattern_Final[:,0]-Center[0]
                DistY = Pattern_Final[:,1]-Center[1]
                Dist2D = np.sqrt(DistX**2+DistY**2)
                Pattern_ResMatrix = apix/Dist2D*np.max(box_padded)

                Pattern_Final_HR = Flank_Pattern_Resolution(Pattern_Final, Pattern_ResMatrix, 100, Resolution_limit)

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

                ####################### Iterate through spots ################################

                Spot_MetaData = []

                for s in range(len(patches_Opt)):

                    spot = patches_Opt[s]

                    coordinate = Pattern_Final_HR[s]

                    DistXY = coordinate-Center
                    Dist2D = np.sqrt(DistXY[0]**2+DistXY[1]**2)
                    Spot_Resolution = round(apix/Dist2D*np.max(box_padded),3)

                    Peak_value = np.mean(spot[Peak_Mask])
                    Bg_value = np.mean(spot[Bg_Mask])

                    Spot_MetaData.append([Peak_value/Bg_value, Spot_Resolution])

                Spot_MetaData = np.array(Spot_MetaData) 

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

                ############################################ SUBTRACTING SIGNAL ##########################################################

                PS_sub = np.copy(Fourier)

                for spot in Pattern_Final_HR_cleaned:
                    PS_sub[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] = PS_sub[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1]*Bg_Mask

                data_sub = np.real(fftpack.ifft2(fftpack.fftshift(PS_sub)))

                data_sub = data_sub[Pads[0]:box[0]+Pads[0],Pads[1]:box[1]+Pads[1]] #unpadding

                #mrc.write(1-np.log10(PS)**2,'PS.mrc')
                mrc.write(data_sub, mic)
                LOG[3] = 'SUCCESS'
            
        except:
            sh.move(mic,mic+'.failed')
            print('failed to process '+mic)
            LOG[3] = 'FAILED'
            
        for element in LOG: log.write(str(element)+'\t')
        log.write('\n')

log.close()
