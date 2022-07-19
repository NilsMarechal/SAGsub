import numpy as np
from scipy import signal
from GridTools import *

def Solve_Orientation(Pattern, PS, Center, SearchStart = 0, Search_Range = 180, step = 1, Patch_Size = 20):

    Intensities = []

    for i in range(SearchStart, int((SearchStart+Search_Range)/step), 1):
                
        Pattern_Rot = Rotate(Pattern, Center, i*step)

        #plt.plot(Pattern_Rot[:,0],Pattern_Rot[:,1], 'ro', fillstyle='none') ; plt.pause(30)
                
        patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
                      spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]

        #plt.imshow(1-np.log10(patches[0])**2, cmap=plt.cm.Greys) ; plt.pause(0.5)
                
        Intensity = np.mean(patches) ; #print(i, Intensity)
        Intensities.append([i*step, Intensity])

    Intensities = np.array(Intensities)
    patches = np.array(patches)
            
    Angle = Intensities[np.where(Intensities[:,1] == np.max(Intensities[:,1]))[0][0]][0] ; #print(Angle)

    return Angle

def Solve_Orientation_New(Pattern, PS, Center, SearchStart = 0, Search_Range = 180, step = 1, Patch_Size = 20):

    Intensities = []

    for i in range(SearchStart, int((SearchStart+Search_Range)/step), 1):
                
        Pattern_Rot = Rotate(Pattern, Center, i*step)
                
        patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
                      spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]
                

        Intensity = np.mean(patches)
        Intensities.append([i*step, Intensity])

    Intensities = np.array(Intensities)
    patches = np.array(patches)
            
    Angle = Intensities[np.where(Intensities[:,1] == np.max(Intensities[:,1]))[0][0]][0] ; #print(Angle)

    Intensity = Intensities[np.where(Intensities[:,1] == np.max(Intensities[:,1]))[0][0]][1]

    return Angle, Intensity


def Solve_Stigmation(Pattern, PS, Center, SearchStartXY = 0.8, SearchRangeX = 40, SearchRangeY  = 40, SearchStep = 0.01, Patch_Size = 20):

    Intensities = []

    for i in range(0, SearchRangeX, 1):
        stigX = SearchStartXY+i*SearchStep
        StigmatedX=StigmateX(Pattern, Center[0], stigX)
        for j in range(0, SearchRangeY, 1):
            stigY = SearchStartXY+j*SearchStep
            StigmatedY=StigmateY(StigmatedX, Center[1], stigY)
             #plt.plot(stigX,stigY, 'go')

            patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
                          spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in StigmatedY]

            #plt.imshow(1-np.log10(patches[0])**2, cmap=plt.cm.Greys) ; plt.pause(0.5)
                    
            Intensity = np.mean(patches) ; #print(Intensity)
            Intensities.append([(stigX, stigY), Intensity])
            
    Intensities = np.array(Intensities) ; #print(Intensities[:,2])
    Shifts = Intensities[np.where(Intensities[:,1] == np.max(Intensities[:,1]))[0][0]][0] #; print(Shifts)

    return Shifts

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

def Get_Shift2Center(Array):

    Raw,Col = Array.shape
    Center = Raw//2 ; #print(Raw,Col,Center)

    MaxPixel = np.concatenate(np.where(Array == Array.max())) ; #print(Array.shape, MaxPixel)
    
    return Center-MaxPixel


############################################################# DEV #############################################################

def Solve_Orientations(Pattern, PS, Center, SearchStart = 0, Search_Range = 180, step = 1, Patch_Size = 20):

    out=open('intensities.txt','w')

    Intensities = []

    for i in range(SearchStart, int((SearchStart+Search_Range)/step), 1):
                
        Pattern_Rot = Rotate(Pattern, Center, i*step)

        #plt.plot(Pattern_Rot[:,0],Pattern_Rot[:,1], 'ro', fillstyle='none') ; plt.pause(30)
                
        patches = [PS[spot[0]-Patch_Size//2:spot[0]+Patch_Size//2+1,
                      spot[1]-Patch_Size//2:spot[1]+Patch_Size//2+1] for spot in Pattern_Rot]

        #plt.imshow(1-np.log10(patches[0])**2, cmap=plt.cm.Greys) ; plt.pause(0.5)
                
        Intensity = np.mean(patches) ; #print(i, Intensity)
        Intensities.append([i*step, Intensity])
        #out.write(str(i*step)+' '+str(Intensity)+'\n')

    Intensities = np.array(Intensities)
    Normalized_Intensities = Intensities
    Normalized_Intensities[:,1] = (Intensities[:,1]-np.min(Intensities[:,1]))/(np.max(Intensities[:,1])-np.min(Intensities[:,1]))
    patches = np.array(patches)

    for element in Normalized_Intensities: out.write(str(element[0])+' '+str(element[1])+'\n')
            
    Angle = Normalized_Intensities[np.where(Normalized_Intensities[:,1] == np.max(Normalized_Intensities[:,1]))[0][0]][0] ; #print(Angle)

    Peaks = signal.find_peaks(Normalized_Intensities[:,1], width=Patch_Size*2)[0]*step
    mosaicity = int(len(Peaks)/(360/Search_Range))
    angles = Peaks[:mosaicity]


    out.close()

    return angles


    
