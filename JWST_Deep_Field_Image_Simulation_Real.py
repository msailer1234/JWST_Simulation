# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:06:09 2021

@author: M.W.Sailer

This program simulates a randomly generated deep field image and its characteristics based on major parameters of the 
universe measured from observations. 1 graph is produced in this simulation depicting a more realistic representation of the
JWST Deep Field image.

Necessary Modules:
"""
import numpy as np
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Control Panel----------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------
"""

omegamass = 0.301 #Baryonic and dark matter mass density
omegalambda = 0.706 #Dark energy density
w = -0.994 #Equation of state parameter of dark energy (between -1/3 and -5/3), typically measured to be ~-1
c = 2.9979*10**5 #Speed of light in km/s
Hubble = 71.4 #Hubble Constant in km/s/Mpc
DE_Ratio = 0.925 # 1 means isotropic expansion, 0.9 equates to 90% of expansion (0.9*H_0) in the y-direction, etc.
xaxisminutes = 5.1333 # Size of x-axis (arcminutes) in simulation. (5.1333 (2-2.2' detectors with 0.7333' gap) for JWST, or 3.1 for HST)
yaxisminutes = 2.2 # Size of y-axis (arcminutes) in simulation (2.2 for JWST, 3.1 for HST)
N_Unseen = 6 # How many times more galaxies exist than are seen in UDF: anywhere from 2 to 10
R_Increase = 130 # % of the radius seen due to higher sensitivity of telescope: 0% means anything beyond the halflight radius is invisible, 160 means the visible radius is 1.6 times larger than the effective radius in Allen et al.'s equation
dpi=1724 #For HST: 1010 yields a resolution of 0.05 arcseconds, for JWST: 1724 yields a resolution of 0.031 arcseconds
ZMAXHUDF = 11 #The maximum measured redshifts in the HUDF image (z = 11)
ZMAX = 15 #The maximum redshift reached by the telescope (11 for UDF, 15 for JWST)
TolmanDimming = 3 #The factor of dimming proportional to (1+z)^x. Recent measurements indicate x = 3 rather than 4. 
SaturationZValue = (100*((1+0.35)**3))**(1/3)-1 #The redshift where galaxies are no longer saturated from long exposure. 0.35 is used as the end of the first bin from Inami et al.'s data of the HUDF image. This value is determined by solving the equation 100*(1+0.35)^3 = (1+z)^3

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------------
End of Control Panel---------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
Preliminary calculations from control panel inputs:
"""

#-----Calculation of the total energy density of the universe------------------------------------------
omegatotal = omegamass+omegalambda

#-----Calculation of the dark energy equation of state-------------------------------------------------
m = 3*(1+w)

#-----Calculation of the universe's age----------------------------------------------------------------
def Ageintegrand(x): #Integration to find the universe's age
    return 1/((((1-omegatotal)*(1+x)**2 + omegamass*(1+x)**3 + omegalambda*(1+x)**m)**0.5)*(1+x))
age, err = quad(Ageintegrand, 0, float('inf')) #integration

UniverseAge = age*(3.086*10**19)/(Hubble*(3.154*10**7))# converts to years


#-----A deep field image is 3-dimensional. Because of this, apparent angular sizes must be calculated for more than
#a single redshift. This simulation approximates a 3-dimensional universe by generating average galaxies on multiple planes
#at different redshifts. To do this, the galaxy density must be known at each redshift. This is a nearly impossible
#value to find as seen in data from Inami, et. al. (2017) Figure 13 (MUSE-z Distributions can be viewed). This is because
#telescopes don't detect every galaxy in existence. As distance increases, only the brightest galaxies are detected.
#Since a reliable equation for galaxy number density based on redshift cannot be obtained, a rough approximation is used until
#more data is found in future surveys.------------------------------------------------------------------------------------------

zlist = [] #start with 2 empty lists
Densitylist = [] #start with 2 empty lists

#Calculate the distance to the maximum redshift defined as ZMAX in Glyrs
ZMAXGlyrs = (UniverseAge/10**9) - (UniverseAge/10**9)/(ZMAX+1)

#Create a list of 24 redshifts to associate a specific plane. This typically corresponds to 0.5 Glyr bins depending on the initial conditions. 
for plane in list(np.linspace(1,int(ZMAXGlyrs),24)):
    zlist.append(((UniverseAge/10**9)/((UniverseAge/10**9) - plane))-1)
zlist.append(ZMAX)

#-------------Calculating the list of galaxy numbers on each plane--------------------------------------------------

#Comoving Volume function from Hogg (2000)'s equations.
def ComovingVolume(z):
    
    if omegatotal == 1:
        
        Dh = c/Hubble
        
        def integrand(x):
            return 1/(((1-omegatotal)*(1+x)**2 + omegamass*(1+x)**3 + omegalambda*(1+x)**m)**0.5)
        ans, err = quad(integrand, 0, z)
    
        Dc = Dh * ans
        Dm = Dc
        Vc = (4/3)*math.pi*Dm**3
        
    elif omegatotal > 1:
        
        Dh = c/Hubble
        
        def integrand(x):
            return 1/(((1-omegatotal)*(1+x)**2 + omegamass*(1+x)**3 + omegalambda*(1+x)**m)**0.5)
        ans, err = quad(integrand, 0, z)
    
        Dc = Dh * ans
        Dm = Dh*(1/abs(1-omegatotal)**0.5)*math.sinh(((abs(1-omegatotal))**0.5)*Dc/Dh)
        Vc = ((4*math.pi*Dh**3)/(2*(1-omegatotal)))*((Dm/Dh)*((1+(1-omegatotal)*(Dm/Dh)**2)**0.5)-((1/(abs(1-omegatotal))**0.5)*math.sinh((abs(1-omegatotal))**0.5 * Dm/Dh)))
        
    else:
        
        Dh = c/Hubble
        
        def integrand(x):
            return 1/(((1-omegatotal)*(1+x)**2 + omegamass*(1+x)**3 + omegalambda*(1+x)**m)**0.5)
        ans, err = quad(integrand, 0, z)
    
        Dc = Dh * ans
        Dm = Dh*(1/abs(1-omegatotal)**0.5)*math.sin(((abs(1-omegatotal))**0.5)*Dc/Dh)
        Vc = ((4*math.pi*Dh**3)/(2*(1-omegatotal)))*((Dm/Dh)*((1+(1-omegatotal)*(Dm/Dh)**2)**0.5)-((1/(abs(1-omegatotal))**0.5)*math.sin((abs(1-omegatotal))**0.5 * Dm/Dh)))
        
    return Vc

#find the integration constant "q" using the HUDF z = 11 approximation
#Integrate q * Comoving Volume * (1+z) from z = 0 to z = 11 and find q that sets integral equal to 10,000

def integrand(i):
    return (ComovingVolume(i)*(1+i))
ans, err = quad(integrand, 0, ZMAXHUDF)
q = 10000/ans #10000 is used for the calibration according to NASA's HUDF estimate

#Calculate the galaxy number densities on each plane by integrating q * Comoving Volume * (1+z) between each zvalue in zlist

for entry in range(len(zlist)-1): #For each plane
    #Run through same integral between each z value
    def integrand(i):
        return q*(ComovingVolume(i)*(1+i))
    ans, err = quad(integrand, zlist[entry], zlist[entry+1])
    Densitylist.append(ans)
    
#--------------Convert Densitylist into number of galaxies per square arcminute----------------------------------------------------------------

for entry in range(len(Densitylist)):
    if Densitylist[entry]/(3.1*3.1) < 1:
        Densitylist[entry] = 1
    else:    
        Densitylist[entry] = Densitylist[entry]/(3.1*3.1)
    
#As a calibration, the number of galaxies are multiplied by N_Unseen on each plane less than z = 1.
#For reference, Conselice et al. (2016) estimates there exist 10 times as many galaxies as can be seen in the HUDF.
#Lauer et al. (2021) estimates only 2 times as many. No matter which number is chosen, a correction must be made to 
#account for these unseen galaxies at low redshifts since they should be seen more readily due to the close proximity.  
#Similarly, the galaxy densities at redshifts should be increased with the JWST (not HST) due to the increased 
#sensitivity of the JWST. All of this is accounted for below.

for z in range(len(zlist)):
    if zlist[z] < 1:
        Densitylist[z] = Densitylist[z] *N_Unseen
    else:
        pass

if xaxisminutes == 5.1333:
    for density in range(len(Densitylist)):
        if zlist[density] > 1:    
            Densitylist[density]=Densitylist[density]*N_Unseen #This will automatically change the density to estimated values (only for galaxies further than z = 1)
        else:
            pass
else:
    pass

#Average lengths at each redshift plane according to Allen et. Al (2017) r_half-light = 7.07(1+z)^-0.89 kpc
#Allen et al's equations are only used at z>1 since galaxies do not appear to grow as much in recent history. 
#These lengths are then increased 160% due to the 100x sensitivity of the JWST according to their Seric profiles.
Lengthlist = []
for z in range(len(zlist)):
    
    if zlist[z] < 1:
        AverageLength = 7.07*(1+1)**(-0.89) *2 #multiply by 2 to get diameter
        AverageLength = (AverageLength/1000) #Convert from kpcs to Mpcs
    else:
        AverageLength = 7.07*(1+zlist[z])**(-0.89) *2 #multiply by 2 to get diameter
        AverageLength = (AverageLength/1000) # Convert from kpc to Mpcs
    
    if zlist[z] > 1:
        AverageLength =  AverageLength*(R_Increase)/100 #convert to Mpc and then adjust based on half-light radius factor
    else:
        pass
    Lengthlist.append(AverageLength)

#Definition of k according to the universe's geometry: 1 for critical density, >1 for spherical, <1 for saddle-shaped
if omegatotal == 1:
    k = 0
elif omegatotal > 1:
    k = 1
else:
    k = -1


"""
End of preliminary calculations
"""

##############################################################################################

"""
Calculation of apparent size using equation 48 from SAHNI, V., & STAROBINSKY, A. (2000). 
Equation tested with the sizes and distances of the Bullet Cluster and the Musket Ball Cluster
as well as a recent paper: Balakrishna Subramani, et al. (2019). 
This equation has been demonstrated to be accurate for distant objects.
"""
#Integration of E(z)
def EZintegrand(x):
    return 1/(((1-omegatotal)*(1+x)**2 + omegamass*(1+x)**3 + omegalambda*(1+x)**m)**0.5)

#Function of SAHNI, V., & STAROBINSKY, A. (2000)'s equations.
def ApparentAngularDiameter(input,i):
    if k == 0:

        ans, err = quad(EZintegrand, 0, i)
        theta = input*(1+i)**2/((1+i)*c*(1/Hubble)*ans)

    elif k == 1:

        ans, err = quad(EZintegrand, 0, i)
        theta = input*(1+i)**2*(abs(omegatotal-1)**0.5)/((1+i)*c*(1/Hubble)*math.sin((abs(omegatotal-1)**0.5)*ans))

    else:

        ans, err = quad(EZintegrand, 0, i)
        theta = input*(1+i)**2*(abs(omegatotal-1)**0.5)/((1+i)*c*(1/Hubble)*math.sinh((abs(omegatotal-1)**0.5)*ans))
        
    theta = theta*(180/math.pi)*60
    return theta

"""
Now that apparent angular sizes have been found, a deep field image can be generated
Galaxies are generated using random ellipses. The major axis of each ellipse is equal to the apparent angular size
at the specified z. The minor axis is a random value between 0% the apparent angular size and 100% to simulate 
a random face-orientation of each galaxy since galaxies are seen anywhere from edge-on to face-on. The 1% lower
boundary comes from the Milky Way's own proportions. Each galaxy is also randomly oriented at a direction between 0 and 360
degrees since galaxies can be at any angle. 

This will generate Figure 3 (Deep field image)
"""

#Set background to black
plt.style.use("dark_background")

#Define figure
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

#Create an empty list to count number of galaxies generated
EllipseNumber = []

#Run through each redshift layer
for layer in range (len(Densitylist)):
    
    #Calculation of the apparent angular diameters from SAHNI, V., & STAROBINSKY, A. (2000)'s equations.
    theta = ApparentAngularDiameter(Lengthlist[layer],zlist[layer])

    #Calculation of the apparent angular diameters from trigonometry if no dark energy existed (for anisotropy incorporation later on)
    NewDistance = (UniverseAge-(UniverseAge/(1+zlist[layer])))/(3.262*10**6)#Convert z value to distance in Mpc
    NoDarkEnergyTheta = (180/math.pi)*np.arctan(Lengthlist[layer]/NewDistance)*60 #180/pi converts from radians to degrees, *60 converts degrees to arcminutes

    #The simulated galaxies will be displayed in a grid. This calculates the number of x and y grid points needed given the
    #galaxy number density of each layer and number of arcminutes in the x and y axis. 
    Numberx = int(xaxisminutes*(Densitylist[layer]**0.5))
    Numbery = int(yaxisminutes*(Densitylist[layer]**0.5))
    
    #Set up the grid
    x = np.linspace(0,xaxisminutes,Numberx) 
    y = np.linspace(0,yaxisminutes,Numbery) 
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    #Generate the ellipses:
    #Create empty list of ellipses
    ells=[]

    #Run through each x and y grid point
    for xx in range(Numberx):
        for yy in range(Numbery):
                        
                        #Generate a random ellipse orientation angle from 0 to 360 degrees with respect to the x axis
                        EllipseAngle = np.random.rand()*360 #360 because input of ellipse is degrees
                        EllipseAngleRadians = math.radians(EllipseAngle)#Convert the ellipse angles to radians since python's trig functions require radians instead of degrees
                        
                        #Calculate the new angle of orientation from the image compression due to anisotropy of dark energy acting on SAHNI, V., & STAROBINSKY, A. (2000)'s equations.
                        if EllipseAngleRadians <= math.radians(90): 
                            EllipseAngleNew = math.degrees(math.atan(((NoDarkEnergyTheta*math.sin(EllipseAngleRadians)) + DE_Ratio*(theta*math.sin(EllipseAngleRadians) - NoDarkEnergyTheta*math.sin(EllipseAngleRadians)))/(theta*math.cos(EllipseAngleRadians))))
                            EllipseAnglePlot = EllipseAngleNew
                        elif EllipseAngleRadians <= math.radians(180):
                            EllipseAngleRadians = math.radians(180)-EllipseAngleRadians
                            EllipseAngleNew = math.degrees(math.atan(((NoDarkEnergyTheta*math.sin(EllipseAngleRadians)) + DE_Ratio*(theta*math.sin(EllipseAngleRadians) - NoDarkEnergyTheta*math.sin(EllipseAngleRadians)))/(theta*math.cos(EllipseAngleRadians))))
                            EllipseAnglePlot = 180 - EllipseAngleNew
                        elif EllipseAngleRadians <= math.radians(270):
                            EllipseAngleRadians = EllipseAngleRadians-math.radians(180)
                            EllipseAngleNew = math.degrees(math.atan(((NoDarkEnergyTheta*math.sin(EllipseAngleRadians)) + DE_Ratio*(theta*math.sin(EllipseAngleRadians) - NoDarkEnergyTheta*math.sin(EllipseAngleRadians)))/(theta*math.cos(EllipseAngleRadians))))
                            EllipseAnglePlot = 180 + EllipseAngleNew
                        else:
                            EllipseAngleRadians = math.radians(360)-EllipseAngleRadians
                            EllipseAngleNew = math.degrees(math.atan(((NoDarkEnergyTheta*math.sin(EllipseAngleRadians)) + DE_Ratio*(theta*math.sin(EllipseAngleRadians) - NoDarkEnergyTheta*math.sin(EllipseAngleRadians)))/(theta*math.cos(EllipseAngleRadians))))
                            EllipseAnglePlot = 360 - EllipseAngleNew
                        
                        #Calculate the Ellipse Width:
                        #Equal to theta with no anisotropy
                        #If anisotropy exists and H0 is less in the y axis than the x axis, the resulting apparent shapes of high-z galaxies will be compressed towards the x axis.
                        EllipseWidth = theta*math.cos(EllipseAngleRadians)/math.cos(math.radians(EllipseAngleNew))
                        
                        #Calculate the Ellipse Height:
                        #Similar to Ellipse Width, but with slight difference to equation and random factor from 0 to 1 (simulating random edge-on to top down view)
                        EllipseHeight = np.random.rand()*theta*math.sin(math.radians(EllipseAngleNew))/math.sin(EllipseAngleRadians) 
                        
                        #Create ellipse with x,y inputs (adjusted for random distribution), width inputs, height inputs, and angle orientation. Add to ells list.
                        ells.append(Ellipse(xy=(x[xx]+(xaxisminutes/Numberx)*(-0.5+np.random.rand()),y[yy]+(yaxisminutes/Numbery)*(-0.5+np.random.rand())), #adds random factor by changing ellipse location by +/- half the distance between grid points
                        width=EllipseWidth, height=EllipseHeight,
                        angle=EllipseAngle))     

                        EllipseNumber.append(0)

    #Calculate the Tolman Dimming
    BrightnessFactor = (1+SaturationZValue)**TolmanDimming/((1+zlist[layer])**TolmanDimming)#Tolman Dimming - Calculated using saturation from long exposure and Tolman dimming factor.

    if BrightnessFactor > 1:
        BrightnessFactor = 1
    else:
        pass
    
    #Plot ellipses
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(BrightnessFactor)
        e.set_facecolor([1,1,1])

    #Set axis limits
    ax.set_xlim(0, xaxisminutes)#xaxisminutes)
    ax.set_ylim(0, yaxisminutes)#yaxisminutes)
    
    #Plot image title based on initial parameters 
    if xaxisminutes == 5.1333:
        plt.title('A James Webb Space Telescope NIRCam Deep Field Simulation \nReaching a Distance of Z='+str(ZMAX)+' with a Total of '
        +str(int(len(EllipseNumber)*80244/93618))+' Galaxies')#subtracts added galaxies behind the black bar in the image
    else:    
        plt.title('A James Webb Space Telescope NIRCam Deep Field Simulation  \nReaching a Distance of Z='+str(ZMAX)+' with a Total of '
                  +str(len(EllipseNumber))+' Galaxies')

    #Plot axis titles and show layer
    plt.xlabel('(\u03f4) arcmin')
    plt.ylabel('(\u03f4) arcmin')
    plt.show()

#If JWST parameters are chosen, this splits the image with a black bar of 0.7333 arcminutes width to create two 2.2x2.2 fields in one image as the JWST NIRCam will do.
if xaxisminutes == 5.1333:
    plt.axvline(x=2.567, color='k', linewidth = 50)
    plt.axvline(x=2.2, color='w', linewidth = 1)
    plt.axvline(x=2.9333, color='w', linewidth = 1)
    plt.show()
else:
    pass

#Save the figure as "fig3"  
plt.savefig('fig3.png',dpi=dpi)