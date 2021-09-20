# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:06:09 2021

@author: M.W.Sailer

This program simulates a randomly generated deep field image and its characteristics based on major parameters of the 
universe measured from observations. 6 graphs are produced in this simulation: 

1. Apparent angular size vs z for an object of fixed proper length in a universe undergoing expansion from dark energy. 
2. Apparent angular size vs z for the same object in a universe without dark energy for comparison.
3. The geometric-focused deep field simulation with galaxies represented as ellipses.
4. A reference graph with a filled deep field image necessary for the galaxy coverage percentage calculation.
5. A reference graph with an empty deep field image necessary for the galaxy coverage percentage calculation.
6. Plot of the average angles of galaxy orientation to display anisotropy in the x/y direction.

Necessary Modules:
"""
import numpy as np
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2  

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
DE_Ratio = 1 # 1 means isotropic expansion, 0.9 equates to 90% of expansion (0.9*H_0) in the y-direction, etc.
xaxisminutes = 3.1 # Size of x-axis (arcminutes) in simulation. (4.4 for JWST, or 3.1 for HST)
yaxisminutes = 3.1 # Size of y-axis (arcminutes) in simulation (2.2 for JWST, 3.1 for HST)
N_Unseen = 6 # How many times more galaxies exist than are seen in UDF: estimated to be between 2 to 10
R_Increase = 100 # % of a galaxy's radius seen due to higher sensitivity of telescope: 100% means anything beyond the halflight radius is invisible, 160 means the visible radius is 1.6 times larger than the effective radius in Allen et al. (2017)'s equation
dpi=1010 #For HST: 1010 yields a resolution of 0.05 arcseconds, for JWST: 1724 yields a resolution of 0.031 arcseconds
ZMAXHUDF = 11 #The maximum measured redshifts in the HUDF image (z = 11)
ZMAX = 11 #The maximum measurable redshift reached by the telescope (11 for HUDF, 15 for JWST)

#Initial values for first 2 images. These do not impact the deep field simulation.
Length = 0.025 # proper length (Mpc) used for first two graphs
ZValue = 20 #Maximum redshift in x-axis of first two graphs

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
age, err = quad(Ageintegrand, 0, float('inf')) 

UniverseAge = age*(3.086*10**19)/(Hubble*(3.154*10**7))# converts to years

#-----A deep field image is 3-dimensional. Because of this, apparent angular sizes must be calculated for more than
#a single redshift. This simulation approximates a 3-dimensional universe by generating average galaxies on multiple planes
#at specific redshifts. To do this, the galaxy density must be known at each redshift. This is a nearly impossible
#value to find as seen in data from Inami, et. al. (2017) Figure 13 (MUSE-z Distributions can be viewed). This is because
#telescopes don't detect every galaxy in existence. As distance increases, only the brightest galaxies are detected.
#Since a reliable equation for galaxy number density based on redshift cannot be obtained, a rough approximation is used until
#more data is found in future surveys. This approximation assumes galaxy density changes proportional to the Comoving volume X
#(1+z) to account for expansion and the merger rate estimate according to Conselice et al. (2016).

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

#For both HST and JWST simulations, this calibration is made for z<1.
for z in range(len(zlist)):
    if zlist[z] < 1:
        Densitylist[z] = Densitylist[z] *N_Unseen
    else:
        pass

#For the JWST simulation only, N_Unseen is added for the rest of the galaxies.
if xaxisminutes == 4.4:
    for density in range(len(Densitylist)):
        if zlist[density] > 1:    
            Densitylist[density]=Densitylist[density]*N_Unseen #This will automatically change the density to estimated values (only for galaxies further than z = 1)
        else:
            pass
else:
    pass

#Average lengths at each redshift plane according to Allen et. Al (2017) r_half-light = 7.07*(1+z)^-0.89 kpc
#Allen et al's equations are only used at z>1 since galaxies do not appear to grow as much in recent history. 
#These lengths are then increased up to 160% due to the 100x sensitivity of the JWST according to their Seric profiles.
Lengthlist = []
for z in range(len(zlist)):
    
    if zlist[z] < 1:
        AverageLength = 7.07*(1+1)**(-0.89) *2 #multiply by 2 to get diameter
        AverageLength = (AverageLength/1000) #Convert from kpcs to Mpcs
    else:
        AverageLength = 7.07*(1+zlist[z])**(-0.89) *2 #multiply by 2 to get diameter
        AverageLength = (AverageLength/1000) # Convert from kpc to Mpcs
    
    if zlist[z] > 1:
        AverageLength =  AverageLength*(R_Increase)/100 #adjust based on half-light radius factor
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

####################################################################################################################

#####The Main Calculations:#########################################################################################

####################################################################################################################

"""
Calculation of apparent size using equation 48 from SAHNI, V., & STAROBINSKY, A. (2000). 
Equation tested with the sizes and distances of the Bullet Cluster and the Musket Ball Cluster
as well as a recent paper: Balakrishna Subramani, et al. (2019). This equation has been 
demonstrated to be accurate for distant objects.
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
FIGURE 1
Now an apparent size can be calculated using equation 48 from SAHNI, V., & STAROBINSKY, A. (2000). 
From testing the sizes and distances of the Bullet Cluster and the Musket Ball Cluster as well as 
a recent paper: Balakrishna Subramani, et al. (2019). This equation has been demonstrated to be 
accurate for distant objects. Equation 48 changes based on k, so an if statement is used for this 
calculation. A graph is generated (Figure 1) depicting the apparent angular size of an average 
galaxy vs Z.
"""

#Create a list of z values from close ~0 to the specified ZValue
xlist = np.linspace(0.0001,ZValue,10000)

#Create empty ylist
ylist = []

#fill ylist with values of theta given z and the specified test length
for i in xlist:
    ylist.append(ApparentAngularDiameter(Length,i)) #180/pi converts from radians to degrees, *60 converts degrees to arcminutes

#Plot and save image
plt.plot(xlist, ylist)
plt.xlabel('redshift')
plt.ylabel('angular size \u0394 \u03f4 (arcminutes)')
plt.title('Apparent Angular Size vs Redshift \nWith Dark Energy \nGalaxy Proper Length: '+str(Length)+' Mpcs')
plt.xlim(0,ZValue)
plt.ylim(0,ylist[9999]*2)
plt.show()
plt.savefig('fig1.png',dpi=dpi)

#Below prints out the apparent magnitude of the specific ZValue defined above
i = ZValue
theta = ApparentAngularDiameter(Length,i)
print('Apparent Magnitude of Test Length '+str(Length)+' Mpcs at z = '+str(i)+': '+str(round(theta,4)) +' arcminutes')

      
"""
FIGURE 2
A second plot is generated using trigonometry to plot a graph of apparent angular size vs Z if the universe was not expanding, 
and the angular size simply decreased with distance (Figure 2). This is created for reference.
"""

#Next Figure
fig, ax = plt.subplots()

#Create new empty ylist
newylist = []

#Calculate a theta using basic trigonometry for each z value
for i in xlist:
    NewDistance = (UniverseAge-(UniverseAge/(i+1)))/(3.262*10**6)#Convert z value to distance in Mpc
    ThetaNoDarkEnergy = (180/math.pi)*np.arctan(Length/NewDistance)*60 #180/pi converts from radians to degrees, *60 converts degrees to arcminutes
    newylist.append(ThetaNoDarkEnergy)

#Plot and save image
plt.plot(xlist, newylist)
plt.xlabel('redshift')
plt.ylabel('angular size \u0394 \u03f4 (arcminutes)')
plt.title('Apparent Angular Size vs Redshift \nWithout Dark Energy \nGalaxy Proper Length: '+str(Length)+' Mpcs')
plt.xlim(0,ZValue)
plt.ylim(0,ylist[9999]*2)
plt.show()
plt.savefig('fig2.png',dpi=dpi)

"""
FIGURE 3
Now that estimated apparent angular sizes have been found for each redshift bin, a deep 
field image can be generated. Galaxies are generated using randomly generated ellipses. 
The major axis of each ellipse is equal to the apparent angular size at the specified z 
values. The minor axis is a random value between 0% the apparent angular size and 100% to 
simulate a random face-orientation of each galaxy since galaxies are seen anywhere from edge-on 
to face-on. The 1% lower boundary comes from the Milky Way's own proportions. Each galaxy is also 
randomly oriented at a direction between 0 and 360 degrees since galaxies can be at any angle. 

This will generate Figure 3 (Deep field image)
"""

#A new figure
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

#Set up an empty list to save the angle of orientation of every ellipse generated. This will be useful when analyzing the effects of anisotropy.
EllipseAnglesTotal = []

#Run through each redshift plane.
for layer in range (len(Densitylist)):
    
    #Flip the order of layers so furthest layer is generated first. This will allow proper overlapping to occur so high-z galaxies are not generated
    #ontop of low-z galaxies
    layer = len(Densitylist)-1-layer
    
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
                angle=EllipseAnglePlot))
                
                #Save the ellipse angle
                EllipseAnglesTotal.append(EllipseAnglePlot)

    #Plot ellipses
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)
        #The color of each ellipse is set so that the ellipses change from blue to red depending on the distance. The colors change proportional to 1/distance(Glyrs)
        e.set_facecolor([1-1/(layer+1),0,1/(layer+1)])

    #Set axis limits
    ax.set_xlim(0, xaxisminutes)#xaxisminutes)
    ax.set_ylim(0, yaxisminutes)#yaxisminutes)
    
    #Plot axis titles and show layer
    plt.title('A Deep Field Simulation Accounting for the Angular-Diameter-Redshift \nRelation Reaching a Distance of Z='+str(ZMAX)+' with a Total of '
              +str(len(EllipseAnglesTotal))+' Galaxies')
    plt.xlabel('Arcminutes')
    plt.ylabel('Arcminutes')
    plt.show()
    
#Save the figure
plt.savefig('fig3.png',dpi=dpi)


#Calculating the average galaxy angle of orientation in the deep field simulation

#Start with 4 values of zero
avgright = 0
numberright = 0
avgleft = 0
numberleft = 0

#Analyze each angle and split them into quadrants. Add up every angle in each quadrant and divide by number of angles to determine the average angle.
for angle in EllipseAnglesTotal:
    if angle<=90:
        avgright+=angle
        numberright+=1
    elif angle>180 and angle<=270:
        avgright+=(angle-180)
        numberright+=1
    elif angle<=180 and angle>90:
        avgleft+=(angle)
        numberleft+=1
    else:
        avgleft+=(angle-180)
        numberleft+=1
#Calculate the average positively oriented angle (quadrants 1 and 3) and negatively oriented angle (quadrants 2 and 4).
AverageAngleRight = avgright/numberright
AverageAngleLeft = avgleft/numberleft

#Print out average positively and negatively oriented angles
print('Average Positively-Oriented Angle = '+str(round(AverageAngleRight,4)))
print('Average Negatively-Oriented Angle = '+str(round(AverageAngleLeft,4)))

#Print out the average distance from the 45 degree and 135 degree lines
print('Anisotropy average distance from 45 or 135 degree angles = '+str(round((abs(45-AverageAngleRight)+abs(45-(AverageAngleLeft-90)))/2,4)))

"""
FIGURES 4 AND 5
2 Reference Images are created to calculate how much space is covered by the galaxies (Figure 4 and Figure 5)
"""

"""
FIGURE 4: A deep field image completely covered by ellipses
"""
#A new figure
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

#Arbitrarily choose a large ellipse diameter to cover the image
theta = 10
    
#Arbitrarily choose a grid spacing to display the large ellipses
Numberx = 5
Numbery = 5
x = np.linspace(0,xaxisminutes,Numberx) 
y = np.linspace(0,yaxisminutes,Numbery) 
X, Y = np.meshgrid(x, y)
XY = np.column_stack((X.ravel(), Y.ravel()))

#Create new ellipse list
ells=[]

#Create the ellipses
for xx in range(Numberx):#Number):
    for yy in range(Numbery):#Number):
            EllipseAngle = np.random.rand()*360 #360 because input of ellipse is degrees
            EllipseAngleRadians = EllipseAngle*2*math.pi/360
            EllipseWidth = theta
            EllipseHeight = theta
            ells.append(Ellipse(xy=(x[xx]+(xaxisminutes/Numberx)*(-0.5+np.random.rand()),y[yy]+(yaxisminutes/Numbery)*(-0.5+np.random.rand())), #adds random factor by changing ellipse location by +/- galaxy radius
            width=EllipseWidth, height=EllipseHeight,
            angle=EllipseAngle))

#Make ellipses fully opaque
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(1) #full opaque color = galaxy overlap
    e.set_facecolor('r')

#Plot Ellipses with the exact same title, axes, size, etc. as the deep field image
ax.set_xlim(0, xaxisminutes)#xaxisminutes)
ax.set_ylim(0, yaxisminutes)#yaxisminutes)
plt.title('A Deep Field Simulation Accounting for the Angular-Diameter-Redshift \nRelation Reaching a Distance of Z='+str(ZMAX)+' with a Total of '
  +str(len(EllipseAnglesTotal))+' Galaxies')
plt.xlabel('Arcminutes')
plt.ylabel('Arcminutes')
plt.show()

#Save the figure as "Maximum"
plt.savefig('Maximum.png',dpi=dpi)

"""
FIGURE 5: A deep field image completely empty of ellipses
"""
#A new figure
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

#Plot Ellipses with the exact same title, axes, size, etc. as the deep field image
ax.set_xlim(0, xaxisminutes)#xaxisminutes)
ax.set_ylim(0, yaxisminutes)#yaxisminutes)
plt.title('A Deep Field Simulation Accounting for the Angular-Diameter-Redshift \nRelation Reaching a Distance of Z='+str(ZMAX)+' with a Total of '
  +str(len(EllipseAnglesTotal))+' Galaxies')
plt.xlabel('Arcminutes')
plt.ylabel('Arcminutes')
plt.show()

#Save the figure as "Minimum"
plt.savefig('Minimum.png',dpi=dpi)


"""
FIGURE 6
Finally, a 6th image is generated which displays the effects from anisotropy.
"""

#A new figure
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

#Plot a cartesian plane
V = np.array([[1,0], [-2,0], [0,-2], [0,1]])
origin = np.array([[0, 0, 0, 0],[0, 0, 0, 0]]) # origin point
plt.quiver(*origin, V[:,0], V[:,1], color=['k','k','k','k'], scale=2, width=0.015)
V = np.array([[1,1], [-1,-1], [1,-1], [-1,1]])
origin = np.array([[0, 0, 0, 0],[0, 0, 0, 0]]) # origin point
plt.quiver(*origin, V[:,0], V[:,1], color=['k','k','k','k'], scale=2, linestyle='dashed', width=0.001)

#Plot 2 vectors representing the average positively and negatively oriented galaxy angle 
V = np.array([[math.cos(AverageAngleRight*2*math.pi/360),math.sin(AverageAngleRight*2*math.pi/360)], [math.cos(AverageAngleLeft*2*math.pi/360),math.sin(AverageAngleLeft*2*math.pi/360)]])
origin = np.array([[0, 0],[0, 0]]) # origin point
plt.quiver(*origin, V[:,0], V[:,1], color=['r','b',], scale=2, headwidth=2,width=0.01)
V = np.array([[-math.cos(AverageAngleRight*2*math.pi/360),-math.sin(AverageAngleRight*2*math.pi/360)], [-math.cos(AverageAngleLeft*2*math.pi/360),-math.sin(AverageAngleLeft*2*math.pi/360)]])
origin = np.array([[0, 0],[0, 0]]) # origin point
plt.quiver(*origin, V[:,0], V[:,1], color=['r','b',], scale=2, headwidth=1,width=0.01)

#Format the image
ax = fig.gca()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks(np.arange(-1, 1.5, 0.5))
ax.set_yticks(np.arange(-1, 1.5, 0.5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
plt.title('The Average Positively-Oriented Angle (Red) \nand Negatively-Oriented Angle (Blue) \nRepresented with Vectors of Magnitude 1')
plt.xlabel('Total number of galaxies= '+str(len(EllipseAnglesTotal))+'    Positive: '+str(numberright)+'    Negative: '+str(numberleft))

plt.show()
plt.savefig('fig4.png',dpi=dpi)

#Print the average galaxy angle orientation. This should also point along the direction of less expansion indicating an axis for anisotropy
print('Direction of anisotropy: '+str(round((AverageAngleRight+AverageAngleLeft)/2,4)))

"""
FINAL CALCULATION
Now a percentage of space taken up by the galaxies can be calculated by importing the 2 reference images and the deep field image.
"""

#The previously generated "Maximum" reference image is loaded into python and the number of white pixels is counted
img = cv2.imread('Maximum.png', cv2.IMREAD_GRAYSCALE)
n_white_pix_max = np.sum(img == 255)

#The previously generated "Minimum" reference image is loaded into python and the number of white pixels is counted
img = cv2.imread('Minimum.png', cv2.IMREAD_GRAYSCALE)
n_white_pix_min = np.sum(img == 255)

#The white pixel difference is found representing the total number of pixels lying in the deep field simulation image. 
n_difference = n_white_pix_min-n_white_pix_max
ntotal = img.size

#The previously generated deep field image is loaded in and the number of white pixels is calculated
img = cv2.imread('fig3.png', cv2.IMREAD_GRAYSCALE)
n_white_pix_image = np.sum(img == 255)

#The number of white pixels from the "Maximum" deep field image is subtracted from the number of white pixels in the deep field image
#resulting in the number of white pixels (the pixels not covered by an ellipse). This number is divided by the total possible number 
#of pixels in the deep field image to find what percentage of the simulation is covered by galaxies. This percentage is displayed as
#the final calculation. 
fraction=(n_white_pix_image-n_white_pix_max)/n_difference
print("Percentage of Wall Formed: "+str(round((1-fraction)*100,2))+'%')
