#!/usr/bin/env python
# coding: utf-8

# In[111]:


# colour_system.py
import numpy as np
from scipy.constants import h, c, k
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pylab as pl
from scipy import interpolate
import tmm as tmm


# In[112]:


##Some functions we will use later.
##These functions are from the website.

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

class ColourSystem:
    """A class representing a colour system.

    A colour system defined by the CIE x, y and z=1-x-y coordinates of
    its three primary illuminants and its "white point".

    TODO: Implement gamma correction

    """

    # The CIE colour matching function for 380 - 780 nm in 5 nm intervals
    cmf = np.loadtxt('cie-cmf.txt', usecols=(1,2,3))

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.

        """

        # Chromaticities
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        if not np.all(rgb==0):
            # Normalize the rgb vector
            rgb /= np.max(rgb)

        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """Convert a spectrum to an xyz point.

        The spectrum must be on the same grid of points as the colour-matching
        function, self.cmf: 380-780 nm in 5 nm steps.

        """

        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        if den == 0.:
            return XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """Convert a spectrum to an rgb value."""

        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)

illuminant_D65 = xyz_from_xy(0.3127, 0.3291)
cs_hdtv = ColourSystem(red=xyz_from_xy(0.67, 0.33),
                       green=xyz_from_xy(0.21, 0.71),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)

cs_smpte = ColourSystem(red=xyz_from_xy(0.63, 0.34),
                        green=xyz_from_xy(0.31, 0.595),
                        blue=xyz_from_xy(0.155, 0.070),
                        white=illuminant_D65)

cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)


cs = cs_hdtv


# In[113]:


##Interpolation for the real part of permitivity for the MoS_2.

##First, let's input the data points we got from the data theif. a1 is the x values while b1 is the y values of
a1 =[1.54743, 1.59013, 1.62578, 1.66309, 1.70413, 1.73391, 1.76204, 1.78755, 1.80973, 1.83047, 1.85061, 1.86481, 1.87445, 1.89485, 1.92214, 1.96137, 1.98686, 2.02093, 2.04077, 2.06862, 2.10944, 2.14015, 2.1824, 2.2151, 2.25966, 2.29344, 2.33476, 2.3752, 2.41416, 2.45014, 2.49356, 2.52849, 2.56867, 2.60684, 2.65665, 2.68859, 2.72961, 2.75672, 2.80441, 2.83166, 2.85551, 2.88276, 2.91001, 2.94407, 2.99176]
b1 =[21.515, 21.769, 22.344, 22.914, 23.889, 24.871, 26.443, 27.824, 29.429, 30.331, 28.556, 26.535, 25.115, 22.831, 23.382, 24.518, 23.644, 20.343, 18.274, 17.609, 18.519, 19.299, 20.295, 20.988, 21.348, 22.104, 22.582, 23.361, 23.995, 24.908, 25.588, 26.309, 26.912, 27.568, 27.871, 27.968, 27.124, 26.084, 23.064, 19.622, 16.181, 12.882, 9.44, 6.138, 3.118]

##Plot the data points.
pl.plot(a1,b1,"ro")

for kind in ["cubic"]:                                       #The interpolation method is "Cubic method".
 
    f1=interpolate.interp1d(a1,b1,fill_value="extrapolate")  #We name the function of real part for MoS2 as f1.
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    xnew=a1
    ynew=f1(a1)
    pl.plot(xnew,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('E(in eV)')
plt.title('real part of permitivity for MoS2')

##This is the plot of the interpolation.
pl.show()


# In[114]:


##Interpolation for the imaginary part of the permitivity for the MoS_2.
##The procedurte is similar as before.
a2 =[1.54665, 1.588, 1.62809, 1.66493, 1.70959, 1.75495, 1.78146, 1.80499, 1.82014, 1.83011, 1.83853, 1.84654, 1.86032, 1.869, 1.89185, 1.90386, 1.91474, 1.9294, 1.95811, 1.9776, 1.99673, 2.01711, 2.03736, 2.05652, 2.07334, 2.09582, 2.12997, 2.17017, 2.21468, 2.29297, 2.37477, 2.44997, 2.51869, 2.57735, 2.62608, 2.67149, 2.7101, 2.72915, 2.74539, 2.78068, 2.81597, 2.8358, 2.86125, 2.88565, 2.91518, 2.92859, 2.95268, 2.97141, 2.98992, 3.01423]

b2 =[0.951, 0.911, 1.033, 0.945, 1.257, 1.589, 2.751, 3.678, 5.955, 7.465, 9.313, 10.905, 12.669, 13.456, 11.935, 11.374, 10.503, 9.784, 11.336, 12.582, 14.399, 15.032, 14.23, 12.261, 11.114, 9.224, 8.685, 8.198, 8.485, 9.132, 10.057, 11.409, 13.467, 15.814, 18.73, 21.788, 24.851, 26.209, 28.057, 31.263, 34.469, 35.78, 37.247, 37.426, 36.52, 36.075, 34.281, 33.095, 31.423, 30.114]
pl.plot(a2,b2,"ro")

for kind in ["cubic"]:#The interpolation method is "Cubic method".
    
    f2=interpolate.interp1d(a2,b2,fill_value="extrapolate")      #Define a function f2, for the imaginary part.
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    xnew=a2
    ynew=f2(a2)
    pl.plot(xnew,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('E(in eV)')
plt.title('imaginary part of permitivity for MoS2')
pl.show()


# In[115]:


##Real part of the refractive index for the si

a3 =[208.219, 218.493, 227.397, 234.247, 247.945, 261.644, 267.808, 271.918, 274.658, 276.712, 278.767, 280.822, 282.192, 284.247, 285.616, 286.986, 288.356, 289.726, 291.096, 293.151, 294.521, 297.26, 306.164, 322.603, 335.616, 345.89, 351.37, 355.479, 358.219, 360.274, 363.014, 365.068, 373.973, 379.452, 382.877, 386.301, 390.411, 395.89, 400.685, 407.534, 415.068, 423.973, 434.932, 445.89, 458.904, 472.603, 487.671, 502.74, 519.178, 534.932, 551.37, 567.808, 583.562, 600.0, 616.438, 632.877, 649.315, 666.438, 682.877, 699.315, 716.438, 732.877, 750.0, 766.438, 782.877, 800.0, 816.438]

b3 =[1.0, 1.15323, 1.33065, 1.51613, 1.56452, 1.67742, 1.8629, 2.05645, 2.25806, 2.45161, 2.65323, 2.85484, 3.05645, 3.25806, 3.45968, 3.66129, 3.85484, 4.05645, 4.25806, 4.45968, 4.66129, 4.8629, 5.01613, 5.03226, 5.15323, 5.32258, 5.50806, 5.70968, 5.90323, 6.10484, 6.30645, 6.5, 6.67742, 6.52419, 6.32258, 6.12903, 5.93548, 5.74194, 5.54839, 5.3629, 5.18548, 5.00806, 4.85484, 4.70968, 4.57258, 4.45968, 4.35484, 4.27419, 4.19355, 4.12903, 4.07258, 4.02419, 3.97581, 3.93548, 3.90323, 3.87097, 3.83871, 3.81452, 3.79839, 3.77419, 3.75806, 3.74194, 3.72581, 3.70968, 3.70161, 3.68548, 3.66935]

pl.plot(a3,b3,"ro")

for kind in ["cubic"]:#The interpolation method is "Cubic method".
    
    f3=interpolate.interp1d(a3,b3,fill_value="extrapolate")  #define a function named f3, which is the real part of the refractive index for the Si.
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
    xnew=a3
    ynew=f3(a3)
    pl.plot(xnew,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('wavelength (in nm)')
plt.title('Real part of refractive index for Si')
pl.show()


# In[116]:


##Extinctive part of the refractive index for the si (k)

a4 =[206.944, 215.278, 222.917, 231.25, 243.75, 250.0, 254.167, 258.333, 261.806, 264.583, 267.361, 270.139, 272.222, 275.0, 277.778, 281.944, 289.583, 291.667, 293.056, 295.139, 296.528, 298.611, 300.0, 302.083, 304.167, 307.639, 311.111, 315.278, 321.528, 329.861, 340.972, 356.25, 361.806, 365.278, 367.361, 368.75, 370.139, 371.528, 373.611, 375.0, 376.389, 377.778, 380.556, 383.333, 386.111, 390.278, 397.917, 408.333, 422.917, 438.889, 455.556, 472.222, 488.889, 505.556, 522.917, 540.278, 556.944, 574.306, 591.667, 609.028, 626.389, 643.75, 661.111, 677.778, 695.139, 712.5, 729.861, 747.222, 764.583, 781.944, 799.306, 816.667]


b4 =[2.90411, 3.04795, 3.20548, 3.35616, 3.42466, 3.58219, 3.74658, 3.91781, 4.08219, 4.25342, 4.41781, 4.58904, 4.76027, 4.93151, 5.09589, 5.26712, 5.31507, 5.15068, 4.97945, 4.80822, 4.63699, 4.46575, 4.29452, 4.12329, 3.9589, 3.78767, 3.61644, 3.45205, 3.29452, 3.14384, 3.0137, 2.9863, 2.82877, 2.65753, 2.49315, 2.32192, 2.15068, 1.97945, 1.80822, 1.63699, 1.46575, 1.29452, 1.13014, 0.9589, 0.78767, 0.62329, 0.46575, 0.33562, 0.24658, 0.19178, 0.15753, 0.12329, 0.09589, 0.08219, 0.07534, 0.06164, 0.05479, 0.04795, 0.04795, 0.0411, 0.0411, 0.03425, 0.03425, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.0274, 0.02055]

pl.plot(a4,b4,"ro")

for kind in ["cubic"]:#The interpolation method is "Cubic method".
    
    f4=interpolate.interp1d(a4,b4,fill_value="extrapolate")
    # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)

    ynew=f4(a4)
    pl.plot(a4,ynew,label=str(kind))
pl.legend(loc="lower right")
plt.xlabel('wavelength (in nm)')
plt.title('imaginary part of refractive index for Si')
pl.show()


# In[117]:


##In this section, we will combine the material and the substrate together, to form a 2D device. And calculate the 
##reflectance of the whole system.
#This function will return a function f5, which is the function of the reflectance, by given the thickness of SiO2.


def reflec_absorp_func(d_list,dSiO2,th_0, pol, lam_max, lam_min):
    
    """d_list is the thickness of the material, dSiO2 is the thickness of the SiO2 layer, th_0 is the incident angle,
    pol is the polarization of incident light, lam_max and lam_min are the ranges of the incident light. """
    
    d_list.insert(0, np.inf)                       ###Insert infinity at zero position.
    d_list = d_list + [dSiO2,np.inf]
    n_SiO2 = 1.479                           ##This is the permitivity of the silicon. It's a constant!
    
    mu=1                                           ### Suppose the relative permeability is ONE for all layers
    
  
    lam_range = np.linspace(lam_min,lam_max,num=500)
    

    reflectance = []
    for lam_vac in lam_range:
        
        energy = 1240/lam_vac
        
        dielectric1 =  f1(energy)                  ###Calculate the real part of permitivity.
        dielectric2 =  f2(energy)                  ###Calculate the imaginary part of permitivity.


        dielectric = [complex(dielectric1,dielectric2)]  ##This is the permitivity in the specific energy.
        n_si = complex(f3(lam_vac),f4(lam_vac))          ##input the refractive index for silicon.
        
        n_list = np.array(dielectric*mu)**(1/2)     ### Calculate the refraction index for the material
        n_list = n_list.tolist()
        n_list.insert(0, 1)
        n_list = n_list + [n_SiO2, n_si]
        
        coh_tmm_data = tmm.tmm_core.coh_tmm(pol, n_list, d_list, th_0, lam_vac)
        r = coh_tmm_data['r']
        reflectance.append(tmm.R_from_r(r))
        
    absorption = 1-np.array(reflectance)    
    h = 6.62607004*1e-34
    E = h*3*1e8/(lam_range*1e-9*1.602176634*1e-19) ### in eV
    E = E.tolist()
    

    
    for kind in ["cubic"]:
        f5=interpolate.interp1d(E,reflectance,fill_value="extrapolate")
        
        """f5 is the function of the reflectance for MoS2o"""
        # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
        
        xnew=E
        ynew=f5(E)
        
        
    ##The folowing few lines can draw the plot of the caculated refelctance.
    ##pl.plot(xnew,ynew,label=str(kind))
   
    #plt.xlabel('E(in eV)')
    #plt.ylabel('reflectance')
    #plt.title('relationship between reflectance and photon energy(%s'%(pol)+' polarized)')
    #pl.show()
    return f5


# In[118]:


#This function will return a value re, which is the reflectance at wavelength lam, with thickness of material as d. The thickness of the SiO2 is 300nm.

def reflectance(lam,d):
    x = 1240/lam                                           ##Convert the wavelength to the energy.
    r = reflec_absorp_func([d], 300, 0, 'p', 330, 800)       ##The range of the wavelength must include the range of the visible light.
    re = r(x)
    return re


# In[119]:


#This function will give back the radiation power, at wavelength lam, at temperature T, with the thickness of SiO2 as d.
def planck(lam, T, d):

    re = reflectance(lam,d)
    lam_m = lam / 1.e9
    fac = h*c/lam_m/k/T
    B = (2*h*c**2)/(lam_m**5)/(np.exp(fac) - 1)*(re)  ##Since we are calculating the reflected power, we need a square here. 
    return B


# In[120]:


##The color of MoS_2 with the substrate at different thicknesses.(from 0 to 270nm)

fig, ax = plt.subplots()

# The grid of visible wavelengths corresponding to the grid of colour-matching
# functions used by the ColourSystem instance.
lam = np.arange(380., 781., 5)




for i in range(30):
   
    T =  2500              ##T is the temperature of the black body.
    
    d= i*0.65+0.65  
    # Calculate the black body spectrum and the HTML hex RGB colour string
    # it looks like
    spec = planck(lam, T, d)
    html_rgb = cs.spec_to_rgb(spec, out_fmt='html')

    # Place and label a circle with the colour of a black body at temperature T
    x, y = 2*(i % 30), -(i // 30)
    circle = Circle(xy=(x, y*1.2), radius=0.75, fc=html_rgb)
    ax.add_patch(circle)
    #ax.annotate('{:2f} nm'.format(d), xy=(x, y*1.2-0.5), va='center',
                #ha='center', color=html_rgb)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-2.5,60)
ax.set_ylim(-1.65, 1.7)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
# Make sure our circles are circular!
ax.set_aspect("equal")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




