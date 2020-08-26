#!/usr/bin/env python
# coding: utf-8

# In[15]:


# colour_system.py
import numpy as np
from scipy.constants import h, c, k
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pylab as pl
from scipy import interpolate
import tmm as tmm


# In[16]:


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


# In[17]:


##Interpolation for the real part of permitivity for the MoS_2.

##First, let's input the data points we got from the data theif. a1 is the x values while b1 is the y values of
a1 =[1.56399, 1.64344, 1.7235, 1.7839, 1.8344, 1.87473, 1.89572, 1.94215, 2.0074, 2.03881, 2.08437, 2.15772, 2.23449, 2.31419, 2.39743, 2.47407, 2.55402, 2.63384, 2.71634, 2.78318, 2.8285, 2.85292, 2.87391, 2.89845, 2.92287, 2.95426, 2.99958]
b1 =[21.465, 22.351, 23.953, 26.549, 29.571, 28.727, 25.302, 23.603, 23.913, 20.635, 17.934, 19.676, 21.42, 22.593, 23.91, 25.511, 26.97, 28.286, 28.744, 26.909, 23.922, 20.499, 17.074, 13.794, 10.371, 7.093, 4.106]

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


# In[18]:


##Interpolation for the imaginary part of the permitivity for the MoS_2.
##The procedurte is similar as before.

a2 =[1.54664, 1.62841, 1.71022, 1.78209, 1.82026, 1.83803, 1.85921, 1.90321, 1.95771, 1.99585, 2.03669, 2.07349, 2.13087, 2.21599, 2.29449, 2.37645, 2.4517, 2.52028, 2.57871, 2.62704, 2.67199, 2.71014, 2.7449, 2.77967, 2.81444, 2.85933, 2.92721, 2.97085, 3.01448]

b2 =[0.894, 1.008, 1.265, 2.812, 6.087, 9.513, 12.938, 11.635, 11.615, 14.747, 14.59, 11.43, 8.978, 8.805, 9.492, 10.463, 11.866, 13.986, 16.397, 19.382, 22.512, 25.645, 28.921, 32.198, 35.475, 38.319, 37.15, 34.131, 31.112]
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


# In[19]:


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


# In[20]:


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


# In[21]:


##In this section, we will combine the material and the substrate together, to form a 2D device. And calculate the 
##reflectance of the whole system.
#This function will return a function f5, which is the function of the reflectance, by given the thickness of SiO2.


def reflec_absorp_func(d_list,dSiO2,th_0, pol, lam_max, lam_min):
    
    """d_list is the thickness of the material, dSiO2 is the thickness of the SiO2 layer, th_0 is the incident angle,
    pol is the polarization of incident light, lam_max and lam_min are the ranges of the incident light. """
    
    d_list.insert(0, np.inf)                       ###Insert infinity at zero position.
    d_list = d_list + [dSiO2,np.inf]
    epsilon_SiO2 = 1.479                           ##This is the permitivity of the silicon. It's a constant!
    
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
        n_list = n_list + [(epsilon_SiO2*mu)**(1/2), n_si]
        
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

    
    #plt.figure()
    #plt.plot(E, absorption)
    #plt.xlabel('E(in eV)')
    #plt.ylabel('absorption')
    #plt.title('relationship between absorption and photon energy(%s'%(pol)+' polarized)')
    #plt.show() 
    return f5


# In[22]:


#This function will return a value re, which is the reflectance at wavelength lam, with thickness of SiO2 as d.

def reflectance(lam,d):
    x = 1240/lam                                           ##Convert the wavelength to the energy.
    r = reflec_absorp_func([0.615],d, 0, 'p', 330, 800)    ##The range of the wavelength must include the range of the visible light.
    re = r(x)
    return re


# In[23]:


#This function will give back the radiation power, at wavelength lam, at temperature T, with the thickness of SiO2 as d.
def planck(lam, T, d):

    re = reflectance(lam,d)
    lam_m = lam / 1.e9
    fac = h*c/lam_m/k/T
    B = 2*h*c**2/lam_m**5 / (np.exp(fac) - 1)*re*re  ##Since we are calculating the reflected power, we need a square here. 
    return B


# In[24]:


##The color of MoS_2 with the substrate at different thicknesses.(from 0 to 270nm)

fig, ax = plt.subplots()

# The grid of visible wavelengths corresponding to the grid of colour-matching
# functions used by the ColourSystem instance.
lam = np.arange(380., 781., 5)




for i in range(28):
   
    T =  2500              ##T is the temperature of the black body.
    
    d= 10*i                ##d is ranging from 0 nm to 270 nm.
    # Calculate the black body spectrum and the HTML hex RGB colour string
    # it looks like
    spec = planck(lam, T, d)
    html_rgb = cs.spec_to_rgb(spec, out_fmt='html')

    # Place and label a circle with the colour of a black body at temperature T
    x, y = i % 7, -(i // 7)
    circle = Circle(xy=(x, y*1.2), radius=0.35, fc=html_rgb)
    ax.add_patch(circle)
    ax.annotate('{:4d} nm'.format(d), xy=(x, y*1.2-0.5), va='center',
                ha='center', color=html_rgb)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-0.5,6.5)
ax.set_ylim(-4.35, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
# Make sure our circles are circular!
ax.set_aspect("equal")
plt.show()







##The color of MoS_2 with the substrate at different thicknesses.(from 280 nm to 560 nm)

fig, ax = plt.subplots()

# The grid of visible wavelengths corresponding to the grid of colour-matching
# functions used by the ColourSystem instance.
lam = np.arange(380., 781., 5)

for i in range(28):
   
    T =  2500
    
    d= 10*i+280
    # Calculate the black body spectrum and the HTML hex RGB colour string
    # it looks like
    spec = planck(lam, T, d)
    html_rgb = cs.spec_to_rgb(spec, out_fmt='html')

    # Place and label a circle with the colour of a black body at temperature T
    x, y = i % 7, -(i // 7)
    circle = Circle(xy=(x, y*1.2), radius=0.35, fc=html_rgb)
    ax.add_patch(circle)
    ax.annotate('{:4d} nm'.format(d), xy=(x, y*1.2-0.5), va='center',
                ha='center', color=html_rgb)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-0.5,6.5)
ax.set_ylim(-4.35, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
# Make sure our circles are circular!
ax.set_aspect("equal")
plt.show()







##The color of MoS_2 with the substrate at different thicknesses.(from 560 nm to 830 nm)

fig, ax = plt.subplots()

# The grid of visible wavelengths corresponding to the grid of colour-matching
# functions used by the ColourSystem instance.
lam = np.arange(380., 781., 5)

for i in range(28):
   
    T =  2500
    
    d= 10*i+560
    # Calculate the black body spectrum and the HTML hex RGB colour string
    # it looks like
    spec = planck(lam, T, d)
    html_rgb = cs.spec_to_rgb(spec, out_fmt='html')

    # Place and label a circle with the colour of a black body at temperature T
    x, y = i % 7, -(i // 7)
    circle = Circle(xy=(x, y*1.2), radius=0.35, fc=html_rgb)
    ax.add_patch(circle)
    ax.annotate('{:4d} nm'.format(d), xy=(x, y*1.2-0.5), va='center',
                ha='center', color=html_rgb)

# Set the limits and background colour; remove the ticks
ax.set_xlim(-0.5,6.5)
ax.set_ylim(-4.35, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('k')
# Make sure our circles are circular!
ax.set_aspect("equal")
plt.show()

