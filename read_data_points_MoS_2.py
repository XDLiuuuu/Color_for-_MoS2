#!/usr/bin/env python
# coding: utf-8

# In[6]:


##real part for MoS_2.
import numpy as np
import tmm as tmm
import matplotlib.pyplot as plt
from scipy import interpolate
import pylab as pl
import csv 
from functools import reduce

csv_file=csv.reader(open('realpart.csv','r'))

contentx=[]                           ##Construct two lists to store the data.
contenty=[]

for line in csv_file:                ##Import the data
    print (line)
    contentx.append(line[0])
    contenty.append(line[1])

T1= contentx[1:]                      ##Change the list to array, in order to do the interpolation in the next step.
T2= contenty[1:]
T3=[]
T4=[]
for i in range(0,len(T1)):
    T3.append([])
    b=float(T1[i])
    T3[i].append(b)
single_list1 = reduce(lambda x,y: x+y, T3)
print ("The x axis is:\n")
print (single_list1)                         ##Print the data of the x axis.
for i in range(0,len(T2)):
    T4.append([])
    b=float(T2[i])
    T4[i].append(b)
single_list2 = reduce(lambda x,y: x+y, T4)
print ("\nThe y axis is:\n")
print (single_list2)                         ##Print the data of the y axis.


pl.plot(T3,T4,"ro") 


# In[ ]:





# In[7]:


##Imaginary part for MoS_2


csv_file=csv.reader(open('imaginarypart.csv','r'))

contentx=[]                           ##Construct two lists to store the data.
contenty=[]

for line in csv_file:                 ##Import the data
    contentx.append(line[0])
    contenty.append(line[1])


T1= contentx[1:]                      ##Change the list to array, in order to do the interpolation in the next step.
T2= contenty[1:]
T3=[]
T4=[]
for i in range(0,len(T1)):
    T3.append([])
    b=float(T1[i])
    T3[i].append(b)
single_list1 = reduce(lambda x,y: x+y, T3)
print ("The x axis is:\n")
print (single_list1)                         ##Print the data of the x axis.
for i in range(0,len(T2)):
    T4.append([])
    b=float(T2[i])
    T4[i].append(b)
single_list2 = reduce(lambda x,y: x+y, T4)
print ("\nThe y axis is:\n")
print (single_list2)                         ##Print the data of the y axis.


pl.plot(T3,T4,"ro") 


# In[ ]:





# In[ ]:





# In[8]:


##real part of refractive index for Si

csv_file=csv.reader(open('refractive_index_silicon.csv','r'))

contentx=[]                           ##Construct two lists to store the data.
contenty=[]

for line in csv_file:                 ##Import the data
    contentx.append(line[0])
    contenty.append(line[1])


T1= contentx[1:]                      ##Change the list to array, in order to do the interpolation in the next step.
T2= contenty[1:]
T3=[]
T4=[]
for i in range(0,len(T1)):
    T3.append([])
    b=float(T1[i])
    T3[i].append(b)
single_list1 = reduce(lambda x,y: x+y, T3)
print ("The x axis is:\n")
print (single_list1)                         ##Print the data of the x axis.
for i in range(0,len(T2)):
    T4.append([])
    b=float(T2[i])
    T4[i].append(b)
single_list2 = reduce(lambda x,y: x+y, T4)
print ("\nThe y axis is:\n")
print (single_list2)                         ##Print the data of the y axis.


pl.plot(T3,T4,"ro") 


# In[9]:


##extinctive part of refractive index for Si


csv_file=csv.reader(open('extinction_coefficient _silicon.csv','r'))

contentx=[]                           ##Construct two lists to store the data.
contenty=[]

for line in csv_file:                 ##Import the data
    contentx.append(line[0])
    contenty.append(line[1])


T1= contentx[1:]                      ##Change the list to array, in order to do the interpolation in the next step.
T2= contenty[1:]
T3=[]
T4=[]
for i in range(0,len(T1)):
    T3.append([])
    b=float(T1[i])
    T3[i].append(b)
single_list1 = reduce(lambda x,y: x+y, T3)

print ("The x axis is:\n")
print (single_list1)                         ##Print the data of the x axis.
for i in range(0,len(T2)):
    T4.append([])
    b=float(T2[i])
    T4[i].append(b)
single_list2 = reduce(lambda x,y: x+y, T4)
print ("\nThe y axis is:\n")
print (single_list2)                         ##Print the data of the y axis.


pl.plot(T3,T4,"ro") 


# In[ ]:




