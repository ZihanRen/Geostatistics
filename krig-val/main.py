# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:32:40 2019

@author: zur74 - ZihanRen - Pennsylvania State University - Energy and Mineral Engineering Department
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv


class Axisrotate():

    '''
    Object concerning with axis rotation and transformed lag computation
    Prerequisite: computed x and y meshgrid
    '''
    
    def __init__(self,a_max,a_min,azimuth):
        
        self.a_max = a_max
        self.a_min = a_min
        self.azimuth = azimuth
        
    def transform(self,hx,hy):
        '''
        INPUT:
            the difference between estimated value and known data points hx'' - h
        Output the transformed h in vmodel
        '''
        
        rotate = np.array([[math.sin(self.azimuth),math.cos(self.azimuth)],[-math.cos(self.azimuth),math.sin(self.azimuth)]])
        scale = np.array([[self.a_min/self.a_max,0],[0,1]])
        transformed = np.matmul(rotate,np.array([hx,hy]))
        new_xy = np.matmul(scale,transformed)
        new_h = (new_xy[0]**2 + new_xy[1]**2)**0.5
        
        return new_h
    
    def sph_model(self,h):
        '''
        INPUT:
            transformed value of h
        OUTPUT:
            variogram value or covariance value
        '''
        if h <= self.a_min:
        
            gamma_value = 1.5*(h/self.a_min) - 0.5*( (h/self.a_min)**3 )
            cov_value = 1 - gamma_value
            
        else:
            gamma_value = 1
            cov_value = 0
            
        return gamma_value,cov_value

def locmap(df,xcol,ycol,vcol,xmin,xmax,ymin,ymax,vmin,vmax,title,xlabel,ylabel,vlabel,cmap):
    plt.figure(figsize=(9,7))    
    im = plt.scatter(df[xcol],df[ycol],s=50, c=df[vcol], marker=None, cmap=cmap, norm=None, vmin=vmin, vmax=vmax, alpha=1, linewidths=0.8, verts=None, edgecolors="black")
    plt.title(title)
    plt.xlim(xmin-200,xmax+200)
    plt.ylim(ymin-200,ymax+200)    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar(im, orientation = 'vertical',ticks=np.linspace(vmin,vmax,10))
    cbar.set_label(vlabel, rotation=270, labelpad=20)
    plt.show()
    return im

# some visulization
coal = pd.read_csv("coal.csv")
train = pd.read_csv("train.txt",delim_whitespace = True,header=None)
validate = pd.read_csv("validate.txt",delim_whitespace = True,header=None)

locmap(train,0,1,2,np.min(coal['x']),np.max(coal['x']),np.min(coal['y']),
   np.max(coal['y']),np.min(coal['com-calorific']),np.max(coal['com-calorific'])
   ,'calorific value','x','y','calorific', 'Reds')
locmap(validate,0,1,2,np.min(coal['x']),np.max(coal['x']),np.min(coal['y']),
   np.max(coal['y']),np.min(coal['com-calorific']),np.max(coal['com-calorific'])
   ,'calorific value','x','y','calorific', 'Reds')
            
# grid statistics          
x_unique = np.sort(coal['x'].unique())
y_unique = np.sort(coal['y'].unique())
z_unique = np.sort(coal['z'].unique()) # elevation is not equally distributed
x_number = len(x_unique)
y_number = len(y_unique)
x_axis,y_axis = np.meshgrid(x_unique,y_unique)
xy_data_full = np.ones((len(x_unique)*len(x_unique),2))
for i in range(len(x_unique)):   # turn the meshgrid format into column format
    for j in range(len(y_unique)):
        xy_data_full[j+11*i,0] = x_unique[i]
        xy_data_full[j+11*i,1] = y_unique[j]
xy_train = train[[0,1,2]].values           
print('I am going to do both simple kriging and ordinary kriging')   

# let's first try ordinary kriging and see the validation error - whole data points doing kriging

mean = 12967
axis = Axisrotate(800,650,math.radians(135))
variance = 1

def cov_value(dx1,dy1,dx2,dy2):
    
    dx = dx2 - dx1
    dy = dy2 - dy1
    new_h = axis.transform(dx,dy)
    cov_value = axis.sph_model(new_h)[1]
    
    return cov_value

def matrix_ab_ordinary(train_data):
    train_covab = np.zeros((len(train_data)+1,len(train_data)+1))
    for i in range(len(train_data)):
        for j in range(len(train_data)):
            x1 = train_data[i,0]
            y1 = train_data[i,1]
            x2 = train_data[j,0]
            y2 = train_data[j,1]
            train_covab[i,j] = cov_value(x1,y1,x2,y2)
            
    train_covab[-1,:] = 1
    train_covab[:,-1] = 1
    train_covab[-1,-1] = 0
            
    return train_covab

def matrix_ab_simple(train_data):
    train_covab = np.zeros((len(train_data),len(train_data)))
    for i in range(len(train_data)):
        for j in range(len(train_data)):
            x1 = train_data[i,0]
            y1 = train_data[i,1]
            x2 = train_data[j,0]
            y2 = train_data[j,1]
            train_covab[i,j] = cov_value(x1,y1,x2,y2)
                        
    return train_covab

def matrix_a0_ordinary(x,y,train_data):
    train_cova0 = np.zeros(len(train_data)+1)
    for i in range(len(train_data)):
        x2 = train_data[i,0]
        y2 = train_data[i,1]
        train_cova0[i] = cov_value(x,y,x2,y2)
        
    train_cova0[-1] = 1
    return train_cova0

def matrix_a0_simple(x,y,train_data):
    train_cova0 = np.zeros(len(train_data))
    for i in range(len(train_data)):
        x2 = train_data[i,0]
        y2 = train_data[i,1]
        train_cova0[i] = cov_value(x,y,x2,y2)
        
    return train_cova0

def matrix_inverse_dist(x,y,train_data,power):
    dist_matrix_weight = np.zeros(len(train_data))
    
    for i in range(len(train_data)):
        x2 = train_data[i,0]
        y2 = train_data[i,1]
        if abs(x-x2) + abs(y-y2) < 1e02:
            dist_matrix_weight[i] = (1/1e-09)
        else:
            dist_matrix_weight[i] = 1/( ( abs((x-x2))**power + abs((y-y2))**power )**(1/power) )
    
    weight_matrix = dist_matrix_weight/np.sum(dist_matrix_weight)
    
    return weight_matrix
        
def rmse(validate,prediction,prediction_title,title):
    
    '''
    INPUT:
        validate dataframe
        prediction result of matrix
    OUTPUT:
        RMSE value and matrix and graph
    '''
    
    validate_calor = validate.loc[:,[0,1,2]]
    new = pd.DataFrame(prediction)
    merge = pd.merge(new,validate_calor,on=[0,1]).values  
    rmse_matrix = merge[:,2] - merge[:,3]
    rmse_average = np.sum((rmse_matrix**2)**0.5)/len(rmse_matrix)
    # visulization
    x_axis = [x for x in range(len(merge))]
    plt.plot(x_axis,merge[:,2],c='blue',label=prediction_title)
    plt.legend()
    plt.plot(x_axis,merge[:,3],c='red',label='True value in Validation dataset')
    plt.legend()
    plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
    plt.xlabel('index')
    plt.ylabel('calorific value')
    plt.title(title)
    plt.show()    
    
    
    return rmse_average,rmse_matrix

def rmse_value(validate,prediction):
    
    '''
    INPUT:
        validate dataframe
        prediction result of matrix
    OUTPUT:
        RMSE value and matrix and graph
    '''
    
    validate_calor = validate.loc[:,[0,1,2]]
    new = pd.DataFrame(prediction)
    merge = pd.merge(new,validate_calor,on=[0,1]).values  
    rmse_matrix = merge[:,2] - merge[:,3]
    rmse_average = np.sum((rmse_matrix**2)**0.5)/len(rmse_matrix)

    return rmse_average

# begin simulation: 1st - inverse distance method
xy_data_full_dist = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))
for i in range(len(xy_data_full_dist)):
    weight_matrix_dist = matrix_inverse_dist(xy_data_full[i,0],xy_data_full[i,1],xy_train,2)
    xy_data_full_dist[i,2] = np.dot(weight_matrix_dist,xy_train[:,2])

rmse_dist = rmse(validate,xy_data_full_dist,'distance inversion value','Distance inversion simulation versus true value')[0]
print('632')

################################## parameter tuning in inverse distance function ######################### not meaningful #############
#xy_data_full_dist = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))
#power_list = [1,2,3]
#rmse_group = []
#power_sum = []
#for power_loop in power_list:
#    for i in range(len(xy_data_full_dist)):
#        weight_matrix_dist = matrix_inverse_dist(xy_data_full[i,0],xy_data_full[i,1],xy_train,power_loop)
#        xy_data_full_dist[i,2] = np.dot(weight_matrix_dist,xy_train[:,2])
#        
#    rmse_dist = rmse_value(validate,xy_data_full_dist)
#    rmse_group.append(rmse_dist)
#    power_sum.append(power_loop)
#    
#plt.figure()
#plt.scatter(power_sum,rmse_group,c='Green',s = 50, marker='*')
#plt.xlabel('looping power - com(calorific)')
#plt.ylabel('Root mean square error')
#plt.title('power selection vs RMSE')
#########################################################################################################################################

# begin simulation - ordinary kriging
ab_matrix = matrix_ab_ordinary(xy_train)
inverse_ab = inv(ab_matrix)
zero_matrix = np.zeros(len(ab_matrix))
zero_matrix[-1] = 1
mean_matrix = np.matmul(inverse_ab,zero_matrix)
mean_predict = np.dot(xy_train[:,2],mean_matrix[:-1])

xy_data_full_ord = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))
xy_data_full_ord_var = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))



for i in range(len(xy_data_full_ord)):
    a0_matrix = matrix_a0_ordinary(xy_data_full[i,0],xy_data_full[i,1],xy_train)
    weight_matrix = np.matmul(inverse_ab,a0_matrix)
    xy_data_full_ord[i,2] = np.dot(weight_matrix[:-1],xy_train[:,2])
    krig_var = np.multiply(weight_matrix,a0_matrix)
    xy_data_full_ord_var[i,2] = variance - np.sum(krig_var)
    
    if abs(xy_data_full_ord_var[i,2]) < 1e-04:
        xy_data_full_ord_var[i,2] = 0
rmse_ordinary = rmse(validate,xy_data_full_ord,'Calorific value','Ordinary Kriging simulation versus true value')[0]

# simple kriging
## the result is slightly changing
rmse_summary = []
mean_sum = []
mean_list = [x+mean-500 for x in range(0,2000,100)]

ab_matrixs = matrix_ab_simple(xy_train)
inverse_abs = inv(ab_matrixs)
xy_data_full_sim = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))

for mean_loop in mean_list: 
    for i in range(len(xy_data_full_ord)):
        a0_matrixs = matrix_a0_simple(xy_data_full[i,0],xy_data_full[i,1],xy_train)
        weight_matrixs = np.matmul(inverse_abs,a0_matrixs)
        xy_data_full_sim[i,2] = ( 1-np.sum(weight_matrixs) )*mean_loop + np.dot(weight_matrixs,xy_train[:,2])

    rmse_simple = rmse_value(validate,xy_data_full_sim)
    rmse_summary.append(rmse_simple)
    mean_sum.append(mean_loop)
# plot the RMSE vs mean selection
plt.scatter(mean_sum,rmse_summary,c='Green',s = 50, marker='*')
plt.xlabel('looping mean - com(calorific)')
plt.ylabel('Root mean square error')
plt.title('mean selection vs RMSE')

print(str(mean_sum[5]))
print('The simple kriging with mean equal to ' + str(mean_sum[5]) + ' have RMSE ' +
      str(rmse_summary[5]))

res_list = [i for i, value in enumerate(rmse_summary) if value == min(rmse_summary)]
print('The updated mean is ' + str(mean_sum[res_list[0]]) )

# formal simulation
ab_matrixs = matrix_ab_simple(xy_train)
inverse_abs = inv(ab_matrixs)
xy_data_full_sim_final = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))
xy_data_full_sim_var = np.hstack(( xy_data_full,np.zeros(len(xy_data_full)).reshape(-1,1) ))


for i in range(len(xy_data_full_sim_final)):
    a0_matrixs = matrix_a0_simple(xy_data_full[i,0],xy_data_full[i,1],xy_train)
    weight_matrixs = np.matmul(inverse_abs,a0_matrixs)
    xy_data_full_sim_final[i,2] = ( 1-np.sum(weight_matrixs) )*13567 + np.dot(
            weight_matrixs,xy_train[:,2])
    
    krig_var = np.multiply(a0_matrixs,weight_matrixs)
    xy_data_full_sim_var[i,2] = variance - np.sum(krig_var)
    
    if abs(xy_data_full_sim_var[i,2]) < 1e-04:
        xy_data_full_sim_var[i,2] = 0

rmse_sim_final = rmse(validate,xy_data_full_sim_final)[0]
print(rmse_sim_final)



















         
            
            
            
            
            
            
            
            
            
            
