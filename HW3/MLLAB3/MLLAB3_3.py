#!/usr/bin/env python
# coding: utf-8

# # 讀檔

# In[39]:


import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
from matplotlib import pyplot as plt
import cv2
import random
img = cv2.imread('HW3.jpg')
height=90
width=180
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
img_rgb = img[:,:,::-1]
print( img_rgb.shape )
plt.imshow(img_rgb)
plt.show()


# # K-means 參數 函數

# In[40]:


def cal_dis(x1,y1,z1,x2,y2,z2):
    return ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
f = open('A.txt', 'w', encoding = 'UTF-8')
k=5
kmeans_iteration=10
uk=np.ones((k,3))
rnk=np.zeros((height,width))
for i in range(k):
    for j in range(3):
        uk[i][j]=random.randint(0,256)            


# # K-means

# In[41]:


for iteration in range(kmeans_iteration):
    print("iteration:",iteration,", E step")
    for i in range(height):
        for j in range(width):
            min_uk=-1
            min_dis=1000000000
            for m in range(k):
                distance=cal_dis(img[i][j][0],img[i][j][1],img[i][j][2], uk[m][0], uk[m][1], uk[m][2])
                if distance<min_dis:
                    min_dis=distance
                    min_uk=m
            rnk[i][j]=min_uk
           ## print("rnk[",i,"][",j,"]=",min_uk)
    print("iteration:",iteration,", M step")
    for i in range(k):
        counter0=counter1=counter2=num=0
        for m in range(height):
            for n in range(width):
                if rnk[m][n]==i:
                    counter0=counter0+img[m][n][0]
                    counter1=counter1+img[m][n][1]
                    counter2=counter2+img[m][n][2]
                    num=num+1
        if num==0:
            print("u",i,": ")
            uk[i][0]=random.randint(0,256)    
            uk[i][1]=random.randint(0,256)    
            uk[i][2]=random.randint(0,256)    
        if num!=0:
            print("u",i,": "," ",counter0/num," ",counter1/num," ",counter2/num)
            uk[i][0]=counter0/num
            uk[i][1]=counter1/num
            uk[i][2]=counter2/num
            
    img_rgbtemp=np.zeros((height,width,3))
    for i in range(height):
        for j in range(width):
            group=int(rnk[i][j])
            img_rgbtemp[i][j][0]=int ( uk[ group  ][0] )
            img_rgbtemp[i][j][1]=int ( uk[ group  ][1] )
            img_rgbtemp[i][j][2]=int ( uk[ group  ][2] )
    img_rgbtemp2 = img_rgbtemp.astype(np.int32)
    img_rgb  = img_rgbtemp2[:,:,::-1]
    img_rgb2 = img[:,:,::-1]
    plt.imshow(img_rgb)
    plt.show()


# # GMM

# In[42]:


def gaussian(mean,var,x,D):
    leftterm_Denominator=( ((2*math.pi)**D)*np.linalg.det(var) )**0.5
    rightterm_power=-0.5*(  ((x-mean).T).dot(  np.linalg.pinv(var)  ).dot((x-mean))  )
    #print(leftterm_Denominator)
    #print(rightterm_power)
    return (1.0/leftterm_Denominator*np.exp(rightterm_power) )

D=3
GMM_iteration=100

responsibility=np.zeros((height,width,k))
variance=np.zeros((k,D,D))
for i in range(k):
    variance[i]=128*np.eye(D, dtype = 'int')
latent_var=np.zeros((k))
for i in range(height):
    for j in range(width):
        latent_var  [int(rnk[i][j]) ] =latent_var[ int(rnk[i][j]) ]+1
sum_temp=0
sum_temp2=0
for m in range(k):
    sum_temp=sum_temp+latent_var[m]
#print(sum_temp)
for m in range(k):
    latent_var[m]=latent_var[m]/sum_temp
    sum_temp2=sum_temp2+latent_var[m]
    #print(latent_var[m],sum_temp2)
        


# In[43]:


log_likelyhood=np.zeros((100))
for iteraion in range(GMM_iteration):
    print("第",iteraion,"次 E_step:update resposibility")
    for i in range(height):
        #print("process column:",i)
        for j in range(width):
            res_denomi=0
            for m in range(k):
                #print("debug:",uk[m]," ",img[i][j])
                #print("mode1:",latent_var[m]," ",gaussian(uk[m],variance[m],img[i][j],3))
                res_denomi=res_denomi+latent_var[m]*  gaussian(uk[m],variance[m],img[i][j],3)
            for m in range(k):
                #print("mode2,",m," ",latent_var[m]," ", gaussian(uk[m],variance[m],img[i][j],3)," ",res_denomi)
                responsibility[i][j][m]=latent_var[m]*  gaussian(uk[m],variance[m],img[i][j],3) / res_denomi
                #if((i+1)%20==0):
                    #print(i+1,"responsibility,",i,j,m," ", responsibility[i][j][m])
                
    print("第",iteraion,"次 M_step")
    #print("                        update NK")
    NK=np.zeros((k)) 
    for m in range(k):
        for i in range(height):
            for j in range(width):
                  NK[m]= NK[m]+responsibility[i][j][m]
    #for m in range(k):
          #print( "NK",m,":",NK[m])
                        
    #print("\n","                        update mean")
    for m in range(k):
        #print("process u ",m)
        uk[m][0]=uk[m][1]=uk[m][2]=0
        for i in range(height):
            for j in range(width):
                uk[m][0]=uk[m][0]+responsibility[i][j][m]*img[i][j][0]
                uk[m][1]=uk[m][1]+responsibility[i][j][m]*img[i][j][1]
                uk[m][2]=uk[m][2]+responsibility[i][j][m]*img[i][j][2]
        if NK[m]!=0:
            uk[m]=uk[m]/NK[m]
        else:
            uk[m][0]=random.randint(0,256)
            uk[m][1]=random.randint(0,256)  
            uk[m][2]=random.randint(0,256)  
                
    #print("                        update var ")
    for m in range(k):
        #print("process var ",m)
        var_temp=np.zeros((3,3))
        for i in range(height):
            for j in range(width):
                var_temp=var_temp+responsibility[i][j][m]*( (img[i][j]-uk[m]).T.dot((img[i][j]-uk[m])) )
        var_temp=var_temp/NK[m]
                
    
    #print("                        update latent")   
    for m in range(k):
        latent_var[m]=NK[m]/(width*height)
        
        
    log_likely=0
    for i in range(height):
        for j in range(width):
            temp_value=0
            for m in range(k):
                temp_value=temp_value+latent_var[m]*gaussian(uk[m],variance[m],img[i][j],3)
            log_likely=log_likely+math.log(temp_value)
    
    log_likelyhood[iteraion]=log_likely
    print("log_likely:",log_likely)
    
img_GMM=np.zeros((height,width,3))
for i in range(height):
    for j in range(width):
        max_class=-1
        max_respon=-1
        for m in range(k):
            if responsibility[i][j][m]>max_respon:
                max_class=m
                max_respon=responsibility[i][j][m]
        img_GMM[i][j]=uk[max_class]
img_GMM = img_GMM.astype(np.int32)
img_GMM  = img_GMM[:,:,::-1]
print( img_GMM.shape )
plt.imshow(img_GMM)
plt.show()






# # show image 

# In[44]:


x_temp=np.linspace(0,99,100)
plt.plot(x_temp,log_likelyhood,  color='red')
plt.xlabel('iteration')
plt.ylabel('log_likelyhood')
plt.title('GMM,k=5')
plt.legend()
plt.show()

