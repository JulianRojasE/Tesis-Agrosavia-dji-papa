hola
pirobo el N
Hola x2
import cv2
import numpy as np
import os
import math
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter
BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBb
Base_imagenes_Path = "C:/Users/julia/Desktop/9 semestre/Tesis/GFK/Salidita"
AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa
if not os.path.exists(Base_imagenes_Path):
    os.makedirs(Base_imagenes_Path)
    print("No existia la carpeta pero se creo en: ", Base_imagenes_Path)
aaaaaaaaaaaaaaa
puntos = pd.read_csv('Puntos QGIS_FIN.csv')
puntos['new'] = 0
puntos['new2'] = 0

NumeroFinal=[]
tipo=[]
areas_veg = []
areas = []

eps = 10e-6
eps *= 255*255

Esp = 21
Pad = 25
Alto = 294 + 2*Pad
Ancho = Esp*2+1

fp = r'V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif'
Orto = cv2.imread('V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif', cv2.IMREAD_UNCHANGED)

img = rasterio.open(fp)
for i in range(puntos.shape[0]):
    lal = img.index(puntos['Lat'].iloc[i],puntos['Lon'].iloc[i], z=None, precision=None)
    puntos['new'].iloc[i]= lal[0]
    puntos['new2'].iloc[i]= lal[1]
    
Lista = puntos.to_numpy()

Distanciax = []
Distanciay = []
    
for i in range(0,Lista.shape[0]-2,2):
    
    pt1 = np.float32([[Lista[i,5]-Esp,Lista[i,4]-Pad],[Lista[i,5]+Esp,Lista[i,4]-Pad],[Lista[i+1,5]-Esp,Lista[i+1,4]+Pad],[Lista[i+1,5]+Esp,Lista[i+1,4]+Pad]])
    pt2 = np.float32([[0,0],[Ancho,0],[0,Alto],[Ancho,Alto]])
    
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    output = cv2.warpPerspective(Orto,matrix,(Ancho,Alto))
    
    cv2.imshow('Surco a Analizar', output) 
    
    #Orto[Lista[i,4]:Lista[i,4]+6,:]=255 # x
    #Orto[:,Lista[i,5]:Lista[i,5]+6]=255 # y
    
    img1 = output
    
    TargetB = 135.730
    TargetF = 41.4730
    
    img1 = img1.reshape(-1,1)
    img1 = np.float32(img1)
    
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,10,flags)

    MaskF = np.uint8(labels)
    
    Maskm1 = MaskF.reshape(output.shape[0],output.shape[1])
    Maskm2 = np.where(Maskm1 == 0, 1, 0)
    
    m=[Maskm1,Maskm2]
    
    maskF = np.zeros([Maskm1.shape[0],Maskm1.shape[1]])
    maskB = np.zeros([Maskm1.shape[0],Maskm1.shape[1]])
    
    B = [0,0]
    F = [0,0]  
    
    for j in range(2):
        B[j] = np.sqrt((TargetB-centers[j,0])**2)
        F[j] = np.sqrt((TargetF-centers[j,0])**2)
        
        if B[j] > F[j]:
            maskF = maskF[:,:]+m[j]
        else:
            maskB = maskB[:,:]+m[j]
            
    #cv2.imshow('Mascara frontal', maskF)
            
    maskB = np.where(maskB == 1, 255, 0).astype('uint8')
    maskF = np.where(maskF == 1, 255, 0).astype('uint8')
    
    kernel = np.ones((2,2),np.uint8)
    mask1 = cv2.morphologyEx(maskF, cv2.MORPH_OPEN, kernel)
    mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('Original',maskF)
    cv2.imshow('mask1',mask1)
    cv2.imshow('mask2',mask2)

    Guidedmask = guidedFilter(output,mask2,6,eps)
    
    #cv2.imshow('Guiado',Guidedmask)
    
    segmented_img = cv2.bitwise_or(output, output, mask=Guidedmask)
    
    cont, hie = cv2.findContours(mask2,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE)
    
    Sor_cont = sorted(cont, key=cv2.contourArea, reverse=True)
    
    roiCont = []
    total = 0
    vegetal = 0
    
    for con in Sor_cont:
        
        area = cv2.contourArea(con)
        
        if area<=1:
            continue
        
        roiCont.append(con)
        
        plantas = math.ceil(area/225)#302
        
        areas.append(area)
        
        vegetal += area
        total += plantas
        
        cv2.putText(segmented_img, str(plantas), (con[0,0,0],con[0,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    
    if total>20:
        total = 20
    
    NumeroFinal.append(total)
    areas_veg.append((vegetal/(Ancho*Alto))*100)
    var = Lista[i,1].split("_")
    tipo.append(var[0])
    
    print("-> El",Lista[i,1],"el total de plantas es: ", total)
    
    #if total<10:
        #print("-> El",Lista[i,1],"el total de plantas es: ", total)
        
    cv2.drawContours(segmented_img,roiCont,-1,(255,255,255),1)
    
    cv2.imshow('Salidita',segmented_img)
    
    #cv2.imshow('Orto Lugar', cv2.resize(Orto,None,fx=0.14,fy=0.14))
    
    #Orto = cv2.imread('V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif')
    
    if cv2.waitKey(0) & 0xFF == ord('a'):
        break
    
    cv2.imwrite(Base_imagenes_Path +"/"+"guided"+".PNG",segmented_img)
    #cv2.imwrite(Base_imagenes_Path +"/"+"mask1"+".PNG",mask1)
    #cv2.imwrite(Base_imagenes_Path +"/"+"mask2"+".PNG",mask2)
    #cv2.imwrite(Base_imagenes_Path +"/"+"vista"+".PNG",segmented_img)
    #cv2.imwrite(Base_imagenes_Path +"/"+str(var[0])+"_"+str(total)+".PNG",output)
    #cv2.imwrite(Base_imagenes_Path +"1/M"+str(var[0])+"_"+str(total)+".PNG",mask2)
    
    """if Lista[i,1] == "P138_start":
        
        cv2.imshow('Surco a Analizar', output) 

        Orto[Lista[i,4]:Lista[i,4]+8,:]=255 # x
        Orto[:,Lista[i,5]:Lista[i,5]+8]=255 # y
        
        cv2.imshow('Salidita',segmented_img)
        
        cv2.imshow('Mascara',thresh)
        
        cv2.imshow('Morfologicas',mask2)
        
        cv2.imshow('Orto Lugar', cv2.resize(Orto,None,fx=0.14,fy=0.14))
        
        Orto = cv2.imread('V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif', cv2.IMREAD_UNCHANGED)
        
        print("-> El",Lista[i,1],"el total de plantas es: ", total)
        
        if cv2.waitKey(0) & 0xFF == ord('a'):
            break"""
        
Inventario = np.column_stack((tipo,NumeroFinal,areas_veg))
    
cv2.destroyAllWindows()

cm = plt.cm.get_cmap('RdYlBu_r')

Y,X = np.histogram(NumeroFinal,bins=19)

x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C)
plt.title("Cantidad de surcos por numero de plantas")
plt.xlabel('Numero de plantas')
plt.ylabel('Cantidad de surcos')
plt.grid()
plt.show()  

X = range(1,len(areas_veg)+1)

x_span = max(X)-min(X)
C = [cm(((x-min(X))/x_span)) for x in X]

plt.bar(X,areas_veg,width = 1,color="green")
plt.title("Porcentaje vegetacion por surco")
plt.xlabel('Id del surco')
plt.ylabel('Porcentaje vegetacion')
plt.grid()
plt.show() 