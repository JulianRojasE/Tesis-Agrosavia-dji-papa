import cv2
import numpy as np
import os
import math
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from cv2.ximgproc import guidedFilter

#Path_Orto = r'V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif'
Path_Orto = r'V1_19_J_22_12m_JPG_UHD_P4_R-orthophoto.tif'
Path_Qgis = r'Puntos QGIS_FIN.csv'
Base_Salida_Path = "C:/Users/julia/Desktop/9 semestre/Tesis/GFK/Base_de_Salida"

if not os.path.exists(Base_Salida_Path):
    os.makedirs(Base_Salida_Path)
    print("No existia la carpeta pero se creo en: ", Base_Salida_Path)
    
Orto = cv2.imread(Path_Orto)

NumeroFinal=[]
tipo=[]
areas_veg = []
areas = []

eps = 10e-6
eps *= 255*255

Distanciax = []
Distanciay = []

def GeoreferenciaTran(Path_Qgis,Path_Orto):
    """Retorna un arreglo numpy con los indices para la latitud 
    y longitud de los puntos de inicio y final en el ortomosaico
    indicado en el parametro Path_Orto

    esta funcion recibe dos parametros;
    Path_Qgis -- la ruta donde se encuentran las coordenadas
    extraidas de Qgis de los puntos de inicio y fin de cada surco

    Path_Orto -- Ruta del ortomosaico de interes

    """
    Puntos = pd.read_csv(Path_Qgis)
    Puntos['LatPixel'] = 0
    Puntos['LonPixel'] = 0
    
    imgT = rasterio.open(Path_Orto)
    for i in range(Puntos.shape[0]):
        Cord_Pixel = imgT.index(Puntos['Lat'].iloc[i],Puntos['Lon'].iloc[i], z=None, precision=None)
        Puntos['LatPixel'].iloc[i]= Cord_Pixel[0]
        Puntos['LonPixel'].iloc[i]= Cord_Pixel[1]
        
    return Puntos.to_numpy()

def DistanciasROI(Lista):
    """
    
    """
    for i in range(0,Lista.shape[0]-2,2):
        dist = np.sqrt((Lista[i,4]-Lista[i+2,4])**2+(Lista[i,5]-Lista[i+2,5])**2)/2
        dist = int(dist)
        Distanciax.append(dist)
        
        dist = np.sqrt((Lista[i,4]-Lista[i+1,4])**2+(Lista[i,5]-Lista[i+1,5])**2)
        dist = int(dist)
        Distanciay.append(dist)

def ExtraccionSurco(i):
    pt1 = np.float32([[Lista[i,5]-Esp,Lista[i,4]-Pad],[Lista[i,5]+Esp,Lista[i,4]-Pad],[Lista[i+1,5]-Esp,Lista[i+1,4]+Pad],[Lista[i+1,5]+Esp,Lista[i+1,4]+Pad]])
    pt2 = np.float32([[0,0],[Ancho,0],[0,Alto],[Ancho,Alto]])
    matriz = cv2.getPerspectiveTransform(pt1,pt2)
    output = cv2.warpPerspective(Orto,matriz,(Ancho,Alto))
    return output

def SegUmbralizacion(Surco):
    """Retorna la mascaara binaria resultado de la segmentación por umbralizacion
       para un surco del cultivo

    Esta funcion recibe un parametro:
    Surco -- imagen de uno de los n surcos del cultivo
       producto de aplicar la funcion ExtraccionSurco()
    """
    blur = cv2.GaussianBlur(Surco,(5,5),0)
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def SegEspacioColor(Surco):
    """Retorna la mascara binaria resultado de la segmentación por espacio de
       color
    
    Esta funcion recine un parametro:
    Surco -- imagen de uno de los n surcos del cultivo
       producto de aplicar la funcion ExtraccionSurco()
    """
    blur = cv2.GaussianBlur(Surco,(5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    (h, s, v) = cv2.split(hsv)

    ret,thresh = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret1,thresh1 = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret2,thresh2 = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    lower_bound = np.array([27 , int(round(ret1*0.45)), int(round(ret2*0.47))])   

    upper_bound = np.array([110 , int(round(ret1*0.45)*8), int(round(ret2*0.47)*4)])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    return mask

def SegKmedias(Surco):
    """Retorna la mascara binaria resultado de la segmentacion de k-medias para 
       un surco del cultivo

       Esta funcion recibe un parametro:
       Surco -- imagen de uno de los n surcos del cultivo
            producto de aplicar la funcion ExtraccionSurco()

    """
    img1 = np.copy(Surco)
    
    TargetB = 135.730
    TargetF = 41.4730
    
    img1 = img1.reshape(-1,1)
    img1 = np.float32(img1)
    
    # Define criteria 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    # Set flags 
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    compactness,labels,centers = cv2.kmeans(img1,2,None,criteria,10,flags)

    MaskF = np.uint8(labels)
    
    Maskm1 = MaskF.reshape(Surco.shape[0],Surco.shape[1])
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
            
    maskB = np.where(maskB == 1, 255, 0).astype('uint8')
    maskF = np.where(maskF == 1, 255, 0).astype('uint8')
    
    return maskF


def RefinamientoMask(Mascara,output):
    """Esta funcion retorna el resultado de aplicar a una mascara binaria, resultado
    del proceso de segmentacion las operaciones morfologicas de apertura y cierre de
    forma suceciva y tambien la aplicacion de un filtro guiado que tiene como imagen 
    guia la misma imagen que se sometio al proceso de segmentacion

    maskRef es el resultado de aplicar las operaciones morfologicas con un kernel 2x2
    Guidedmask es el resultado de aplicar un filtro guiado a maskRef
    """
    kernel = np.ones((2,2),np.uint8)
    mask1 = cv2.morphologyEx(Mascara, cv2.MORPH_OPEN, kernel)
    maskRef = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    Guidedmask = guidedFilter(output,maskRef,6,eps)
    return maskRef,Guidedmask

def Conteo(i,Surco_seg,maskRef,mostrar = "n"):
    contorno, hie = cv2.findContours(maskRef,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_NONE)
    Sor_cont = sorted(contorno, key=cv2.contourArea, reverse=True)
    
    roiCont = []
    total = 0
    vegetal = 0
    
    for con in Sor_cont:
        area = cv2.contourArea(con)
        if area<=1:
            continue
        roiCont.append(con)
        plantas = math.ceil(area/225)
        areas.append(area)
        vegetal += area
        total += plantas
        cv2.putText(Surco_seg, str(plantas), (con[0,0,0],con[0,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    
    cv2.drawContours(Surco_seg,roiCont,-1,(255,255,255),1)
    
    if total>20:
        total = 20
        
    if total<14 and mostrar == "y":
        print("X> El surco",Lista[i,1],"tiene un total de:",total,"plantas")
        
    NumeroFinal.append(total)
    areas_veg.append((vegetal/(Ancho*Alto))*100)
    
    return total
    
def MostrarSurcoN(i,output,Surco_seg,semegmentacion,maskRef,total,GenotipoN = "n"):
    if Lista[i,1] == GenotipoN and GenotipoN != "n":
        OrtoMos = np.copy(Orto)
        OrtoMos[Lista[i,4]:Lista[i,4]+8,:]=255 # x
        OrtoMos[:,Lista[i,5]:Lista[i,5]+8]=255 # y
        
        cv2.imshow('Orto Lugar', cv2.resize(OrtoMos,None,fx=0.14,fy=0.14))
        cv2.imshow('Surco a Analizar', output)
        cv2.imshow('Mascara',semegmentacion)
        cv2.imshow('Morfologicas',maskRef)
        cv2.imshow('Conteo',Surco_seg)
        
        print("-> El surco",Lista[i,1],"tiene un total de:",total,"plantas")
        
        cv2.waitKey(0)
    else:
        return
        
        
def MostrarOperaciones(i,output,Surco_seg,semegmentacion,maskRef,Guidedmask,total,mostrar = "n"):
    if mostrar == "y":   
        OrtoMos = np.copy(Orto)
        OrtoMos[Lista[i,4]:Lista[i,4]+8,:]=255 # x
        OrtoMos[:,Lista[i,5]:Lista[i,5]+8]=255 # y
        
        cv2.imshow('Orto Lugar', cv2.resize(OrtoMos,None,fx=0.14,fy=0.14))
        cv2.imshow('Surco a Analizar', output)
        cv2.imshow('Mascara',semegmentacion)
        cv2.imshow('Morfologicas',maskRef)
        cv2.imshow('Filtro Guiado',Guidedmask)
        cv2.imshow('Conteo',Surco_seg)
        
        print("-> El surco",Lista[i,1],"tiene un total de:",total,"plantas")
        
        cv2.waitKey(0)
    else:
        return

def GuardarSalida(Surco,total,Mascara,Guidedmask,Genotipo,Guardar = "n"):
    if Guardar == "y":
        cv2.imwrite(Base_Salida_Path +"/O_"+str(Genotipo)+"_"+str(total)+".PNG",Surco)
        cv2.imwrite(Base_Salida_Path +"1/M_"+str(Genotipo)+"_"+str(total)+".PNG",Mascara)
        cv2.imwrite(Base_Salida_Path +"2/G_"+str(Genotipo)+"_"+str(total)+".PNG",Guidedmask)
    else:
        return    

# Main 

Lista = GeoreferenciaTran(Path_Qgis,Path_Orto)

DistanciasROI(Lista)

Esp = 21
Pad = 25
Alto = 294 + 2*Pad
Ancho = Esp*2+1

       
for i in range(0,Lista.shape[0]-2,2):
    Surco = ExtraccionSurco(i)
    Mascara = SegEspacioColor(Surco)
    maskRef,Guidedmask = RefinamientoMask(Mascara,Surco)
    Surco_seg = cv2.bitwise_or(Surco, Surco, mask=maskRef)
    NumeroPlantas = Conteo(i,Surco_seg,maskRef,mostrar = "n")
    MostrarSurcoN(i,Surco,Surco_seg,Mascara,maskRef,NumeroPlantas,GenotipoN = "n")
    MostrarOperaciones(i,Surco,Surco_seg,Mascara,maskRef,Guidedmask,NumeroPlantas,mostrar = "n")
    Genotipo = Lista[i,1].split("_")
    tipo.append(Genotipo[0])
    GuardarSalida(Surco,NumeroPlantas,Mascara,Guidedmask,Genotipo[0],Guardar = "y")
    if cv2.waitKey(0) & 0xFF == ord('a'):
        break
    
Inventario = np.column_stack((tipo,NumeroFinal,areas_veg))
    
np.savetxt(Base_Salida_Path+"/Inventario.txt", Inventario,fmt="%s", delimiter =",",header="Genotipo, Conteo, % Veg")

cv2.destroyAllWindows()

cm = plt.cm.get_cmap('RdYlBu_r')

Y,X = np.histogram(NumeroFinal,bins=19)

x_span = X.max()-X.min()
C = [cm(((x-X.min())/x_span)) for x in X]

plt.bar(X[:-1],Y,color=C)
plt.title("Cantidad de surcos por número de plantas")
plt.xlabel('Número de plantas')
plt.ylabel('Cantidad de surcos')
plt.grid()
plt.show()  

X = range(1,len(areas_veg)+1)

plt.bar(X,areas_veg,width = 1,color="green")
plt.title("Porcentaje vegetación por surco")
plt.xlabel('Id del surco')
plt.ylabel('Porcentaje vegetación')
plt.grid()
plt.show()