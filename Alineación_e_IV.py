
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

"""Se realiza la lectura de los ortomosaicos necesarios para estimar los 
indices vegetativos"""

orto_nir=cv2.imread("ortofotos 19 julio 12m/V1_19_J_22_12m_NIIR_UHD_P4_R-orthophoto.tif",
                   cv2.IMREAD_UNCHANGED)
orto_red=cv2.imread("ortofotos 19 julio 12m/V1_19_J_22_12m_Red_UHD_P4_R-orthophoto.tif/",
                    cv2.IMREAD_UNCHANGED)
orto_rededge=cv2.imread("ortofotos 19 julio 12m/V1_19_J_22_12m_RedEdge_UHD_P4_R-orthophoto.tif",
                        cv2.IMREAD_UNCHANGED)
orto_green=cv2.imread("ortofotos 19 julio 12m/V1_19_J_22_12m_Green_UHD_P4_R-orthophoto.tif",
                      cv2.IMREAD_UNCHANGED)
orto_blue=cv2.imread("ortofotos 19 julio 12m/V1_19_J_22_12m_Blue_UHD_P4_R-orthophoto.tif",
                     cv2.IMREAD_UNCHANGED)
 

def redimension ( imagen_r ,Ancho , Alto):
    """Retorna la misma imagen de entrada, con el rescalamiento correspondiente 
    
    Esta función recibe 3 parametros:
    imagen_r -- Imagen 2d a la cual se desea reescalar
    Ancho -- Número que indica el ancho final deseado para la imagen
    Alto -- Número quu indica el alto final deseado para la imagen    
    """
    
    imagen_r=cv2.resize(imagen_r,(Ancho, Alto), interpolation = cv2.INTER_LINEAR)
    return imagen_r

    
def unir3bandas(banda1,banda2,banda3,Ancho, Alto,mostrar = "n"):
    """Retorna una imagen 3D, construida a partir de las imagenes de entrada
    banda1, banda2 y banda3
        
    Esta función recibe 5 parametros:
    banda1 -- Primra imagen 2d para ser sobrepuesta
    banda2 -- Segunda imagen 2d para ser sobrepuesta
    banda3 -- Tercera imagen 2d para ser sobrepuesta
    Ancho -- Ancho con el que se imprime la imagen en pantalla
    Alto -- Alto con el que se imprime la imagen resultante en pantalla
    mostrar -- Variable con la cual se espesifica si mostrar o no la imagen
               resultante
    
    

    """    
    
    imagen=cv2.merge((banda1 ,banda2 ,banda3))
    imagen=cv2.resize(imagen,(Ancho, Alto), interpolation = cv2.INTER_LINEAR)
    if (mostrar == "y" ):
        cv2.imshow("Union de bandas ",imagen)
        if(cv2.waitKey(0) & 0xFF == ord('a')):
            cv2.destroyAllWindows()   

    return imagen



def alineador(ima_procesar,ima_base):
    """Retorna ima_procesar alineada respecto a ima_base, resultado de la 
    busqueda y detección de puntos clave  con sus descriptores, usando un
    obejto ORB. Despues se usa un objeto DescriptorMatcher con KNN para medir
    la coincidencia entre descriptores, seleccionar los mejores y usando su
    matriz de homografia hacer un cambio de perspectiva para alinear ambas 
    imagenes
    
    Esta función recibe dos parametros:
    ima_procesar -- imagen 2d que se alineara a ima_base
    ima_base -- imagen 2d que se tomara de base para la alineación
    """
    ratio= 0.75
    error_reproyeccion = 4.0
    
    """Se crea un objeto ORB para detectar los puntos de interes de las imagenes"""
    """En este caso el parametro modificado es el de nFeatures en 5000"""
    orb_detector = cv2.ORB_create(5000)
    "Se hayan los puntos clave Kp y descriptores d de ambas imagenes "
    kp1, d1 = orb_detector.detectAndCompute(ima_procesar, None)
    kp2, d2 = orb_detector.detectAndCompute(ima_base, None)
    
    """Se crea el objeto DescriptorMatcher para despues usar el knnMatch """
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(d1, d2, 2)
    matches = []
    
    "Se encuentran las mejores coincidencias entre descriptores"
    count=0                                
    for m in rawMatches:

        #distance es medida de referencia de la cantidad de realción 
        #entre descriptores
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
            
    "Se guardan los puntos de interes de mejor coincidencia en ptsA y ptsB"
    if len(matches) > 4:
          
            ptsA = np.float32([kp1[i].pt for (_, i) in matches])
            ptsB = np.float32([kp2[i].pt for (i, _) in matches])
            
            
            """Se calcula la matriz de homografia usando ambos conjuntos de puntos"""
            """Con la homografía se realizar la transformación de perspectiva y  se alinea"""
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, error_reproyeccion)
            estabilizada = cv2.warpPerspective(ima_procesar, H, (ima_procesar.shape[1], ima_procesar.shape[0]))
    return estabilizada

def ndvi_iv(img_nir , img_red, numbar, anchobar, hlimx_neg, hlimx_pos, hlimy_neg, hlimy_pos) :
    """Retorna los valores del indice ndvi por pixel en una matriz de dos dimensiones, ndvi_index,
    ademas retorna una imagen que representa el indice ndvi con un mapa de color (RdYlBu_r)
    en im_color_ndvi. Para esto se procesa la ecuación del indice ndvi corrigiendo resultados
    'nan' para poderlos operar. Con ese resultado se realiza un histograma con el valor
    del indice por pixel corrigiendo valores atipicos del indice vegetativo, tambien 
    se transforma el valor a escala de 0 a 255 para poder transformar sus valores con un
    mapa de color, teniendo una imagen de representación.
    
    Esta función recibe 8 parametros:
    img_nir -- Imagen 2d con la información de la banda NIR
    img_red -- Imagen 2d con la información de la banda RED
    numbar -- Número para modificar el número de barras en el histograma 
    anchobar -- Número para modificar el ancho de las barras en el histograma
    hlimx_neg -- Limine inferior del histograma en x
    hlimx_pos -- Limite superior del histograma en x
    hlimy_neg -- limite inferior del histograma en y
    hlimy_pos -- Limite superior del histograma en y
    """
    """Se transforma el tipo dee varaibel a float para operar con decimales"""
    nir = np.array(img_nir, dtype=float)
    red = np.array(img_red, dtype=float)
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    """ Calculo y corrección de valores del indice"""
    ndvi = (nir - red) / (nir + red)
    ndvi=np.nan_to_num(ndvi)
    ndvi_index=ndvi.copy()
    ndvi_graf=ndvi.copy()
    ndvi_graf[ndvi_graf==0] = 1
    ndvi_graf[ndvi_graf > 1] = 1
    ndvi_graf[ndvi_graf < -1] = -1
    
    """Creación del histograma"""
    Y1,X1 = np.histogram(ndvi_graf,numbar)
    x1_span = X1.max()-X1.min()
    C1 = [cm(((x-X1.min())/x1_span)) for x in X1]
    plt.bar(X1[:-1],Y1,color=C1,width=anchobar)
    plt.ylim([hlimy_neg,hlimy_pos])
    plt.xlim([hlimx_neg,hlimx_pos])
    plt.xlabel("Valor índice vegetativo")
    plt.ylabel("No. pixeles por valor")
    plt.title("NDVI")
    plt.show() 
    
   
    """Creación de imagen con mapa de color"""    
    ndvi = ((ndvi) +(ndvi.min() * -1))
    z=np.sort(ndvi)
    max_=statistics.median(z[:,(red.shape[1])-1])
    ndvi = (ndvi*255)/max_ 
    ndvi = ndvi.round()
    ndvi_image = np.array(ndvi, dtype=np.uint8)
    im_color_ndvi = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_RAINBOW)

    return ndvi_index , im_color_ndvi

def gci_iv(img_nir , img_green, numbar, anchobar, hlimx_neg, hlimx_pos, hlimy_neg, hlimy_pos) :
    """Retorna los valores del indice gci por pixel en una matriz de dos dimensiones, gci_index,
    ademas retorna una imagen que representa el indice gci con un mapa de color (RdYlBu_r)
    en im_color_gci. Para esto se procesa la ecuación del indice gci corrigiendo resultados
    'nan' y '+-inf' para poderlos operar. Con ese resultado se realiza un histograma con el valor
    del indice por pixel corrigiendo valores atipicos del indice vegetativo, tambien 
    se transforma el valor a escala de 0 a 255 para poder transformar sus valores con un
    mapa de color, teniendo una imagen de representación.
    
    Esta función recibe 8 parametros:
    img_nir -- Imagen 2d con la información de la banda NIR
    img_green -- Imagen 2d con la información de la banda GREEN
    numbar -- Número para modificar el número de barras en el histograma 
    anchobar -- Número para modificar el ancho de las barras en el histograma
    hlimx_neg -- Limine inferior del histograma en x
    hlimx_pos -- Limite superior del histograma en x
    hlimy_neg -- limite inferior del histograma en y
    hlimy_pos -- Limite superior del histograma en y
    """
    """Se transforma el tipo dee varaibel a float para operar con decimales"""
    nir = np.array(img_nir, dtype=float)
    green = np.array(img_green, dtype=float)
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    """ Calculo y corrección de valores del indice"""
    gci =  (nir/green)-1
    gci[np.isneginf(gci)] = 0
    gci[np.isinf(gci)] = 0
    gci=np.nan_to_num(gci)
    gci_index=gci.copy()
    gci_graf=gci.copy()
    gci_graf[gci_graf==0] = 1
    gci_graf[gci_graf > 1] = 1
    gci_graf[gci_graf < -1] = -1
    
    """Creación del histograma"""
    Y2,X2 = np.histogram(gci_graf,numbar)
    x2_span = X2.max()-X2.min()
    C2 = [cm(((x-X2.min())/x2_span)) for x in X2]
    plt.bar(X2[:-1],Y2,color=C2,width=anchobar)
    plt.ylim([hlimy_neg,hlimy_pos])
    plt.xlim([hlimx_neg,hlimx_pos])
    plt.xlabel("Valor índice vegetativo")
    plt.ylabel("No. pixeles por valor")
    plt.title("GCI")
    plt.show() 
    
 
    """Creación de imagen con mapa de color"""
    gci = gci + (gci.min() * -1)
    z=np.sort(gci)
    max_=statistics.median(z[:,(nir.shape[1])-1])
    gci = (gci*255)/max_
    gci = gci.round()
    gci_image = np.array(gci, dtype=np.uint8)
    im_color_gci = cv2.applyColorMap(gci_image, cv2.COLORMAP_RAINBOW)
    cv2.imwrite('indice gci.png', im_color_gci)
    
    return gci_index , im_color_gci 



def gndvi_iv(img_nir , img_green, numbar, anchobar, hlimx_neg, hlimx_pos, hlimy_neg, hlimy_pos) :
    """Retorna los valores del indice gndvi por pixel en una matriz de dos dimensiones, gndvi_index,
    ademas retorna una imagen que representa el indice gndvi con un mapa de color (RdYlBu_r)
    en im_color_gndvi. Para esto se procesa la ecuación del indice gndvi corrigiendo resultados
    'nan' para poderlos operar. Con ese resultado se realiza un histograma con el valor
    del indice por pixel corrigiendo valores atipicos del indice vegetativo, tambien 
    se transforma el valor a escala de 0 a 255 para poder transformar sus valores con un
    mapa de color, teniendo una imagen de representación.
    
    Esta función recibe 8 parametros:
    img_nir -- Imagen 2d con la información de la banda NIR
    img_green -- Imagen 2d con la información de la banda GREEN
    numbar -- Número para modificar el número de barras en el histograma 
    anchobar -- Número para modificar el ancho de las barras en el histograma
    hlimx_neg -- Limine inferior del histograma en x
    hlimx_pos -- Limite superior del histograma en x
    hlimy_neg -- limite inferior del histograma en y
    hlimy_pos -- Limite superior del histograma en y
    """
    """Se transforma el tipo dee varaibel a float para operar con decimales"""
    nir = np.array(img_nir, dtype=float)
    green = np.array(img_green, dtype=float)
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    """ Calculo y corrección de valores del indice"""    
    gndvi = (nir - green) / (nir + green)  
    gndvi = np.nan_to_num(gndvi)
    gndvi_index = gndvi.copy()
    gndvi_graf= gndvi.copy() 
    gndvi_graf[gndvi_graf==0] = 1
    gndvi_graf[gndvi_graf > 1] = 1
    gndvi_graf[gndvi_graf < -1] = -1
    
    """Creación del histograma"""
    Y,X = np.histogram(gndvi_graf,numbar)
    x_span = X.max()-X.min()
    C = [cm(((x-X.min())/x_span)) for x in X]
    plt.bar(X[:-1],Y,color=C,width=anchobar)
    plt.ylim([hlimy_neg,hlimy_pos])
    plt.xlim([hlimx_neg,hlimx_pos])
    plt.xlabel("Valor índice vegetativo")
    plt.ylabel("No. pixeles por valor")
    plt.title("GNDVI")
    plt.show() 
    
    
    """Creación de imagen con mapa de color"""
    gndvi = gndvi + (gndvi.min() *-1)
    z=np.sort(gndvi)
    max_=statistics.median(z[:,(nir.shape[1])-1])
    gndvi = (gndvi*255)/max_
    gndvi = gndvi.round()
    gndvi_image = np.array(gndvi, dtype=np.uint8)
    im_color_gndvi = cv2.applyColorMap(gndvi_image, cv2.COLORMAP_RAINBOW)

    return gndvi_index , im_color_gndvi 

def gli_iv(img_red , img_green, img_blue, numbar, anchobar, hlimx_neg, hlimx_pos, hlimy_neg, hlimy_pos) :
    """Retorna los valores del indice gli por pixel en una matriz de dos dimensiones, gli_index,
    ademas retorna una imagen que representa el indice gli con un mapa de color (RdYlBu_r)
    en im_color_gli. Para esto se procesa la ecuación del indice gli corrigiendo resultados
    'nan' y '+-inf' para poderlos operar. Con ese resultado se realiza un histograma con el valor
    del indice por pixel corrigiendo valores atipicos del indice vegetativo, tambien 
    se transforma el valor a escala de 0 a 255 para poder transformar sus valores con un
    mapa de color, teniendo una imagen de representación.
    
    Esta función recibe 9 parametros:
    img_red -- Imagen 2d con la información de la banda RED
    img_green -- Imagen 2d con la información de la banda GREEN
    img_blue -- Imagen 2d con la información de la banda BLUE  
    numbar -- Número para modificar el número de barras en el histograma 
    anchobar -- Número para modificar el ancho de las barras en el histograma
    hlimx_neg -- Limine inferior del histograma en x
    hlimx_pos -- Limite superior del histograma en x
    hlimy_neg -- limite inferior del histograma en y
    hlimy_pos -- Limite superior del histograma en y
    """
    """Se transforma el tipo dee varaibel a float para operar con decimales"""
    red = np.array(img_red, dtype=float)
    green = np.array(img_green, dtype=float)
    blue = np.array(img_blue, dtype=float)
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    """ Calculo y corrección de valores del indice"""
    gli = ((green - red)+(green-blue))/((2*green)+red+blue)
    gli[np.isneginf(gli)] = 0
    gli[np.isinf(gli)] = 0
    gli= np.nan_to_num(gli)
    gli_index=gli.copy()
    gli_graf=gli.copy()
    gli_graf[gli_graf==0] = 1
    gli_graf[gli_graf > 1] = 1
    gli_graf[gli_graf < -1] = -1
    
    """Creación del histograma"""
    Y,X = np.histogram(gli_graf,numbar)
    x_span = X.max()-X.min()
    C = [cm(((x-X.min())/x_span)) for x in X]
    plt.bar(X[:-1],Y,color=C,width=anchobar)
    plt.ylim([hlimy_neg,hlimy_pos])
    plt.xlim([hlimx_neg,hlimx_pos])
    plt.xlabel("Valor índice vegetativo")
    plt.ylabel("No. pixeles por valor")
    plt.title("GLI")
    plt.show()   
    

    """Creación de imagen con mapa de color"""
    gli = (gli + (gli.min() * -1)) 
    z=np.sort(gli)
    max_=statistics.median(z[:,(red.shape[1])-1])
    gli = (gli*255)/max_
    gli = gli.round()
    gli_image = np.array(gli, dtype=np.uint8)
    im_color_gli = cv2.applyColorMap(gli_image, cv2.COLORMAP_RAINBOW)
    
    return gli_index , im_color_gli
 


orto_nir = redimension(orto_nir ,7881 ,7788)
orto_green = redimension(orto_green ,7881 ,7788)
orto_blue = redimension(orto_blue ,7881 ,7788)
orto_red = redimension(orto_red ,7881 ,7788)

img_prealineacion=unir3bandas(orto_nir ,orto_green, orto_blue,780,770,mostrar = "n")


img_nir_green=alineador(orto_nir,orto_green) 
img_red_green=alineador(orto_red,orto_green) 
img_rededge_green=alineador(orto_rededge,orto_green) 
img_blue_green=alineador(orto_blue,orto_green)
img_green_green=alineador(orto_green, orto_green)

cv2.imwrite("NIR Julio.tif",img_nir_green)
cv2.imwrite("RED Julio.tif",img_red_green)
cv2.imwrite("BLUE Julio.tif",img_blue_green)
cv2.imwrite("REDEDGE Julio.tif",img_rededge_green)
cv2.imwrite("GREEN Julio.tif",img_green_green)

      
estabilizada = unir3bandas(img_nir_green,img_green_green,img_blue_green,780,770,mostrar = "n")    

indice_ndvi,img_ndvi = ndvi_iv(img_nir_green,img_red_green,15000,0.01,-0.75,0.75,0,125000) 
indice_gci, img_gci  = gci_iv(img_nir_green,img_green_green,15000,0.01,-0.75,0.75,0,125000)
indice_gndvi, img_gndvi = gndvi_iv(img_nir_green,img_green_green,15000,0.01,-1.0,1.0,0,125000)
indice_gli, img_gli  = gli_iv(img_red_green,img_green_green,img_blue_green,15000,0.01,-1.0,1.0,0,125000)


cv2.waitKey(0)
cv2.destroyAllWindows()   
