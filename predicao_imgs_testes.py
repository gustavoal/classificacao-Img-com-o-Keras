import cv2
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image



MODEL = load_model("./modelo03_09072020_200x200_v2.h5")

caracteristica = list()
color = (0, 255, 0)


img = cv2.imread("data/original/DJI_0004.JPG")
print(img.shape)
#print("Imagem Carregada")
print('\n')
img0 = img
#linha
for i in range(0,img.shape[0], 200):
    inicial = i
    segundo = i + 200
    #coluna
    for x in range(0, img.shape[1], 200):  # coluna
        coluna_inicial = x
        coluna_seguinte = x + 200
        h = 200
        w = 200
        img2 = img[inicial:inicial + h, coluna_inicial:coluna_inicial + w]
        cv2.imwrite('data/original/img_janela.png', img2)

        ########
        from keras.preprocessing import image
        imagem_teste = image.load_img('data/original/img_janela.png',
                                      target_size=(200, 200))

        imagem_teste = image.img_to_array(imagem_teste)

        imagem_teste /= 255
        imagem_teste = np.expand_dims(imagem_teste, axis=0)
        previsao = MODEL.predict(imagem_teste)
        previsao = (previsao > 0.91)
        #print(previsao[0][0])
        os.remove('data/original/img_janela.png')
        if previsao:
            img0 = cv2.rectangle(img0, (coluna_inicial, inicial), (coluna_seguinte, segundo), (0, 255, 0), 14)

        ########
'''
            os.remove(saida)
            os.remove('imgOPENCV.JPG')

            if modelo1 == 1:
                print("********************************passou aqui***************************************")
                image = cv2.rectangle(image, (coluna_inicial, inicial), (coluna_seguinte, segundo), color, 6)
                #cv2.imwrite('DJI_0199.JPG', image)
                
'''
#cv2.imshow('img', image)
cv2.imwrite('DJI_0004_valida.JPG', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()