from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


classificador = Sequential()
#filtro, dimensão de caracteriticas, altura e lagura da imagem e numero de canais, função de ativação (tira o valores negativos
classificador.add(Conv2D(32, (3,3), input_shape = (200, 200, 3), activation = 'relu'))
#normalizar entre 0 e 1
classificador.add(BatchNormalization())
#
classificador.add(MaxPooling2D(pool_size = (3,3)))

classificador.add(Conv2D(32, (3,3), input_shape = (200, 200, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (3,3)))

classificador.add(Flatten())
#qtd de neuronios
#fução de ativao
#20%
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('cnn_training_data',
                                                           target_size = (200, 200),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
classificador.fit_generator(base_treinamento,
                    steps_per_epoch=1500,
                    epochs=2,
                    validation_steps=1000)

# Salva tudo em um arquivo
classificador.save('modelo03_09072020_200x200_v2.h5')

# model_json = model.to_json()
# with open("./model.json", "w") as json_file:
#     json_file.write(model_json)
#
# model.save_weights("./model.h5")
# print("saved model..! ready to go.")
