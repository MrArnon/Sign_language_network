from __future__ import print_function
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing
def renorm (y,metka,a=0,b=2062):
    for j in range(a,b):
        for i in range(0, 10):
            if (i == metka):
                y[j][i] = 1
            else:
                y[j][i] = 0  # 9
    new_y=y
    return new_y
# input image dimensions
img_size = 64
# Lets load in the data
#X = np.load('X_custom.npy')
#y = np.load('Y_custom.npy')
X = np.load('X_custom.npy')
y = np.load('Y_custom.npy')
y=renorm(y,9,0,204)
y=renorm(y,0,204,409)
y=renorm(y,7,409,615)
y=renorm(y,6,615,822)
y=renorm(y,1,822,1028)
y=renorm(y,8,1028,1236)
y=renorm(y,4,1236,1443)
y=renorm(y,3,1443,1649)
y=renorm(y,2,1649,1845)
y=renorm(y,5,1845,2062)
print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))
#for i in range(0,2062):
   # X[i] = X[i].astype(float)
   # X[i]=preprocessing.scale(X[i])
    #X[i]=preprocessing.normalize((X[i]))
# create a data generator using Keras image preprocessing

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=8)

# add another axis representing grey-scale
X= X[:,:,:,np.newaxis]
Xtest = Xtest[:,:,:,np.newaxis]
Xtrain=Xtrain[:,:,:,np.newaxis]
#print('Xtest shape : {}  Ytest shape: {}'.format(Xtest.shape, ytest.shape))
load_model_check=load_model('model_check.h5')
load_model_save=load_model('sign_lang_model.h5')
load_model_check.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta()
                 optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
load_model_save.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
                optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
print('Check model :')
score = load_model_check.evaluate(Xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Save model')
score = load_model_save.evaluate(Xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

load_model_check.summary()
plot_model(load_model_check,to_file='model.png')
for i in range(0,10):
    img_path = 'pic_data/flig{id}.jpg'.format(id=i)
    img = image.load_img(img_path, target_size=(img_size, img_size),color_mode = 'grayscale')
    # = image.img_to_array(img)

    # normalize the data attributes
    #normalized_X = preprocessing.normalize(img)
    # standardize the data attributes
    standardized_X = preprocessing.scale(img)
    #x /= 255

    #normalized_X = np.expand_dims(normalized_X, axis=0)
    #normalized_X=normalized_X[:,:,:,np.newaxis]
    standardized_X = np.expand_dims(standardized_X, axis=0)
    standardized_X = standardized_X[:, :, :, np.newaxis]
    plt.imshow(standardized_X.reshape(img_size,img_size),'gray')
    #plt.show()
    #plt.imshow(normalized_X.reshape(img_size, img_size), 'gray')
    #plt.show()
    #plt.imshow(standardized_X.reshape(img_size, img_size),'gray')
    plt.show()
#print('X shape[] : {} '.format(x.shape))
#print(x)
    #prediction = load_model_check.predict_classes(normalized_X)
    #print(load_model_check.predict(normalized_X))
    #print("Prediction: ",prediction)
    prediction = load_model_check.predict_classes(standardized_X)
    print(load_model_check.predict(standardized_X))
    print("Prediction: ", prediction)
#print(ytest[300])
#xyz=X[0]
#xyz=np.expand_dims(xyz,axis=0)
#print(xyz)
#print('Xtest shape[0] : {} '.format(xyz.shape))
#plt.imshow(xyz.reshape(img_size, img_size),'gray')
#plt.show()
#prediction = load_model_save.predict_classes(xyz)
#print(load_model_save.predict(xyz))
#print("Prediction: ",prediction)
#print(y[2078])