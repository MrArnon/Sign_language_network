from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.preprocessing import image

batch_size = 8

num_classes = 10
# input image dimensions
epochs = 100
img_size = 64
def renorm (y,metka,a=0,b=2062):
    for j in range(a,b):
        for i in range(0, 10):
            if (i == metka):
                y[j][i] = 1
            else:
                y[j][i] = 0  # 9
    new_y=y
    return new_y
def loading_custom_data (str,arr,y_arr):
    for i in range (0,num_classes):

        img_path = 'pic_data/{id}.jpg'.format(id=str+'{num}').format(num=i)
        try:

            img = image.load_img(img_path, target_size=(img_size, img_size),color_mode = 'grayscale')
            y_img=np.zeros(num_classes)
            y_img[i]=1
            y_arr=np.append(y_arr,y_img)
            img=np.array(img).astype(float)
            img=img.reshape(img_size,img_size)
            standardized_X=preprocessing.scale(img)
            arr=np.append(arr,standardized_X)

        except FileNotFoundError:
            if(i<num_classes):   
                i=(i+1)
            else:
                break

    return arr,y_arr
# Lets load in the data
X = np.load('X.npy')
y = np.load('Y.npy')

#print('X shape : {}  Y shape: {}'.format(X.shape, y.shape))
for i in range(0,2062):
    X[i] = X[i].astype(float)
    #X[i]=preprocessing.scale(X[i])
    #X[i]=preprocessing.normalize(X[i])
    X[i]=preprocessing.scale(X[i])
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
arr=np.zeros(0)
y_arr= np.zeros(0)
arr,y_arr=loading_custom_data('a',arr,y_arr)
arr,y_arr=loading_custom_data('p',arr,y_arr)
arr,y_arr=loading_custom_data('n',arr,y_arr)
arr,y_arr=loading_custom_data('l',arr,y_arr)
arr,y_arr=loading_custom_data('ler',arr,y_arr)
arr,y_arr=loading_custom_data('le',arr,y_arr)
arr,y_arr=loading_custom_data('al',arr,y_arr)
arr,y_arr=loading_custom_data('ali',arr,y_arr)
arr,y_arr=loading_custom_data('bak',arr,y_arr)
arr,y_arr=loading_custom_data('baku',arr,y_arr)
arr,y_arr=loading_custom_data('br',arr,y_arr)
arr,y_arr=loading_custom_data('bru',arr,y_arr)
arr,y_arr=loading_custom_data('bu',arr,y_arr)
arr,y_arr=loading_custom_data('buk',arr,y_arr)
arr,y_arr=loading_custom_data('dan',arr,y_arr)
arr,y_arr=loading_custom_data('dash',arr,y_arr)
arr,y_arr=loading_custom_data('dem',arr,y_arr)
arr,y_arr=loading_custom_data('demi',arr,y_arr)
arr,y_arr=loading_custom_data('fli',arr,y_arr)
arr,y_arr=loading_custom_data('flig',arr,y_arr)
arr,y_arr=loading_custom_data('iv',arr,y_arr)
arr,y_arr=loading_custom_data('iva',arr,y_arr)
arr,y_arr=loading_custom_data('jul',arr,y_arr)
arr,y_arr=loading_custom_data('kat',arr,y_arr)
arr,y_arr=loading_custom_data('kate',arr,y_arr)
arr,y_arr=loading_custom_data('kol',arr,y_arr)
arr,y_arr=loading_custom_data('kor',arr,y_arr)
arr,y_arr=loading_custom_data('kse',arr,y_arr)
arr,y_arr=loading_custom_data('ksee',arr,y_arr)
arr,y_arr=loading_custom_data('ksu',arr,y_arr)
arr,y_arr=loading_custom_data('mash',arr,y_arr)
arr,y_arr=loading_custom_data('masha',arr,y_arr)
arr,y_arr=loading_custom_data('my',arr,y_arr)
arr,y_arr=loading_custom_data('myy',arr,y_arr)
arr,y_arr=loading_custom_data('nas',arr,y_arr)
arr,y_arr=loading_custom_data('nast',arr,y_arr)
arr,y_arr=loading_custom_data('nat',arr,y_arr)
arr,y_arr=loading_custom_data('nata',arr,y_arr)
arr,y_arr=loading_custom_data('ne',arr,y_arr)
arr,y_arr=loading_custom_data('nel',arr,y_arr)
arr,y_arr=loading_custom_data('neli',arr,y_arr)
arr,y_arr=loading_custom_data('nelli',arr,y_arr)
arr,y_arr=loading_custom_data('nik',arr,y_arr)
arr,y_arr=loading_custom_data('niki',arr,y_arr)
arr,y_arr=loading_custom_data('niko',arr,y_arr)
arr,y_arr=loading_custom_data('pet',arr,y_arr)
arr,y_arr=loading_custom_data('petr',arr,y_arr)
arr,y_arr=loading_custom_data('po',arr,y_arr)
arr,y_arr=loading_custom_data('pol',arr,y_arr)
arr,y_arr=loading_custom_data('poli',arr,y_arr)
arr,y_arr=loading_custom_data('pos',arr,y_arr)
arr,y_arr=loading_custom_data('re',arr,y_arr)
arr,y_arr=loading_custom_data('rez',arr,y_arr)
arr,y_arr=loading_custom_data('roz',arr,y_arr)
arr,y_arr=loading_custom_data('sh',arr,y_arr)
arr,y_arr=loading_custom_data('sha',arr,y_arr)
arr,y_arr=loading_custom_data('tem',arr,y_arr)
arr,y_arr=loading_custom_data('ul',arr,y_arr)
arr,y_arr=loading_custom_data('ula',arr,y_arr)
arr,y_arr=loading_custom_data('ura',arr,y_arr)
arr,y_arr=loading_custom_data('uraz',arr,y_arr)
arr,y_arr=loading_custom_data('ze',arr,y_arr)
arr,y_arr=loading_custom_data('zen',arr,y_arr)
arr,y_arr=loading_custom_data('ver',arr,y_arr)
arr,y_arr=loading_custom_data('pro',arr,y_arr)
arr,y_arr=loading_custom_data('pop',arr,y_arr)
arr,y_arr=loading_custom_data('tar',arr,y_arr)
arr,y_arr=loading_custom_data('tara',arr,y_arr)
arr,y_arr=loading_custom_data('mal',arr,y_arr)
arr,y_arr=loading_custom_data('malt',arr,y_arr)
arr,y_arr=loading_custom_data('she',arr,y_arr)
arr,y_arr=loading_custom_data('popo',arr,y_arr)

k=int((int(y_arr.shape[0])/num_classes))
y_arr=y_arr.reshape(k,num_classes)
#print(y_arr)
arr=arr.reshape(k,img_size,img_size)
X=np.vstack((X, arr))
y=np.vstack((y,y_arr))

print(X.shape)
print(y.shape)
np.save('X_custom.npy',X)
np.save('Y_custom.npy',y)



















