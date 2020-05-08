# -*- coding: utf-8 -*-

# DESCRIPTION : PROGRAM TO CLASSIFY IMAGES .

#LOADING THE DATASET .
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#PRINT THE DATATYPE
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

#GET THE SHAPES OF THE DATA
print('x_train shape:' , x_train.shape)
print('y_train shape:' , y_train.shape)
print('x_test shape:' , x_test.shape)
print('y_test shape:' , y_test.shape)

#FRIST IMAGE IN THE TRAINING DATASET(AT INDEX=0)
x_train[0]

#SHOW IMAGE AS PICTURE
import matplotlib.pyplot as plt
img =plt.imshow(x_train[0])

#LABELLING THE IMAGE 
print('The label is :', y_train[0])

#CONVERTING THE LABELS INTO THE SET OF 10 NUMBERS TO INPUT INTO THE NUERAL NETWORK LATER
from keras.utils import to_categorical
y_train_one_hot= to_categorical(y_train)
y_test_one_hot= to_categorical(y_test)

#PRINTING THE NEW LABELS IN THE TRAINING DATASET
print(y_train_one_hot)

#TESTING  THE EXAMPLES OF THE NEW LABELS 
print('The new label is' , y_train_one_hot[0])

#NORMALIZE THE PIXELS IN THE IMAGE BETWEEN 0 AND 1
x_train=x_train/255
x_test=x_test/255

#BUILDING THE CNN
from keras.models import Sequential
from keras.layers import Dense ,Flatten, Conv2D ,MaxPooling2D

#CREATING THE ARCHITECTURE 
model=Sequential()

#CONVOLUTIONAL LAYER
model.add(Conv2D(32 ,(5,5), activation='relu' ,input_shape=(32,32,3)))

#MAXPOOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))

#CONVOLUTIONAL LAYER
model.add(Conv2D(32 ,(5,5), activation='relu'))

#MAXPOOLING LAYER
model.add(MaxPooling2D(pool_size=(2,2)))




#FLATTEN LAYER TO THE NEURONS
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#COMPILE THE MODEL 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#TRANING THE MODEL
hist=model.fit(x_train,y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3)

#GETTING THE MODEL ACCURACY 
model.evaluate(x_test,y_test_one_hot)[1]

#VISUALIZING THE MODEL ACCURACY 
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper left')
plt.show()

#VISUALIZING THE MODEL LOSS 
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='lower left')
plt.show()

#LOADING THE DATA 
from google.colab import files
uploaded=files.upload()
my_image=plt.imread('dog.jpeg')

#SHOWING THE UPLOADED IMAGE 
img=plt.imshow(my_image)

#RESIZING THE IMAGE 
from skimage.transform import resize
my_image_resized=resize(my_image,(32,32,3))
img=plt.imshow(my_image_resized)

#GETTING THE PROBABILITY FOR THE CLASSES IT BELONG
import numpy as np
probabilities=model.predict(np.array([my_image_resized,]))

#PRINTING THE PROBABILITIES
probabilities

number_to_class=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
index=np.argsort(probabilities[0,:])
print('Firstlikely class:', number_to_class[index[9]], '--probabilities:' , probabilities[0,index[9]])
print('Second likely class:', number_to_class[index[8]], '--probabilities:' , probabilities[0,index[8]])
print('Third likely class:', number_to_class[index[7]], '--probabilities:' , probabilities[0,index[7]])
print('Fourth likely class:', number_to_class[index[6]], '--probabilities:' , probabilities[0,index[6]])
print('Fifth likely class:', number_to_class[index[5]], '--probabilities:' , probabilities[0,index[5]])
print('Sixthlikely class:', number_to_class[index[4]], '--probabilities:' , probabilities[0,index[4]])
print('Seventh likely class:', number_to_class[index[3]], '--probabilities:' , probabilities[0,index[3]])
print('Eighth likely class:', number_to_class[index[2]], '--probabilities:' , probabilities[0,index[2]])
print('Nineth likely class:', number_to_class[index[1]], '--probabilities:' , probabilities[0,index[1]])
print('Tenth likely class:', number_to_class[index[0]], '--probabilities:' , probabilities[0,index[0]])

#Saving the model
model.save('Image Classifier')

#Loading the model 
from keras.models import load_model
model=load_model('Image Classifier')
