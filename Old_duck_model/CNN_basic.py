import glob
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 


duck_list = []

#Import all files from a specific folder
for filename in glob.glob('C:/Users/Noksuu/Desktop/small_stop_signs\*.jpg'): 
    im = Image.open(filename).convert("LA")
    im = np.array(im)
    im = (im[:,:,0]/255.0).flatten()
    duck_list.append(im)


for filename in glob.glob('C:/Users/Noksuu/Desktop/Small_r_ducks\*.jpg'): 
    im = Image.open(filename).convert("LA")
    im = np.array(im)
    im = (im[:,:,0]/255.0).flatten()
    duck_list.append(im)

#Reformat files
#It is important that the format is (Number of images, 100,100,1)
my_data2 = pd.DataFrame(duck_list)
x_train_img = (my_data2.iloc[:,:].values).astype('float32') 
x_train_img = x_train_img.reshape(x_train_img.shape[0], 100, 100, 1)

#Create the dependant variables
#signs 140 = 0 rubber ducks = 138
signs = np.zeros((140,1))
r_ducks = np.zeros((138,1))+1
y_train = np.concatenate((signs,r_ducks))

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
batch_size = 32
num_classes = 2
epochs = 50
input_shape = (100, 100,1)

#One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train, x_test, y_train, y_test = train_test_split(x_train_img, y_train, test_size = 0.20, random_state=123)

#Setting up a model
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Dropout(0.25))
model.add(Conv2D(10, kernel_size = (3, 3)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))


model.summary()

#Compile it
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


#This will run the model itself.
res = model.fit(x_train,y_train, batch_size = batch_size, epochs = epochs)

#Evaluate model
model.evaluate(x_test, y_test)

print(res.history.keys())
accuracy = res.history['acc']
loss = res.history['loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.title('Training')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.title('Training')
plt.legend()
plt.show()

#Save model
model.save("Duck_Sign_model")




