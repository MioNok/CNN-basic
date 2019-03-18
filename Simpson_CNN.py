import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from sklearn.metrics import confusion_matrix
import itertools



#The characters we will use for training.
char_list = ["chief_wiggum","homer_simpson","marge_simpson","lisa_simpson","principal_skinner",
             "charles_montgomery_burns",
             "moe_szyslak","bart_simpson","krusty_the_clown","milhouse_van_houten"]

#The height and width for the final picture for learning.
picture_h_w = 64

#This list will contain the dependant variables automatically numbered, we will one hot encode them later
num_of_chars = []

# all the image numpy arrays will append here, later we will reshape this to the right format.
simpson_list = []

char_counter = 0
#Grab a coffee.
for char_name in char_list:
    for filename in glob.glob("C:/Users/Noksuu/Desktop/pydir/simpsons_dataset/"+char_name+"\*.jpg"): 
        im = Image.open(filename)
        im = im.resize((picture_h_w,picture_h_w),Image.ANTIALIAS)
        im = np.array(im)
        im = im.astype("float32")/255
        simpson_list.append(im)
        num_of_chars.append(char_counter)        
    char_counter = char_counter +1
    
    
#A better way of reformatting my files than the on Ducks CNN.. The more you know
#It is important that the format is (Number of images, 100,100,3)
x_train_img = np.array(simpson_list)


#Somehow I need this to GPU accelearting to work, might be optional for you

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

batch_size = 32
num_classes = 10
epochs = 75
input_shape = (picture_h_w, picture_h_w, 3)

#One hot encoding
y_train = keras.utils.to_categorical(num_of_chars, num_classes)

#Train and test split
x_train, x_test, y_train, y_test = train_test_split(x_train_img, y_train, test_size = 0.20, random_state=123)

#Train and val split
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size = 0.10, random_state=123)

#Setting up a model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))
model.add(Conv2D(64, kernel_size = (3, 3)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size = (4, 4)))
model.add(Conv2D(128, kernel_size = (4, 4)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.3))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))


#Defining the model
model.compile(loss="categorical_crossentropy",
              optimizer="RMSprop",
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

res = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                          epochs = epochs, validation_data = (x_val,y_val),
                          verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                          , callbacks=[learning_rate_reduction])


#Evaluate model
model.evaluate(x_test, y_test)

print(res.history.keys())
accuracy = res.history['val_acc']
val_accuracy = res.history['acc']
loss = res.history['loss']
val_loss = res.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r-', label='Validation accuracy')
plt.title('Training')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r-',label='Validation loss')
plt.title('Training')
plt.legend()
plt.show()

#Save model
model.save("Simpsons_model")

y_pred = model.predict(x_test)

y_pred = y_pred.round()




#This confusion matrix is taken from the SKLEARN website, minor modifications has been done.
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_val, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, char_list)




