from numpy import *
# import cv2 as cv
from time import sleep
import os
import errno
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
import pickle

def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred



def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def read_data(folder):
    x_data_temp = []
    y_data_temp = []
    x_test_data_temp = []
    y_test_data_temp = []
    # We don't use numpy's vstack here as that would be wasteful, because every time you do a vstack, numpy would end
    # up copying the whole array to a new location. Hence we use a little trick to first store the data in a list and
    # then convert it to an numpy array

    for file in os.listdir(folder):
        if file.endswith(".meta") or file.endswith(".html"):
            print("Ignoring html and meta files")
        elif "test_batch" in file:
            # test data file detected. we are gonna load it separately
            test_data_temp = unpickle(folder + "/" + file)
            x_test_data_temp.append(test_data_temp[b'data'])
            y_test_data_temp.append(test_data_temp[b'labels'])
        else:
            temp_data = unpickle(folder + "/" + file)
            x_data_temp.append(temp_data[b'data'])
            y_data_temp.append(temp_data[b'labels'])
    x_data = array(x_data_temp)
    y_data = array(y_data_temp)
    x_test_data = array(x_test_data_temp)
    y_test_data = array(y_test_data_temp)
    return [x_data, y_data, x_test_data, y_test_data]


X_train_temp, y_train_temp, X_test_temp, y_test_temp = read_data("cifar-10-batches-py")

# At this time, since we converted from list to numpy array, there ia an extra dimension added to the array
# X_train_temp.shape = (6, 10000, 3072) and y_train_temp.shape = (6, 10000)
# In order to fix this, we will need to reshape the stack.

X_train_temp = X_train_temp.reshape(X_train_temp.shape[0] * X_train_temp.shape[1], X_train_temp.shape[2])
y_train_temp = y_train_temp.reshape(y_train_temp.shape[0] * y_train_temp.shape[1])

# Similarly for X_test_temp and y_test_data

X_test_temp = X_test_temp.reshape(X_test_temp.shape[0] * X_test_temp.shape[1], X_test_temp.shape[2])
y_test_temp = y_test_temp.reshape(y_test_temp.shape[0] * y_test_temp.shape[1])

print(X_train_temp.shape, X_train_temp.ndim, type(X_train_temp))
print(y_train_temp.shape, y_train_temp.ndim, type(y_train_temp))

print(X_test_temp.shape, X_test_temp.ndim, type(X_test_temp))
print(y_test_temp.shape, y_test_temp.ndim, type(y_test_temp))

# Now lets shuffle the data a bit with random state 4

X_train, y_train = shuffle(X_train_temp, y_train_temp, random_state=4)
X_test, y_test = shuffle(X_test_temp, y_test_temp, random_state=4)

# Splitting X and y in training and val data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=4)

# Keras Parameters
batch_size = 32
nb_classes = 10
nb_epoch = 20
img_rows, img_col = 32, 32
img_channels = 3
nb_filters = 32
nb_pool = 2
nb_conv = 3

# Now that our data has been shuffled and spitted,  lets reshape it and get it ready to be fed into our CCN model

X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
y_train = np_utils.to_categorical(y_train, nb_classes)

X_val = X_val.reshape(X_val.shape[0], 3, 32, 32)
y_val = np_utils.to_categorical(y_val, nb_classes)

X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
y_test = np_utils.to_categorical(y_test, nb_classes)

# Finally print shape of this data :

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

# Regularize the data
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_val /= 255
X_test /= 255

# Starting with Keras now

model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                 padding='valid',
                 activation='relu',
                 input_shape=(img_channels, img_rows, img_col),
                 data_format='channels_first',))

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Test this trained model on our test data
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score :", score[0])
print("Test accuracy: ", score[1])
print(model.predict_classes(X_test[1:50]))
print(y_test[1:50])


# Now lets save the model to disk
# serialize model to JSON

print("First deleting old models if they exists !!")
silent_remove("model.json")
silent_remove("model.h5")
silent_remove("whole_model.h5")

sleep(2)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
model.save("whole_model.h5")
print("Saved model to disk")

