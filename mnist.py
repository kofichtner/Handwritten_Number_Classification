import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
import numpy as np
import os
import random
import seaborn as sns
sns.set_theme()

DATADIR = './'
IMAGES = 'images.npy'
LABELS = 'labels.npy'

def create_training_data():
    training_data = []
    images = np.load(os.path.join(DATADIR,IMAGES))
    labels = np.load(os.path.join(DATADIR,LABELS))
    for i,label in enumerate(labels):
        training_data.append([images[i], label])
    return training_data

def split_data(training_data):
    train_data = []
    validation_data = []
    test_data = []
    for i in range(0,10):
        i_data = []
        for features,label in training_data:
            if label == i:
                i_data.append([features,label])
        random.shuffle(i_data)
        num_samples = len(i_data)
        test_split = round(num_samples*0.6)
        validation_split = round(num_samples*.15)
        train_data+= i_data[0:test_split]
        validation_data += i_data[test_split:test_split+validation_split]
        test_data+=i_data[test_split+validation_split:len(i_data)]
    return train_data, validation_data, test_data

def reshape_data(data):
    X = []
    Y = []
    for features, label in data:
        X.append(features.reshape(28*28, ))
        Y.append(tf.keras.utils.to_categorical(label, num_classes=10, dtype="float32"))
    X = np.asarray(X)/255.0
    Y = np.asarray(Y)
    return X,Y
training_data = create_training_data()
train_data, validation_data, test_data = split_data(training_data)
print('data lengths: ', len(train_data), len(validation_data), len(test_data))
x_train, y_train = reshape_data(train_data)
x_val, y_val = reshape_data(validation_data)
x_test, y_test = reshape_data(test_data)

# Model Template
model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # firstlayer
model.add(Activation('relu'))
#### Fill in Model Here## TODO add model layers
model.add(Dense(2048, activation=tf.nn.relu))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=100, batch_size=64)
# Report Results
print(history.history)
# model.predict()
score = model.evaluate(x_test, y_test, verbose=2)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

from matplotlib import pyplot as plt
plt.clf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predicted_classes = np.argmax(model.predict(x_test), axis=-1)
incorrect_indices = []
correct_indices = []
confusion_data = np.zeros([10,10])
for i, predicted_class in enumerate(predicted_classes):
    if predicted_class != np.argmax(y_test[i]):
        incorrect_indices.append(i)
    else:
        correct_indices.append(i)
    confusion_data[predicted_class][np.argmax(y_test[i])]+=1

print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")
#plot incorrectly classified images
plt.rcParams['figure.figsize'] = (10,8)
# plt.figure()
# random.shuffle(incorrect_indices)
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Truth: {}".format(predicted_classes[incorrect], np.argmax(y_test[incorrect])))
    plt.xticks([])
    plt.yticks([])
plt.show()
heat_map = sns.heatmap(confusion_data, cmap='Blues',annot=True, linewidths=0, fmt='g')
# plt.tick_params(axis='both', which='major', labelbottom = False, bottom=False, top = False, labeltop=True)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.title("Confusion Matrix")
plt.show()
