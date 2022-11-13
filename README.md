# website
Assignment 2: Feedforward network using Tensorflow and keras
#Import Packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

------
#load the data #len shape array feature scaling array
mnist=tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

----
print(len(x_train))
print(len(x_test))

---
print(x_train.shape)
print(x_test.shape)
print(x_train[0])

---
plt.matshow(x_train[0])

---
#Feature Scaling
x_train=x_train/255
x_test=x_test/255 

----
print(x_train[0])

-----
#defining network architecture using keras
model=tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

-----
model.summary()

---
#train the model using sgd
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train, y_train, validation_data= (x_test,y_test),epochs=6 , verbose=2)

----
#evaluate the network
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=",test_loss)
print("Accuracy=",test_acc)

----
n=random.randint(0,9999)
plt.imshow(x_test[n])

---
#plot the training loss and accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Training Loss and Accuracy')
plt.xlabel("epochs")
plt.legend(["Accuracy","Training Loss"],loc= 'lowerleft')

---
