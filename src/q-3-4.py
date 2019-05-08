#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt


# In[2]:


(X_train,y_train),(X_test,y_test) = cifar10.load_data()
print('X_train shape: ',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')


# In[ ]:


Y_train = np_utils.to_categorical(y_train,10)
Y_test = np_utils.to_categorical(y_test,10)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[4]:


model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


# In[5]:


model.compile(loss='categorical_crossentropy',
             optimizer=RMSprop(),
             metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=128,
         epochs=20,
         validation_split=0.2,
         verbose =1)


# In[7]:


score = model.evaluate(X_test,Y_test,
                       batch_size=128,
                       verbose=1)


# In[9]:


print(score)


# In[ ]:




