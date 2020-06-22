#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras, keras.backend as K
from keras.layers import Dense, MaxPooling2D, Input, Conv2D, Conv2DTranspose, Lambda
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.models import Sequential


# In[3]:


def modelTransform():

  inp = Input(shape=(512, 512, 6))

  # Encoder part
  model = Conv2D(32,[3,3],padding='same',activation='elu')(inp)
  model = MaxPooling2D([2,2])(model)
  model = Conv2D(64,[3,3],padding='same',activation='elu')(model)
  model = MaxPooling2D([2,2])(model)
  model = Conv2D(128,[3,3],padding='same',activation='elu')(model)
  model = MaxPooling2D([2,2])(model)
  model = Conv2D(256,[3,3],padding='same',activation='elu')(model)

  # Decode part

  model = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same')(model)
  model = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same')(model)
  model = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same')(model)
  model = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=1, activation='elu', padding='same', name = 'genOutput')(model)
  

  model = Model(inp, model)

  return model


# In[4]:


generator = modelTransform()


# In[5]:


generator.summary()


# In[6]:


def completeModel(generator):


  newModel = Sequential()


  for layer in generator.layers:
    newModel.add(layer)

  model = VGG16(include_top =False, input_shape = (512, 512, 3), weights='imagenet')

  for layer in model.layers[1:]:
    layer.trainable = False
    newModel.add(layer)

  model = Model(newModel.input, newModel.output)

  return model


# In[7]:


generator.output_shape


# In[8]:


model = completeModel(generator)
model.summary()


# In[9]:


import numpy as np
import pickle as pk
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input
import os
import keras.backend as K


# In[10]:


# Static info for experiment

path = './fastImages'

# Defining layer to find the correlational costs

contextLayer = 'block4_conv2'

styleLayers =  ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

# Model 

vgg = VGG16(include_top=False, weights='imagenet',input_shape=(512,512,3))


# In[11]:


# Util functions for creating dataset and preprocessing

def loadImages(addr):
  contImg = load_img(path=addr, target_size=(512, 512))
  contImg = img_to_array(contImg)
  contImg = np.expand_dims(contImg, axis=0)
  contImg = preprocess_input(contImg)

  return contImg

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((0,1, 2))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


getStyleActivations = K.function([vgg.input], [vgg.get_layer(layer).output for layer in styleLayers])
getContextActivation = K.function([vgg.input], [vgg.get_layer(contextLayer).output])


# In[14]:


# Creating dataset for our experiment

# contextImageFiles = list(map(lambda x: os.path.join(path, os.path.join('context', x)), os.listdir(os.path.join(path, 'context'))))

# styleImageFiles = list(map(lambda x: os.path.join(path, os.path.join('style', x)), os.listdir(os.path.join(path, 'style'))))


# contextImages = list(map(loadImages,contextImageFiles))
# styleImages = list(map(loadImages,styleImageFiles))


# In[15]:


# Getting respective activations

# contextActivations = list(map(lambda x: getContextActivation(x)[0], contextImages))
# styleActivations = list(map(getStyleActivations, styleImages))


# In[16]:


# contextData = list(zip(contextImages, contextActivations))
# styleData = list(zip(styleImages, styleActivations))


# In[17]:


# Saving generated data for future use
# Combined data is more than 1GB (only for 5-6 images) it better to store, than regenerate every time (:

# pk_out = open(os.path.join(path, 'contextDataset.pickle'), 'wb')
# pk.dump(contextData, pk_out)
# pk_out.close()

# pk_out = open(os.path.join(path, 'styleDataset.pickle'), 'wb')
# pk.dump(styleData, pk_out)
# pk_out.close()


# In[12]:


pk_in = open(os.path.join(path, 'contextDataset.pickle'), 'rb')
contextData = pk.load(pk_in)
pk_in.close()

pk_in = open(os.path.join(path, 'styleDataset.pickle'), 'rb')
styleData = pk.load(pk_in)
pk_in.close()


# In[13]:


def concat(context, style):
  return np.concatenate((context, style), axis = -1)


# In[14]:


# Mix and match images to generate active dataset


concatImages = []
activationOutputs = {block :[] for block in styleLayers}
activationOutputs[contextLayer] = list()

for contextItem in contextData:
  for styleItem in styleData:
    concatImages.append(concat(contextItem[0][0], styleItem[0][0]))

    activationOutputs[contextLayer].append(contextItem[1][0])

    for block, styleActBlock in zip(styleLayers, styleItem[1]):
      activationOutputs[block].append(styleActBlock[0])

activationOutputs = {block : np.array(act) for block, act in activationOutputs.items()}

concatImages = np.array(concatImages)


# In[15]:


activationOutputs.keys()


# In[16]:


blockShapes = { block: act.shape[1:] for block, act in activationOutputs.items()}

blockShapes


# In[17]:


# Workaround to convert sequential model to functional

from keras import layers, models

input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
prev_layer = input_layer
for layer in model.layers[1:]:
    layer._inbound_nodes = []
    prev_layer = layer(prev_layer)

funcModel = models.Model([input_layer], [prev_layer])


# In[18]:


newModel = Model([funcModel.input], [funcModel.get_layer(contextLayer).output] + [funcModel.get_layer(block).output for block in styleLayers])


# In[19]:


newModel.summary()


# In[20]:


def contextLoss(y_true, y_pred):
  return K.sum(K.square(y_true - y_pred))


def gram_matrix(mat):
    mat = K.reshape(mat,(K.shape(mat)[1]*K.shape(mat)[2],-1))
    return K.dot(K.transpose(mat), mat)

def styleLoss1(y_true, y_pred):
  sp = blockShapes['block1_conv2']
  return K.sum( K.square(gram_matrix(y_true) - gram_matrix(y_pred)))/(20*(sp[0]*sp[1]*sp[2])**2)

def styleLoss2(y_true, y_pred):
  sp = blockShapes['block2_conv2']
  return K.sum( K.square(gram_matrix(y_true) - gram_matrix(y_pred)))/(20*(sp[0]*sp[1]*sp[2])**2)


def styleLoss3(y_true, y_pred):
  sp = blockShapes['block3_conv3']
  return K.sum( K.square(gram_matrix(y_true) - gram_matrix(y_pred)))/(20*(sp[0]*sp[1]*sp[2])**2)


def styleLoss4(y_true, y_pred):
  sp = blockShapes['block4_conv3']
  return K.sum( K.square(gram_matrix(y_true) - gram_matrix(y_pred)))/(20*(sp[0]*sp[1]*sp[2])**2)


def styleLoss5(y_true, y_pred):
  sp = blockShapes['block5_conv3']
  return K.sum( K.square(gram_matrix(y_true) - gram_matrix(y_pred)))/(20*(sp[0]*sp[1]*sp[2])**2)


  


# In[99]:





losses = {
    'block4_conv2' : contextLoss,
    'block1_conv2' : styleLoss1,
    'block2_conv2' : styleLoss2,
    'block3_conv3' : styleLoss3,
    'block4_conv3' : styleLoss4,
    'block5_conv3' : styleLoss5,
}


lossWeights = {
    
    'block4_conv2' : 0.005,
    'block1_conv2' : 4,
    'block2_conv2' : 4,
    'block3_conv3' : 4,
    'block4_conv3' : 4,
    'block5_conv3' : 4,
    
}




# In[84]:


expModel = keras.models.clone_model(newModel)


# In[85]:


# First I will train autoencoder part for faster convergence 

autoEncoder = Model(expModel.input, expModel.get_layer('genOutput').output)


# In[86]:


autoEncoder.compile(
    optimizer = 'adam',
    loss = 'mse',
    metrics = ['mse']
)


# In[105]:


autoEncoder.fit(x = concatImages, y = concatImages[:,:,:,:3], epochs = 30, batch_size = 8)


# In[142]:


# Check if autoencoder is trained properly


ind = 2

npimg = np.array([concatImages[ind]])
img = autoEncoder.predict(npimg)[0]
img = cv2.cvtColor(deprocess_image(img), cv2.COLOR_BGR2RGB)

plt.imshow(img)


# In[114]:


#  Now training it completely

opt = keras.optimizers.Adam(learning_rate=0.0001)


# In[115]:


#  this step might change with respect to the version of keras

expModel.compile(
    optimizer = opt,
    loss = losses,
    loss_weights = lossWeights,
    metrics =['mse']
)


# In[135]:


expModel.fit(x = concatImages, y = activationOutputs, epochs = 100,batch_size=8)


# In[136]:


genModel = Model(expModel.input, expModel.get_layer('genOutput').output)


# In[137]:


genModel.summary()


# In[138]:


from matplotlib import pyplot as plt
import cv2


# In[141]:


# See results after full training 

ind = 13

npimg = np.array([concatImages[ind]])
img = genModel.predict(npimg)[0]
img = cv2.cvtColor(deprocess_image(img), cv2.COLOR_BGR2RGB)

plt.imshow( img )



# In[ ]:




