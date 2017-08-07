# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:43:49 2017

@author: Rafael Espericueta
USEFUL:  https://github.com/fchollet/keras/issues/3465
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout
from keras import applications
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.core import Activation   # , Reshape
from keras.layers import Add

import os
package_directory = os.path.dirname(os.path.abspath('FCN_VGG16_bug.py'))
print package_directory

# nb_classes = 4
input_shape = (560, 840, 3)   #360, 640

#base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
base_model = applications.VGG16(include_top = False, 
                                input_shape = input_shape,
                                weights = 'imagenet')
x = base_model.output

# Layer 19, 20
x = Conv2D(4096, (7, 7), 
           strides = (1, 1), 
           padding = 'valid', 
           activation = 'relu', 
           use_bias = True, #??
           kernel_initializer = 'glorot_uniform')(x)
x = Dropout(0.5)(x)

# Layer 21, 22
x = Conv2D(4096, (1, 1), 
           strides = (1, 1), 
           padding = 'valid', 
           activation = 'relu', 
           use_bias = True, #??, Dense
           kernel_initializer = 'glorot_uniform')(x)
x = Dropout(0.5)(x)

# Layer 23
x = Conv2D(filters = 4, 
           kernel_size = (1, 1), 
           strides = (1, 1), 
           padding = 'valid', 
           activation = 'relu', 
           use_bias = True, #??
           kernel_initializer = 'glorot_uniform')(x)  
           
# Layer 24
x = Conv2DTranspose(filters = 4, 
                    kernel_size = (4, 4), 
                    strides = (2, 2), 
                    padding = 'valid', 
                    activation = None, 
                    use_bias = False,
                    kernel_initializer = 'glorot_uniform')(x)  

# Layer 25   
p4 = base_model.layers[14].output
p4 = Conv2D(4, (1, 1),
           strides = (1, 1), 
           padding = 'valid', 
           activation = 'relu', 
           use_bias = True, 
           kernel_initializer = 'glorot_uniform')(p4)

# Layer 26
# Cropping is needed, but accessing the deconvolution shape isn't possible
# due to an outstanding Keras bug. Trying to add reveals its output shape
# to be (24, 42). So the output from Pool4 with shape (35, 52) needs to be cropped:
# upleft = ((35 - 24) / 2, (52 - 42) / 2) = (5, 5)
TL = ((35 - 24) / 2, (52 - 42) / 2)
BR = (35 - TL[0] - 24, 52 - TL[1] - 42)
p4c = Cropping2D(((TL[0], BR[0]), (TL[1], BR[1])))(p4)  # from block4_pool

# Layer 27
x = Add()([x, p4c])

# Layer 28
xc = Conv2DTranspose(4, (32, 32), 
           strides = (16, 16), 
           padding = 'valid', 
           activation = None, 
           use_bias = False,
           kernel_initializer = 'glorot_uniform')(x)  
# NOTE:  xc.get_shape() = (400, 688, 4), if that command worked
          
# Layer 29
#  needs to be same size as model.layers[0].input
TL = ((400 - 360) / 2, (688 - 640) / 2)
BR = (400 - TL[0] - 360, 688 - TL[1] - 640)
xc = Cropping2D(((TL[0], BR[0]), (TL[1], BR[1])))(xc) 
# NOTE:  xc.get_shape() = (360, 640, 4)

# Layer 30
x = Activation('softmax')(xc)

# Create graph for this model
model = Model(inputs = base_model.input, outputs = x)

# Freeze base layers
for l, layer in enumerate(model.layers):
    if l < 19:
        layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#
# Now to prepare the data.
#

def subtract_ave(A):
    # Subtract the means of each color channel from each pixel.
    # [110.776777, 116.093566, 120.824449]
    A -= np.array([111, 116, 121])   # B G R
    return A

image_datagen = ImageDataGenerator(preprocessing_function = subtract_ave)
mask_datagen = ImageDataGenerator()

image_dir = os.path.join(package_directory, 'images')
label_dir = os.path.join(package_directory, 'labels')

image_generator = image_datagen.flow_from_directory(
    image_dir,  
    class_mode = None,
    batch_size = 1,
    seed = 123)

mask_generator = mask_datagen.flow_from_directory(
    label_dir,  
    class_mode = None,
    batch_size = 1,
    seed = 123)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch = 1000,
    epochs = 100)
