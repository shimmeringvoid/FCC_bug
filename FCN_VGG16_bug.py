# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:43:49 2017

@author: 
USEFUL:  https://github.com/fchollet/keras/issues/3465
"""

import numpy as np
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout
from keras import applications
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.layers.core import Activation   # , Reshape
from keras.layers import Add
from keras.utils import to_categorical
import os
from os.path import join
import cv2

nb_classes = 4
input_shape = (560, 840, 3)   #360, 640

base_model = applications.VGG16(include_top = False, 
                                input_shape = input_shape,
                                weights = 'imagenet')
x = base_model.output

# Layer 19, 20
x = Conv2D(filters = 4096, 
           kernel_size = 7, 
           strides = 1, 
           padding = 'valid', 
           activation = 'relu', 
           use_bias = True)(x)
           
x = Dropout(0.5)(x)

# Layer 21, 22
x = Conv2D(filters = 4096, 
           kernel_size = 1, 
           strides = 1,
           padding = 'valid',
           activation = 'relu', 
           use_bias = True)(x)
           
x = Dropout(0.5)(x)

# Layer 23
x = Conv2D(filters = 4, 
           kernel_size = 1, 
           strides = 1, 
           padding = 'valid', 
           activation = None, 
           use_bias = True, 
           kernel_initializer = 'glorot_uniform')(x)

# Layer 24
xc = Conv2DTranspose(filters = 4, 
                     kernel_size = 4, 
                     strides = 2, 
                     padding = 'valid', 
                     activation = None, 
                     use_bias = False)(x)  

# Layer 25   
p4 = base_model.layers[14].output
p4 = Conv2D(filters = 4, 
            kernel_size = 1,
            strides = 1, 
            padding = 'valid',
            activation = 'relu',
            use_bias = True)(p4)

# Layer 26
'''
 Cropping is needed, but accessing the deconvolution shape isn't possible,
 via xc.get_shape(), which outputs (None, None, ...) due to an outstanding Keras bug. 
 A cludge to find this shape:  Try to add, Add()([x, xc]). 
 Since these shapes don't match, the resulting error message
 reveals their shapes, in this case, (11, 20, 4) and (24, 42, 4), respectively.
 p4.get_shape() outputs (35, 52), hence this output from Pool4 needs to be cropped
 before it can be added to xc which has shape (24, 42).
 '''
TL = ((35 - 24) / 2, (52 - 42) / 2)
BR = (35 - TL[0] - 24, 52 - TL[1] - 42)
p4c = Cropping2D(((TL[0], BR[0]), (TL[1], BR[1])))(p4)  # from block4_pool
# Now the shape is correct:  p4c.get_shape() ---> (24, 42, 4)

# Layer 27
x = Add()([xc, p4c])  # this has shape (24, 42, 4)

# Layer 28
xc = Conv2DTranspose(4, (32, 32), 
           strides = (16, 16), 
           padding = 'valid', 
           activation = None,
           use_bias = False)(x)  
# NOTE:  xc.get_shape() = (400, 688, 4), if that command worked,
#        as revealed by  Add()[x, xc]
           
# Layer 29
# The original training images were [360, 640], but were padded by 100,
# prior to setting up this network. So although the input images were
# of dimension [560, 840], only the middle [360, 640] pixels are relevant.
# Thus we crop xc to be dimension [360, 640]:
TL = ((400 - 360) / 2, (688 - 640) / 2)
BR = (400 - TL[0] - 360, 688 - TL[1] - 640)
xc = Cropping2D(((TL[0], BR[0]), (TL[1], BR[1])))(xc) 
# NOTE:  xc.get_shape() = (360, 640, 4), 
# again revealed by  Add()([x, xc]).

# Layer 30
net_output = Activation('softmax')(xc)
#net_output = softmax(axis = 3)(xc)
# net_output.get_shape() = (360, 640, 4),
# again revealed by Add()([x, net_output]).

# Create the graph of this model.
model = Model(inputs = base_model.input, outputs = net_output)

# Freeze the base layers.
for l, layer in enumerate(model.layers):
    if l < 19:
        layer.trainable = False

# Compile the model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Let's look at it!
#print model.summary()

#
# Now to prepare the data.
#

# These globals are used in generate_image_mask_pairs() 
path  = '/home/icg/Martin/train_data_graz/'
images = os.listdir( join(path, 'images/images_rect_r640x360') )
masks = os.listdir( join(path, 'labels/labels_rect_r640x360') )
nb_imgs = len(images)
# for one-hot encoding of mask values
nmb_msk_rows, nmb_msk_cols = cv2.imread(join(path, 'labels/labels_rect_r640x360', masks[0]), 0).shape
mskv = np.empty((nmb_msk_rows, nmb_msk_cols, 4))

def generate_image_mask_pairs(): 
    while 1:
        # pick a random image and its associated mask
        i = np.random.randint(nb_imgs)
        img = cv2.imread(join(path, 'images/images_rect_r640x360', images[i])).astype(np.int8)
        # Subtract the means of each color channel from each pixel.
        # [110.776777, 116.093566, 120.824449]
        img -= np.array([111, 116, 121])   # B G R
        msk = cv2.imread(join(path, 'labels/labels_rect_r640x360', masks[i]), 0)
        # one-hot encode the mask values
        for i in range(nmb_msk_rows):
            for j in range(nmb_msk_cols):
                mskv[i, j, :] = to_categorical(msk[i, j], 4)
        yield np.expand_dims(img, axis=0), np.expand_dims(mskv, axis=0)
        
gen_im_msk = generate_image_mask_pairs()

model.fit_generator(gen_im_msk, epochs=15, steps_per_epoch=100)

# Save the trained model.
model.save('/home/icg/Martin/keras_model.h5')
# To load this compiled model.
#model = keras.models.load_model('/home/icg/Martin/keras_model.h5') 

'''
def subtract_ave(A):
    # Subtract the means of each color channel from each pixel.
    # [110.776777, 116.093566, 120.824449]
    A -= np.array([111, 116, 121])   # B G R
    return A

image_datagen = ImageDataGenerator(preprocessing_function = subtract_ave)
mask_datagen = ImageDataGenerator()

image_generator = image_datagen.flow_from_directory(
    '/home/icg/Martin/train_data_graz/images',  #_rect_r640x360
    class_mode = 'categorical',
    target_size = input_shape[:2],  # (560, 840) = input image dimensions
    batch_size = 1,
    seed = 123)

mask_generator = mask_datagen.flow_from_directory(
    '/home/icg/Martin/train_data_graz/labels',  #_rect_r640x360
    class_mode = 'categorical',
    target_size = (360, 640),  # (360, 640) = input mask dimensions
    batch_size = 1,
    seed = 123)

# Combine generators into one which yields both images and masks.
# NOTE: For Python3 just use zip.
from itertools import izip
train_generator = izip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 10)
'''
