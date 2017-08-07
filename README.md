# FCC_bug
Repository to find a bug...

This code is for image segmentation, though here I've included only 10 image files and 10 masks. 

The problem seems to be the command:

    train_generator = zip(image_generator, mask_generator)

Upon excecuting this command, memory useage expands to the maximum possible, 
and then swapped memory increases to the max, and then everything freezes.

The problematic command was used in the Keras documentation I was attempting to emulate, 
on this page:  https://keras.io/preprocessing/image/ , 

with the  heading:  "Example of transforming images and masks together."
