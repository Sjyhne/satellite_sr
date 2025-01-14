# Satellite SR
Satellite super resolution using LR satellite images by learning the transformations


## Main idea

The main idea is to overfit/optimize a Coordinate-based MLP to learn the sub-pixel transformations in the LR images from a HR image and try to upsample it. This will be a self-supervised approach.

So in this context, the input will be x, y

The flow will be:
1. (x, y) --> MLP --> (x', y') --> apply transformation? --> Pooling --> 

So you predict the RGB values for the HR image, and then you (apply the transformation? and) pool the images to get the LR image.

