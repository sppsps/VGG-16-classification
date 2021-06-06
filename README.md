# VGG-classification
This model is implementing the architecture and training techniques used in VGG paper. We used CIFAR10 dataset by keras. The preprocessing :
A) Dividing all the pixels of images by 225
B) Converting y_train(labels) into float type.
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.

## Architecture 

