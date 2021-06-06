# VGG-classification
This model is implementing the architecture and training techniques used in VGG paper. We used CIFAR10 dataset by keras.<br />The preprocessing :<br />
A) Dividing all the pixels of images by 225<br />
B) Converting y_train(labels) into float type.<br />
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.<br />

## Architecture 

