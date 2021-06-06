# VGG-classification
This model is implementing the architecture and training techniques used in VGG paper. We used CIFAR10 dataset by keras.<br />The preprocessing :<br />
A) Dividing all the pixels of images by 225<br />
B) Converting y_train(labels) into float type.<br />
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.<br />

## Architecture 
![alt text](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)<br />
This model is made for 224x224x3 images, but in this model, it is implemented on 32x32x3 images, so the learable parameters changed from 134 million to 43 million
