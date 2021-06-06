# VGG-classification
This model is implementing the architecture and training techniques used in VGG paper. We used CIFAR10 dataset by keras.<br />The preprocessing :<br />
A) Dividing all the pixels of images by 225<br />
B) Converting y_train(labels) into float type.<br />
C) Using to_categorical() on training labels to use categorical loss as the loss function while training.<br />

## Architecture 
![alt text](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)<br />
This model is made for 224x224x3 images, but in this model, it is implemented on 32x32x3 images, so the learable parameters changed from 134 million to 43 million.
The following are the output values of all the layers used:<br/>
Model: "sequential"
_________________________________________________________________<br/>
Layer (type)                 Output Shape              Param #   <br/>
=================================================================<br/>
conv2d (Conv2D)              (None, 32, 32, 64)        1792      <br/>
_________________________________________________________________<br/>
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928     <br/>
_________________________________________________________________<br/>
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         <br/>
_________________________________________________________________<br/>
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856     <br/>
_________________________________________________________________<br/>
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584    <br/>
_________________________________________________________________<br/>
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         <br/>
_________________________________________________________________<br/>
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168    <br/>
_________________________________________________________________<br/>
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080    <br/>
_________________________________________________________________<br/>
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080    <br/>
_________________________________________________________________<br/>
conv2d_7 (Conv2D)            (None, 8, 8, 256)         590080    <br/>
_________________________________________________________________<br/>
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         <br/>
_________________________________________________________________<br/>
conv2d_8 (Conv2D)            (None, 4, 4, 512)         1180160   <br/>
_________________________________________________________________<br/>
conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359808   <br/>
_________________________________________________________________<br/>
conv2d_10 (Conv2D)           (None, 4, 4, 512)         2359808   <br/>
_________________________________________________________________<br/>
conv2d_11 (Conv2D)           (None, 4, 4, 512)         2359808   <br/>
_________________________________________________________________<br/>
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         <br/>
_________________________________________________________________<br/>
conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359808   <br/>
_________________________________________________________________<br/>
conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808   <br/>
_________________________________________________________________<br/>
conv2d_14 (Conv2D)           (None, 2, 2, 512)         2359808   <br/>
_________________________________________________________________<br/>
conv2d_15 (Conv2D)           (None, 2, 2, 512)         2359808   <br/>
_________________________________________________________________<br/>
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         <br/>
_________________________________________________________________<br/>
flatten (Flatten)            (None, 512)               0         <br/>
_________________________________________________________________<br/>
batch_normalization (BatchNo (None, 512)               2048      <br/>
_________________________________________________________________<br/>
dense (Dense)                (None, 4096)              2101248   <br/>
_________________________________________________________________<br/>
dropout (Dropout)            (None, 4096)              0         <br/>
_________________________________________________________________<br/>
batch_normalization_1 (Batch (None, 4096)              16384     <br/>
_________________________________________________________________<br/>
dense_1 (Dense)              (None, 4096)              16781312  <br/>
_________________________________________________________________<br/>
dropout_1 (Dropout)          (None, 4096)              0         <br/>
_________________________________________________________________<br/>
batch_normalization_2 (Batch (None, 4096)              16384     <br/>
_________________________________________________________________<br/>
dense_2 (Dense)              (None, 1000)              4097000   <br/>
_________________________________________________________________<br/>
batch_normalization_3 (Batch (None, 1000)              4000      <br/>
_________________________________________________________________<br/>
dense_3 (Dense)              (None, 10)                10010     <br/>
_________________________________________________________________<br/>
activation (Activation)      (None, 10)                0         <br/>
_________________________________________________________________<br/>

So in the end, we get a softmax layer with 10 units of output.
