# Dog Breed Recognition with Convolutional Neural Network.

<div align="center">

  <img src="https://raw.githubusercontent.com/UrkoRegueiro/Dog_Breed_Recognition/main/images/cnn.png" alt="CNN" width="30%">
  
</div>

## Technologies Used

**Language:** Python.

**Libraries:** numpy, pandas, matplotlib, seaborn, joblib, tensorflow, keras, sklearn, cv2, PIL

------------

<h2>
  
To view the detailed version of the current project, please refer to the following [notebook](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/breed_recognition.ipynb).

<h2>
  
------------

## 1. **Introduction**

The application of convolutional neural networks has proven to be effective in solving complex problems related to visual recognition. In this context, this project focuses on the recognition of four dog breeds through the use of advanced deep learning techniques.

The aim of this project is to develop two image recognition models capable of accurately identifying the selected dog breeds. The first model will be a convolutional neural network built from scratch, and the second will involve the utilization of a pre-trained convolutional neural network, specifically the Xception network.

The chosen set of dog breeds will serve as a starting point for training and validating the neural networks, allowing us to explore the models' ability to discriminate between specific and subtle features that distinguish each breed.

The project workflow covers the collection and processing of an image dataset, the design and training of the convolutional neural networks, and finally, the evaluation of the two resulting models by comparing which one provides a more accurate outcome.

</div>

## 2. **Data and packages importation**<br>
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.1. Packages

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The packages and functions used throughout the analysis and modeling process are imported from a [script](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/custom_functions.ipynb), making this option suitable for a presentation as clean as possible.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.2. Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The working dataset has been provided by HACKABOSS.
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.2.1. Train Data
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2.2.2. Test Data

## 3. **Exploratory Data Analysis**<br>

The total number of images in both train and test data was the following:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/num_images.png)

</div>

If we take a look at the different number of dog breeds we can see the following:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/dog_breeds.png)

</div>

Let's take a look now at an image of each breed:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/img_breeds.png)

</div>

Finally let's plot an histogram of the dimensions of our images:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/img_dims.png)

</div>

## 4. **Image Processing**<br>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.1. X Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; I's decided to use a size of 224x224 for the images.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.2. y Data

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this subsection y data is encoded.

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.3. Data Augmentation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; For each image, we will generate 8 more by applying rotation, zooming, and flipping them. The result is the shown below:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/aug_data.png)

</div>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4.4. Data Preparation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this subsection we trasformed y data to categorical and shuffled the X train data.

## 5. **Model Selection**

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.1. Convolutional Neural Network from scratch

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We can see the architecture of our CNN used below:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/cnn_arch.png)

</div>

The results obtained in this model were the following:

<div align="center">
  
| Accuracy |
|----------|
| 0.4833   |

</div>

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/cnn_results.png)

</div>

### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 5.2. Xception CNN

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The architecture of the Xception model used is shown below:

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/xception_architecture.png)

</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; It works as follow:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; First the data goes through the entry flow, then through the middle flow which is repeated eight times, and finally through the exit flow. Note that all Convolution and SeparableConvolution layers are followed by batch normalization.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In our case we used images with an input size of 224x224.

## 6. **Final Results**<br>

Among the two trained neural networks, the Xception neural network significantly outperforms the other in terms of accuracy.

We can observe in the confusion matrix below that the two breeds with the highest accuracy are "Doberman" and "ShihTzu," with one misclassification each:

In the case of the "Doberman" breed, one misprediction occurred, mistakenly classifying it as the "Labrador" breed.

For the "ShihTzu" breed, there was one misprediction, where the model incorrectly classified the "Yorkshire" breed as "ShihTzu."

However, for the "Labrador" and "ShihTzu" breeds, it's obtained a lower accuracy, showing a few confusions with the rest of the breeds.

Despite these small errors, the results are satisfactory, been a good starting point for future implementations and improvements. For instance, we could consider adding more dog breeds, increasing the diversity of images in the data augmentation section, and experimenting with fully connected layers after the Xception model to fine-tune the network to our dataset.

<div align="center">

![](https://github.com/UrkoRegueiro/Dog_Breed_Recognition/blob/main/images/final_results.png)

</div>





