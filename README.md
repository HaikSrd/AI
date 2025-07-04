# Neural Network
The NN.py file contains a NeuralNetwork class that takes: <br>
- dataset inputs <br>
- dataset values <br>
- size of the output <br>
- number of hidden layers <br>
- number of neurons per each hidden layer (example = [128, 256, 128]) <br>
- epochs <br>
- learning rate <br>

To train a model you must call the `train` function after initiating an instance of the class. <br>
A pre trained model with 3 hidden layers on the mnist dataset is also in the project folder with an accuracy of 99.81%. <br> <br>
\* Be sure to save the file as an .npz file so you dont lose the model. (notebook is recommended) <br>
\* To evaluate new images with a trained model you can use the `forward_pass_hidden` function with some minor and simple tweaks

# Edge detection
the edge_detection.py file conatains some function that do the task of edge detection.
## function explainations:
### Kernel:
used to generate gaussian kernel matrix that follow the formula below:
### $G(x, y) = \frac{1}{2\pi\sigma^2} \cdot e^{-\frac{x^2 + y^2}{2\sigma^2}}$

## Grayscale:
simply takes the values and uses certain weights to turn RGB into grayscale

## Blur:
uses the kernel function and blurs the image using convolution

## Sobel Filter:
uses the two matrices below to find parts of the image that have the biggest difference in value

$$ \begin{bmatrix} 
   -1 & 0 & 1 \\
   -2 & 0 & 2 \\
   -1 & 0 & 1 \\
   \end{bmatrix}       
 \begin{bmatrix} 
   -1 & -2 & -1 \\
    0 & 0 & 0 \\
    1 & 2 & 1 \\
   \end{bmatrix} $$

## finding_components:
finds all the closed loops in the filtered image

## area:
calculates the area of a given closed loop

## EdgeDetection:
uses the functions to find the biggest area in the image and returns it

![My Image](Images/edge_detection.png)

