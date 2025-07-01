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
   \end{bmatrix} $$
$$ \begin{bmatrix} 
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


