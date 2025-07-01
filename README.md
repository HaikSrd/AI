# Edge detection
the edge_detection.py file conatains some function that do the task of edge detection.
## function explainations:
### Kernel:
used to generate gaussian kernel matrix that follow the formula below:

$$
G(x, y) = \frac{1}{2\pi\sigma^2} \cdot e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$
where:
- \( x, y \) are the pixel distances from the center of the kernel
- \(\sigma) is the standard deviation (`dev` in the code)

