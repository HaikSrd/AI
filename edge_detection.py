import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

'''
IMPORTANT:
    Must use matplot lib since different libraries import images in different coordinations, for
    example matplotlib reads images in [y,x] but cv2 does it in [x,y].
    Also, the images should be normalized (0-1) because different image types are saved in different ranges, 
    for example png is 0-1 but jpg is 0-255, so in order to process a jpg image you should /255
    
    dev -> lower dev (1.5) = sharper image (not suitbale for edge detection)
        -> higher dev (3) = smoother image (suitable for edge detection)
    
    size of kernel (size) -> size of the kernel matrix (bigger size = blurier image, must be an odd number)
    
    threshold -> higher threshold (200) saves less detail from the image
              -> lower threshold (50) saves more detail from the image
              *lower is generally better (100-150)
'''



# Generates the Gaussian Kernel which is used to blur the image
def Kernel(dev: int, size: int):
    #making the kernel matrix
    kernel = np.zeros((size,size))
    for i in range(int(-size/2), int(size/2) + 1):
        for j in range(int(-size/2), int(size/2) + 1):
            value = np.power(np.e, -((i**2 + j**2)/(2*(dev**2))))*(1/(2*np.pi*(dev**2)))
            kernel[i + size // 2][j + size // 2] = value
    kernel /= np.sum(kernel)
    return kernel

#Simple function to turn RGB images into grayscale
def grayscale(pixels):
    gray = np.zeros((len(pixels),len(pixels[0])))
    if type(pixels[0][0]) != np.ndarray:
        return pixels
    for x in range(len(pixels)):
        for y in range(len(pixels[0])):

            gray[x][y] = pixels[x][y][0]*0.2989 + pixels[x][y][1]*0.5870 + pixels[x][y][2]*0.1140

    return gray

#Uses convolution and a Gaussian Kernel to blur the image
def blur(pixels, dev, size):
    pixels = grayscale(pixels)
    kernel = Kernel(dev,size)
    blurred = np.zeros((pixels.shape))
    pixels = np.pad(pixels, pad_width=int(len(kernel)/2), mode='constant', constant_values=0)
    windows = sliding_window_view(pixels, (len(kernel),len(kernel)))
    for w in range(len(blurred)):
        for k in range(len(blurred[0])):
            blurred[w][k] = np.sum(windows[w][k] * kernel)
    return (blurred * 255).astype('uint8')

#Finding closed loops inside the image
def sobel_filter(pixels, dev: float, size: int, threshold: int):
    pixels = blur(pixels, dev, size)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    pad = 1
    padded = np.pad(pixels, pad, mode='constant', constant_values=0)
    windows = sliding_window_view(padded, (3, 3))

    gx = np.tensordot(windows, sobel_x, axes=((2, 3), (0, 1)))
    gy = np.tensordot(windows, sobel_y, axes=((2, 3), (0, 1)))

    mag = np.sqrt(gx**2 + gy**2)
    return np.where(mag >= threshold, 255, 0).astype(np.uint8)


def find_components(pixels, dev: float, size: int, threshold: int):
    binary_img = sobel_filter(pixels, dev, size, threshold)
    visited = np.zeros_like(binary_img, dtype=bool)
    binary_img = binary_img / 255
    components = []
    neighbors = [(-1, -1), (-1, 0), (-1, 1),( 0, -1),( 0, 1),( 1, -1), ( 1, 0), ( 1, 1)]
    rows, cols = binary_img.shape

    for i in range(rows):
        for j in range(cols):
            if binary_img[i, j] == 1 and not visited[i, j]:
                stack = [(i, j)]
                component = []

                while stack:
                    x, y = stack.pop()

                    if (0 <= x < rows) and (0 <= y < cols):
                        if not visited[x, y] and binary_img[x, y] == 1:
                            visited[x, y] = True
                            component.append((x, y))

                            for dx, dy in neighbors:
                                stack.append((x + dx, y + dy))

                if component:
                    components.append(component)

    if len(components) == 0:
        return []
    return components
# Finding the area in a closed loop
def area(pixels):
  area = 0
  for i in range(len(pixels)):
      x1, y1 = pixels[i]
      x2, y2 = pixels[(i + 1) % len(pixels)]
      area += (x1 * y2) - (x2 * y1)
  return abs(area) / 2
def EdgeDetection(pixels, deviation: float, size: int, threshold: int) -> list[int]:
    if pixels.dtype != 'float32':
        pixels = pixels.astype(np.float32) / 255
    components = sorted(find_components(pixels,deviation,size,threshold), key = len)
    filtered_comp = []
    for j in components:
      cond = False
      for i in j:
        if 0 in i or i[0] == len(pixels) or i[1] == len(pixels[0]):
          cond = True
      if not cond:
        filtered_comp.append(j)

    areas = []
    for i in filtered_comp:
      areas.append(area(i))
    biggest = areas.index(max(areas))
    return filtered_comp[biggest]
# Finding the corners in the detected edges
def corners(pixels):

  pixels = np.array(pixels)
  pixels = np.flip(pixels, axis=1)

  x22 = max(pixels, key = lambda x: x[0] + x[1])
  x11 = min(pixels, key = lambda x: x[0] + x[1])
  x12 = max(pixels, key = lambda x: x[0] - x[1])
  x21 = max(pixels, key = lambda x: -x[0] + x[1])
  return np.array([x11,x12,x21,x22])

# Finding the transformation matrix
def homography(source: np.ndarray, size):

  (x1,y1),(x2,y2),(x3,y3),(x4,y4) = source
  (u1,v1),(u2,v2),(u3,v3),(u4,v4) = [(0,0),(size,0),(0,size),(size,size)]
  A = np.array([[x1, y1, 1, 0, 0, 0, -u1*x1, -u1*y1],
                [0, 0, 0, x1, y1, 1, -v1*x1, -v1*y1],
                [x2, y2, 1, 0, 0, 0, -u2*x2, -u2*y2],
                [0, 0, 0, x2, y2, 1, -v2*x2, -v2*y2],
                [x3, y3, 1, 0, 0, 0, -u3*x3, -u3*y3],
                [0, 0, 0, x3, y3, 1, -v3*x3, -v3*y3],
                [x4, y4, 1, 0, 0, 0, -u4*x4, -u4*y4],
                [0, 0, 0, x4, y4, 1, -v4*x4, -v4*y4]])
  b = np.array([u1, v1, u2, v2, u3, v3, u4, v4])

  h = np.linalg.solve(A, b)

  h = np.append(h, 1)

  H = h.reshape((3, 3))

  return H
# Implementing the transformation matrix
def WarpedImage(img, dev, size_kernel, threshold, size = None):
    if img.dtype != 'float32':
        img = img.astype(np.float32) / 255

    if size is None:
        size = 600 if len(img) > 650 else min(len(img),len(img[0]))

    H = homography(corners(EdgeDetection(img, dev, size_kernel, threshold)), size)

    warped = np.zeros((size, size, 3))
    H_ = np.linalg.inv(H)

    for v in range(size):
        for u in range(size):
            original_pos = H_ @ np.array([u, v, 1])
            x = original_pos[0] / original_pos[2]
            y = original_pos[1] / original_pos[2]

            warped[v, u] = img[int(y), int(x)]
    return grayscale(warped)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
if __name__ == "__main__":
    image = plt.imread('Apple.jpg')/255
    componants = find_components(image,2,5,80)
    componants_image = np.zeros(image.shape[:2])

    for j in componants:
        for i in j:
            componants_image[i[0], i[1]] = 1


    sobel = blur(image, 2, 5)
    plt.subplot(3,1,1)
    plt.imshow(image)
    plt.title('Original')
    plt.subplot(3,1,2)
    plt.imshow(sobel, cmap = 'gray')
    plt.title('Grayscale + Blur')
    plt.subplot(3,2,3)
    plt.imshow(componants_image, cmap = 'gray')
    plt.title('Edge Detection')
    plt.show()