# Example of using the cudaNormalize function to preprocess an
# input image for use in SSD-like object detectors
import cv2
import jetson.utils
import numpy as np
import time
array = np.array([[0, 2, 3], [4, 5, 6], [100, 200, 255]], np.float32)
print(array, "\n")
start = time.time()
arraycpu = (2.0 / 255.0) * array - 1.0
end = time.time()
print("CPU time", end - start)

start = time.time()
imgInput = jetson.utils.cudaFromNumpy(array)
imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format=imgInput.format)
jetson.utils.cudaNormalize(imgInput, (0.,255.), imgOutput, (-1.,1.))
end = time.time()
print("GPU time", end - start)
arraygpu = jetson.utils.cudaToNumpy(imgOutput)
print("GPU Array", arraygpu)
print("CPU Array", arraycpu)
