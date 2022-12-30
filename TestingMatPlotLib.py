import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

data = [0,1,2,3,4]

plot = plt.subplot()
plot.bar(np.arange(len(data)), data, 1, label="Testing If Right", color="blue")
plt.savefig("testingPlot.png")

image = cv.imread("testingPlot.png")
cv.imshow("Plot Result", image)
cv.waitKey()