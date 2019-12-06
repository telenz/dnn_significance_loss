#import section
from matplotlib import pylab
import pylab as plt
import numpy as np

#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(alpha, x):
    return (1 / (1 + np.exp(-alpha*x)))

mySamples = []
mySigmoid = []

# generate an Array with value ???
# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
# 
x = plt.linspace(-0.5,0.5,1000)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, sigmoid(10, x), 'g', label='alpha = 10')
plt.plot(x, sigmoid(20, x), 'b', label='alpha = 20')
plt.plot(x, sigmoid(50, x), 'r', label='alpha = 50')
plt.plot(x, sigmoid(100, x), 'c', label='alpha = 100')
plt.plot(x, sigmoid(200, x), 'm', label='alpha = 200')

# Draw the grid line in background.
plt.grid()

# Title & Subtitle
plt.title('Sigmoid Function')

# place the legen boc in bottom right of the graph
plt.legend(loc='lower right')

# write the Sigmoid formula
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)

# plt.plot(x)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# create the graph
plt.show()
