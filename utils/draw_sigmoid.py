#import section
from matplotlib import pylab
import pylab as plt
import numpy as np

#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

mySamples = []
mySigmoid = []

# generate an Array with value ???
# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
# 
x = plt.linspace(-0.5,0.5,1000)
y = plt.linspace(-0.5,0.5,100000)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, sigmoid(x), 'r', label='10 bins')
plt.plot(y, sigmoid(y), 'b', label='100 bins')

# Draw the grid line in background.
plt.grid()

# Title & Subtitle
plt.title('Sigmoid Function')
plt.suptitle('Sigmoid')

# place the legen boc in bottom right of the graph
plt.legend(loc='lower right')

# write the Sigmoid formula
plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)

# #resize the X and Y axes
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
 

# plt.plot(x)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# create the graph
plt.show()
