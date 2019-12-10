#import section
from matplotlib import pylab
import pylab as plt
import numpy as np

#sigmoid = lambda x: 1 / (1 + np.exp(-x))
def f_three_range(s_1, s_2, s_3, x_1, x_2, y_pred):

    # Calculate scaling factor to bring output between 0 and 1
    a = s_1*x_1 + s_2*(x_2-x_1) + s_3*(1-x_2)
    sf = 1./a

    f_1 = np.maximum( s_1*y_pred , 0 ) - np.maximum( s_1*(y_pred-x_1) , 0)
    f_2 = np.maximum( s_2*(y_pred-x_1), 0) - np.maximum( s_2*(y_pred-x_2), 0)
    f_3 = np.maximum( s_3*(y_pred-x_2), 0) - np.maximum( s_3*(y_pred-1), 0)

    return sf*(f_1 + f_2 + f_3)

# generate an Array with value ???
# linespace generate an array from start and stop value
# with requested number of elements. Example 10 elements or 100 elements.
# 
x = np.linspace(-0.1,1.1,1000)

# prepare the plot, associate the color r(ed) or b(lue) and the label 
plt.plot(x, f_three_range(0.1, 1.0, 0.1, 0.4, 0.6, x), 'g', label='alpha = 10')

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

#plt.yscale('log')

# create the graph
plt.show()
