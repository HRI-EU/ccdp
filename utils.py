import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class my_ploty():
    def __init__(self,):
        pass

    def plot_gradient(self,ax, x,y, cmap = "coolwarm"):
        # Define colors - here we create a gradient from blue to red
        colors = np.linspace(0, 1, len(x))  # Values from 0 to 1 for gradient

        # Create segments of the line
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a LineCollection with a colormap for the gradient effect
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(colors)

        ax.add_collection(lc)
        return ax, lc
