import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class colored_line:
    def __init__(self, x, y, c, **kwargs):
        """Small wrapper for a LineCollection with easier to use set_data functions
               x[N]                   xdata
               y[N]                   ydata
               c[N], c[N,3], c[N,4]   color data
               **kwargs               kwargs valid for matplotlib.LineCollection
        """
        x = np.asarray(x)
        y = np.asarray(y)
        c = np.asarray(c)

        xy = np.vstack([x,y]).T
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        self.collection = LineCollection(segments, **kwargs)
        self.collection.set_edgecolor(c)


    def set_data(self, x, y, c=None):
        """Set new x and y data, optionally new color data
               x[N]                   xdata
               y[N]                   ydata
               c[N], c[N,3], c[N,4]   color data (default: current colors)
        """

        xy = np.vstack([x,y]).T
        xy = xy.reshape(-1, 1, 2)
        segments = np.hstack([xy[:-1], xy[1:]])

        self.collection.set_segments(segments)
        if c is not None:
            self.collection.set_edgecolor(c)
        
    def set_xdata(self, x):
        """Set new x (must be same size as current y data)
               x[N]                   xdata
        """
        segments = np.array(self.collection.get_segments())
        segments[:,0,0] = x[:-1]
        segments[:,1,0] = x[1:]
        self.collection.set_segments(segments)

    def set_ydata(self, y):
        """Set new y (must be same size as current x data)
               y[N]                   ydata
        """
        segments = np.array(self.collection.get_segments())
        segments[:,0,1] = y[:-1]
        segments[:,1,1] = y[1:]
        self.collection.set_segments(segments)

    def set_colors(self, c):
        """Set new color data
               c[N], c[N,3], c[N,4]   color data
        """
        self.collection.set_edgecolor(c)

    def set_animated(self, animated):
        self.collection.set_animated(animated)

    def get_zorder(self):
        return self.collection.get_zorder()

def colored_plot(x, y, c, ax=None, **kwargs):
    """Like matplotlib plot, but use a list of colors c for variable color. Return a colored_line
            x[N]                   xdata
            y[N]                   ydata
            c[N], c[N,3], c[N,4]   color data
            ax                     axis (default: current axis)
            **kwargs               kwargs valid for matplotlib.LineCollection
    """
    if ax is None:
        ax = plt.gca()
    line = colored_line(x,y,c, **kwargs)

    ax.add_collection(line.collection)
    line.axes = line.collection.axes
    line.figure = line.collection.figure
    line.draw = line.collection.draw

    return line

