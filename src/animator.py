import matplotlib.pyplot as plt

try:
    from IPython.display import display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes]

        self.config_axes = lambda: self._set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

        self.display_handle = None
        self.config_axes()
        if HAS_IPYTHON:
            self.display_handle = display(self.fig, display_id=True)

    def _set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.grid(True)

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        
        if self.X is None:
            self.X = [[] for _ in range(n)]
            self.Y = [[] for _ in range(n)]
        
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        self.axes[0].cla()
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)
        self.config_axes()

        if self.display_handle is not None:
            self.display_handle.update(self.fig)
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show(self):
        plt.show(block=False)

    def close(self):
        plt.close(self.fig)