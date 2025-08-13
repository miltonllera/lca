import numpy as np
import matplotlib.pyplot as plt


def strip(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def tile2d(a, w=None, format='NCWH'):
    a = np.asarray(a)

    if format == 'NCHW':
        a = a.transpose(0, 2, 3, 1)

    if w is None:
        w = int(np.ceil(np.sqrt(a.shape[0])))

    th, tw = a.shape[1:3]
    pad = (w - a.shape[0]) % w
    a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
    h = a.shape[0] // w
    a = a.reshape([h, w]+list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))

    if format == 'NCHW':
        a = a.transpose(2, 0, 1)

    return a


#----------------------------------------------- Plots -------------------------------------------

def plot_examples(a, w=None, format='NCHW', ax=None):
    if format=='NCHW':
        a = a.transpose(0, 2, 3, 1)
    fig, ax = plt.subplots()
    a = tile2d(a, w, format='NHWC')
    ax.imshow(a)
    strip(ax)
    return fig

