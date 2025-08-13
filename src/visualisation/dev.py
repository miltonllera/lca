import numpy as np
import jax
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation
from .utils import strip


def plot_state(img: np.ndarray | jax.Array):
    img = np.asarray(img).transpose(1, 2, 0).clip(0.0, 1.0)
    plt.imshow(img, vmin=0, vmax=1)
    # plt.gca().axis('off')
    strip(plt.gca())
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
    return plt.gcf()


def plot_orientation(cells: np.ndarray, use_rgb=False):
    _, H, W = cells.shape
    y, x = np.mgrid[-1:1:H*1j, -1:1:W*1j]  # between [-1;1] with H (W) points between them

    alive = cells[3] > 0.0
    rgb = cells[:3] * alive[None]

    theta = cells[-1] * alive
    u = np.cos(theta)
    v = np.sin(theta)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    if use_rgb:
        rgb = (theta[np.nonzero(alive)] % (2 * np.pi)) / (2 * np.pi)
        rgb = np.asarray([hsv_to_rgb(a, 1.0, 1.0) for a in rgb])
    else:
        rgb = 'k'

    ax.quiver(
        x[alive],
        y[alive],
        u[alive],
        v[alive],
        color=rgb,
        angles='xy',
        pivot='mid',
        scale_units='xy',
        scale=W / 1.5
    )
    # plt.gca().invert_yaxis()  # optional: to match image coordinates
    strip(ax)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    # ax.axis('auto')
    # ax.set_title("Cell orientation")
    return fig


#--------------------------------------------- Animations ----------------------------------------

def plot_dev_path(dev_path: np.ndarray | jax.Array):
    fig = plt.figure()
    ax = plt.gca()
    strip(ax)

    dev_path = dev_path.transpose(0, 2, 3, 1).clip(0.0, 1.0)

    im = plt.imshow(dev_path[0], vmin=0, vmax=1)
    def animate(i):
        ax.set_title(f"Growth step: {i}")
        im.set_array(dev_path[i])
        return im,

    ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(dev_path))
    plt.close(fig)
    return ani


def plot_hidden_state(dev_path):
    S, C, H, W = dev_path.shape

    # extract alive states
    all_cell_states = dev_path.transpose(0, 2, 3, 1).reshape(S * H * W, C)
    alive_idx = np.nonzero(all_cell_states[:, 3:4])
    alive = all_cell_states[alive_idx[0]]

    # pca transform
    pca = PCA(n_components=3, whiten=True, power_iteration_normalizer='auto')
    alive_pca = pca.fit_transform(alive[..., 4:])

    # normalise to color space
    alive_pca = (alive_pca - alive_pca.min()) / (alive_pca.max() - alive_pca.min())

    path_pca = np.zeros((len(all_cell_states), 3))
    path_pca[alive_idx[0]] = alive_pca

    path_pca_image = path_pca.reshape(S, H, W, 3).transpose(0, 3, 1, 2)
    ani = plot_dev_path(path_pca_image);
    return ani

