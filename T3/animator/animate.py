import numpy as np
from mpl_backend_workaround import MPL_BACKEND_USED
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, FuncAnimation
import numpy.random
import sys


def setup_axes(points, title):
    fig, ax = plt.subplots()

    fig.canvas.set_window_title(title)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+100+100")

    ax.set_aspect('equal')

    x_min = np.min(points[:, :, 0])
    x_max = np.max(points[:, :, 0])
    y_min = np.min(points[:, :, 1])
    y_max = np.max(points[:, :, 1])

    percent_left = 0.1
    x_inc = (x_max - x_min) * percent_left
    y_inc = (y_max - y_min) * percent_left

    ax.set_xlim(x_min - x_inc, x_max + x_inc)
    ax.set_ylim(y_min - y_inc, y_max + y_inc)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    ax.grid(True, which='both')

    return fig, ax


class ParticlesReader:
    def __init__(self, path):
        self.path = path
        self.n = None
        self.n_iter = None
        self.h = None
        self.w = None
        self.p = None
        self.tm = None

    def read(self):
        with open(self.path, 'r') as f:
            tm = float(next(f))

            s = next(f).split()
            n, n_iter, h = int(s[0]), int(s[1]), float(s[2])

            s = next(f).split()
            w = np.array([float(w_i) for w_i in s])

            p = np.ndarray((n_iter, n, 2))

            for it in range(n_iter):
                for k in range(2):
                    s = next(f).split()
                    p[it, :, k] = [float(p_i) for p_i in s]

            self.h = h
            self.n = n
            self.n_iter = n_iter
            self.p = p
            self.w = w
            self.tm = tm


def main():
    print('Using MPL backend: {}'.format(MPL_BACKEND_USED))

    pt_r = 0.05
    animation_delay = 300
    path = sys.argv[1]

    r = ParticlesReader(path)
    r.read()
    col = [tuple(numpy.random.uniform(0, 1, 3)) for _ in range(r.n)]

    # Create artists now
    fig, ax = setup_axes(r.p, "Particles")

    artists = [[plt.Circle(r.p[it, i, :], pt_r, color=col[i])
                for i in range(r.n)]
               for it in range(r.n_iter)]

    for l in artists:
        for a in l:
            ax.add_artist(a)

    _ = ArtistAnimation(fig, artists,
                        interval=animation_delay,
                        repeat_delay=None,
                        repeat=True)

    plt.show()


if __name__ == "__main__":
    main()
