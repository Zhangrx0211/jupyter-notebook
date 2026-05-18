"""
lab_utils_uni.py
本地兼容版：复现 Andrew Ng Machine Learning Specialization C1 W1 Cost Function 可视化工具。
包含当前课程笔记会用到的：
    plt_intuition
    plt_stationary
    plt_update_onclick
    soup_bowl
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

try:
    from ipywidgets import interact
except Exception:
    interact = None

from lab_utils_common import compute_cost
from lab_utils_common import dlblue, dlorange, dldarkred, dlmagenta, dlpurple, dlcolors

try:
    plt.style.use('./deeplearning.mplstyle')
except OSError:
    pass

n_bin = 5
dlcm = LinearSegmentedColormap.from_list('dl_map', dlcolors, N=n_bin)


def plt_house_x(X, y, f_wb=None, ax=None):
    """画房价训练数据；如果提供 f_wb，也画预测线。"""
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(X, y, marker='x', c='r', label='Actual Value')
    ax.set_title('Housing Prices')
    ax.set_ylabel('Price (in 1000s of dollars)')
    ax.set_xlabel('Size (1000 sqft)')
    if f_wb is not None:
        ax.plot(X, f_wb, c=dlblue, label='Our Prediction')
    ax.legend()
    return ax


def mk_cost_lines(x, y, w, b, ax):
    """在模型预测线和真实点之间画出误差线。"""
    cstr = 'cost = (1/m)*('
    ctot = 0
    label = 'cost for point'
    addedbreak = False
    for p in zip(x, y):
        f_wb_p = w * p[0] + b
        c_p = ((f_wb_p - p[1]) ** 2) / 2
        ax.vlines(p[0], p[1], f_wb_p, lw=3, color=dlpurple, ls='dotted', label=label)
        label = ''
        cxy = [p[0], p[1] + (f_wb_p - p[1]) / 2]
        ax.annotate(f'{c_p:0.0f}', xy=cxy, xycoords='data', color=dlpurple,
                    xytext=(5, 0), textcoords='offset points')
        cstr += f'{c_p:0.0f} +'
        if len(cstr) > 38 and not addedbreak:
            cstr += '\n'
            addedbreak = True
        ctot += c_p
    ctot = ctot / len(x)
    cstr = cstr[:-1] + f') = {ctot:0.0f}'
    ax.text(0.15, 0.02, cstr, transform=ax.transAxes, color=dlpurple)


def _plt_intuition_once(x_train, y_train, w=150):
    tmp_b = 100
    w_range = np.array([0, 400])
    w_array = np.arange(*w_range, 5)
    cost = np.zeros_like(w_array, dtype=float)
    for i, tmp_w in enumerate(w_array):
        cost[i] = compute_cost(x_train, y_train, tmp_w, tmp_b)

    f_wb = np.dot(x_train, w) + tmp_b
    cur_cost = compute_cost(x_train, y_train, w, tmp_b)

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
    try:
        fig.canvas.toolbar_position = 'bottom'
    except Exception:
        pass
    mk_cost_lines(x_train, y_train, w, tmp_b, ax[0])
    plt_house_x(x_train, y_train, f_wb=f_wb, ax=ax[0])
    ax[1].plot(w_array, cost)
    ax[1].scatter(w, cur_cost, s=100, color=dldarkred, zorder=10, label=f'cost at w={w}')
    ax[1].hlines(cur_cost, ax[1].get_xlim()[0], w, lw=4, color=dlpurple, ls='dotted')
    ax[1].vlines(w, ax[1].get_ylim()[0], cur_cost, lw=4, color=dlpurple, ls='dotted')
    ax[1].set_title('Cost vs. w, (b fixed at 100)')
    ax[1].set_ylabel('Cost')
    ax[1].set_xlabel('w')
    ax[1].legend(loc='upper center')
    fig.suptitle(f'Minimize Cost: Current Cost = {cur_cost:0.0f}', fontsize=12)
    plt.show()


def plt_intuition(x_train, y_train):
    """课程中的成本函数直觉图：固定 b=100，拖动/设置 w 观察成本变化。"""
    if interact is not None:
        @interact(w=(0, 400, 10), continuous_update=False)
        def func(w=150):
            _plt_intuition_once(x_train, y_train, w)
        return func
    return _plt_intuition_once(x_train, y_train, 150)


def plt_stationary(x_train, y_train):
    """画模型、等高线和 3D 曲面，用来点击观察不同 w,b 的 cost。"""
    fig = plt.figure(figsize=(9, 8))
    fig.set_facecolor('#ffffff')
    try:
        fig.canvas.toolbar_position = 'top'
    except Exception:
        pass

    gs = GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :], projection='3d')
    ax = np.array([ax0, ax1, ax2], dtype=object)

    w_range = np.array([200 - 300., 200 + 300.])
    b_range = np.array([50 - 300., 50 + 300.])
    b_space = np.linspace(*b_range, 100)
    w_space = np.linspace(*w_range, 100)

    tmp_b, tmp_w = np.meshgrid(b_space, w_space)
    z = np.zeros_like(tmp_b)
    for i in range(tmp_w.shape[0]):
        for j in range(tmp_w.shape[1]):
            z[i, j] = compute_cost(x_train, y_train, tmp_w[i, j], tmp_b[i, j])
            if z[i, j] == 0:
                z[i, j] = 1e-6

    w0 = 200
    b0 = -100
    f_wb = np.dot(x_train, w0) + b0
    mk_cost_lines(x_train, y_train, w0, b0, ax[0])
    plt_house_x(x_train, y_train, f_wb=f_wb, ax=ax[0])

    ax[1].contour(tmp_w, tmp_b, np.log(z), levels=12, linewidths=2,
                  alpha=0.7, colors=dlcolors)
    ax[1].set_title('Cost(w,b)')
    ax[1].set_xlabel('w', fontsize=10)
    ax[1].set_ylabel('b', fontsize=10)
    ax[1].set_xlim(w_range)
    ax[1].set_ylim(b_range)
    cscat = ax[1].scatter(w0, b0, s=100, color=dlblue, zorder=10,
                          label='cost with \ncurrent w,b')
    chline = ax[1].hlines(b0, ax[1].get_xlim()[0], w0, lw=4, color=dlpurple, ls='dotted')
    cvline = ax[1].vlines(w0, ax[1].get_ylim()[0], b0, lw=4, color=dlpurple, ls='dotted')
    ax[1].text(0.5, 0.95, 'Click to choose w,b', bbox=dict(facecolor='white', ec='black'),
               fontsize=10, transform=ax[1].transAxes, va='center', ha='center')

    ax[2].plot_surface(tmp_w, tmp_b, z, cmap=dlcm, alpha=0.3, antialiased=True)
    ax[2].plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
    ax[2].set_xlabel('$w$')
    ax[2].set_ylabel('$b$')
    ax[2].zaxis.set_rotate_label(False)
    ax[2].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[2].set_zlabel('J(w, b)\n\n', rotation=90)
    ax[2].set_title('Cost(w,b) \n [You can rotate this figure]', size=12)
    ax[2].view_init(30, -120)
    return fig, ax, [cscat, chline, cvline]


class plt_update_onclick:
    """点击等高线图，更新左侧预测线和 3D 图中的点。"""
    def __init__(self, fig, ax, x_train, y_train, dyn_items):
        self.fig = fig
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train
        self.dyn_items = dyn_items
        self.cid = fig.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if event.inaxes == self.ax[1]:
            ws = event.xdata
            bs = event.ydata
            cst = compute_cost(self.x_train, self.y_train, ws, bs)

            self.ax[0].clear()
            f_wb = np.dot(self.x_train, ws) + bs
            mk_cost_lines(self.x_train, self.y_train, ws, bs, self.ax[0])
            plt_house_x(self.x_train, self.y_train, f_wb=f_wb, ax=self.ax[0])

            for artist in self.dyn_items:
                try:
                    artist.remove()
                except Exception:
                    pass
            a = self.ax[1].scatter(ws, bs, s=100, color=dlblue, zorder=10,
                                   label='cost with \ncurrent w,b')
            b = self.ax[1].hlines(bs, self.ax[1].get_xlim()[0], ws, lw=4, color=dlpurple, ls='dotted')
            c = self.ax[1].vlines(ws, self.ax[1].get_ylim()[0], bs, lw=4, color=dlpurple, ls='dotted')
            d = self.ax[1].annotate(f'Cost: {cst:.0f}', xy=(ws, bs), xytext=(4, 4),
                                    textcoords='offset points', bbox=dict(facecolor='white'), size=10)
            e = self.ax[2].scatter3D(ws, bs, cst, marker='X', s=100)
            self.dyn_items = [a, b, c, d, e]
            self.fig.canvas.draw()


def soup_bowl():
    """画一个标准碗形 3D 成本函数示意图。"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)
    z = np.zeros((len(w), len(b)))
    for j, x in enumerate(w):
        for i, y in enumerate(b):
            z[i, j] = x ** 2 + y ** 2

    W, B = np.meshgrid(w, b)
    ax.plot_surface(W, B, z, cmap='Spectral_r', alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel('$w$')
    ax.set_ylabel('$b$')
    ax.set_zlabel('$J(w,b)$', rotation=90)
    ax.set_title('$J(w,b)$\n [You can rotate this figure]', size=15)
    plt.show()
