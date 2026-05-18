"""
lab_utils_common.py
本地兼容版：用于 Andrew Ng Machine Learning Specialization C1 W1/W2 optional labs 的基础工具。
重点支持当前 Cost Function 笔记中用到的 compute_cost 和配色变量。
"""

import numpy as np
import matplotlib.pyplot as plt

# DeepLearning.AI 课程常用配色
_dlstyle = './deeplearning.mplstyle'
try:
    plt.style.use(_dlstyle)
except OSError:
    pass

dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]
dlc = dict(dlblue=dlblue, dlorange=dlorange, dldarkred=dldarkred,
           dlmagenta=dlmagenta, dlpurple=dlpurple)


def compute_cost(x, y, w, b):
    """
    计算线性回归成本函数。

    兼容两种情况：
    1. 一元线性回归：x 是一维数组，w 是标量
    2. 多元线性回归：x 是二维矩阵，w 是向量

    J(w,b) = 1/(2m) * sum((f_wb - y)^2)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = x.shape[0]
    cost_sum = 0.0

    for i in range(m):
        if x.ndim == 1:
            f_wb_i = w * x[i] + b
        else:
            f_wb_i = np.dot(x[i], w) + b
        cost_sum += (f_wb_i - y[i]) ** 2

    return cost_sum / (2 * m)


def compute_gradient(x, y, w, b):
    """计算线性回归梯度；兼容一元和多元输入。"""
    x = np.asarray(x)
    y = np.asarray(y)
    m = x.shape[0]

    if x.ndim == 1:
        dj_dw = 0.0
        dj_db = 0.0
        for i in range(m):
            err = (w * x[i] + b) - y[i]
            dj_dw += err * x[i]
            dj_db += err
        return dj_db / m, dj_dw / m

    n = x.shape[1]
    dj_dw = np.zeros((n,))
    dj_db = 0.0
    for i in range(m):
        err = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * x[i, j]
        dj_db += err
    return dj_db / m, dj_dw / m
