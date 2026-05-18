# MLS C1 W1 工具文件使用说明

这个压缩包里有 3 个文件：

- `lab_utils_uni.py`：课程 Cost Function 可视化工具的本地兼容版
- `lab_utils_common.py`：公共函数和配色，包含 `compute_cost`
- `deeplearning.mplstyle`：Matplotlib 绘图样式

## 放在哪里？

把这 3 个文件复制到你的 notebook 同一个文件夹，例如：

```text
mechine/week1/
├── 01_linear_regression.ipynb
├── lab_utils_uni.py
├── lab_utils_common.py
└── deeplearning.mplstyle
```

## 在 notebook 中怎么导入？

在成本函数这一节需要用课程工具时，再新建一个代码单元格：

```python
%matplotlib widget
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')
```

如果 `%matplotlib widget` 报错，先在终端安装：

```bash
pip install ipympl ipywidgets
```

或者在 notebook 中运行：

```python
%pip install ipympl ipywidgets
```

## 课程函数怎么用？

```python
plt_intuition(x_train, y_train)
```

```python
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
```

```python
soup_bowl()
```
