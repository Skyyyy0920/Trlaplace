import matplotlib.pyplot as plt

data = 'sst2'
vector = 'fastText'
# 数据准备
epsilons = [0.01, 0.1, 1, 10]
Gaussian = [0.8868, 0.8892, 0.8917, 0.8983]
Laplace = [0.8918, 0.8953, 0.8958, 0.8962]
CMP = [0.8769, 0.8923, 0.891, 0.8915]
Mahalanobis = [0.8967, 0.8954, 0.8877, 0.8929]
PE = [0.8838, 0.8914, 0.8962, 0.8964]
TrLaplace = [0.892, 0.8961, 0.8983, 0.8994]
baseline_accuracy = 0.9103

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()

data = 'sst2'
vector = 'GloVe'
# 数据准备
Gaussian = [0.8868, 0.8892, 0.8917, 0.8983]
Laplace = [0.8910, 0.8959, 0.8988, 0.8992]
CMP = [0.8789, 0.8911, 0.8951, 0.8935]
Mahalanobis = [0.8912, 0.8954, 0.8927, 0.8949]
PE = [0.8858, 0.8924, 0.8962, 0.8994]
TrLaplace = [0.8952, 0.8961, 0.8973, 0.8964]
baseline_accuracy = 0.8997

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()

data = 'sst2'
vector = 'random'
# 数据准备
epsilons = [0.01, 0.1, 1, 10]
Gaussian = [0.8649, 0.8527, 0.8651, 0.8643]
Laplace = [0.8679, 0.8626, 0.8703, 0.8662]
CMP = [0.8617, 0.8558, 0.8646, 0.8631]
Mahalanobis = [0.8698, 0.8676, 0.8603, 0.8681]
PE = [0.8584, 0.8698, 0.8594, 0.8581]
TrLaplace = [0.8663, 0.8655, 0.8683, 0.8680]
baseline_accuracy = 0.5607

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()

import matplotlib.pyplot as plt

data = 'ag_news'
vector = 'fastText'
# 数据准备
epsilons = [0.01, 0.1, 1, 10]
Gaussian = [0.8915, 0.8936, 0.8929, 0.8907]
Laplace = [0.8903, 0.8936, 0.8954, 0.8963]
CMP = [0.8899, 0.894, 0.8918, 0.8927]
Mahalanobis = [0.8901, 0.892, 0.896, 0.8947]
PE = [0.8912, 0.8953, 0.8932, 0.8896]
TrLaplace = [0.8959, 0.8959, 0.9008, 0.8975]
baseline_accuracy = 0.9138

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()

data = 'ag_news'
vector = 'GloVe'
# 数据准备
epsilons = [0.01, 0.1, 1, 10]
Gaussian = [0.8866, 0.8882, 0.8889, 0.8932]
Laplace = [0.8874, 0.886, 0.8891, 0.889]
CMP = [0.8863, 0.89, 0.8808, 0.8877]
Mahalanobis = [0.8931, 0.8918, 0.8878, 0.8878]
PE = [0.8866, 0.8863, 0.8871, 0.893]
TrLaplace = [0.8887, 0.8896, 0.8918, 0.8897]
baseline_accuracy = 0.9161

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()

data = 'ag_news'
vector = 'random'
# 数据准备
epsilons = [0.01, 0.1, 1, 10]
Gaussian = [0.8591, 0.8527, 0.8651, 0.8643]
Laplace = [0.8656, 0.8626, 0.8703, 0.8662]
CMP = [0.8617, 0.8558, 0.8646, 0.8631]
Mahalanobis = [0.8698, 0.8676, 0.8603, 0.8681]
PE = [0.8584, 0.8698, 0.8594, 0.8581]
TrLaplace = [0.8653, 0.8655, 0.8627, 0.8644]
baseline_accuracy = 0.2462

Gaussian = sorted(Gaussian)
Laplace = sorted(Laplace)
CMP = sorted(CMP)
Mahalanobis = sorted(Mahalanobis)
PE = sorted(PE)

# 设置字体
plt.rc('font', family='Times New Roman')

# 创建图形对象和轴对象
fig, ax = plt.subplots(figsize=(5, 5))  # 设置宽度为6英寸，高度为8英寸
# fig, ax = plt.subplots()

# 绘制折线图
ax.plot([0, 1, 2, 3], Gaussian, marker='o', label='Gaussian')
ax.plot([0, 1, 2, 3], Laplace, marker='s', label='Laplacian')
ax.plot([0, 1, 2, 3], CMP, marker='^', label='CMP')
ax.plot([0, 1, 2, 3], Mahalanobis, marker='d', label='Mahalanobis')
ax.plot([0, 1, 2, 3], PE, marker='*', label='PTE')
ax.plot([0, 1, 2, 3], TrLaplace, marker='p', label='TrLaplacian')

# 绘制baseline
# ax.axhline(y=baseline_accuracy, color='r', linestyle='--', label='Baseline')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(epsilons)

# 设置轴标签和标题
ax.set_xlabel('Epsilon', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
# ax.set_title('Accuracy vs Epsilon for Different Methods', fontsize=14)

# ax.legend()
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(f'{data}-{vector}.pdf', format='pdf')
# plt.show()
