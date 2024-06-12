import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from time import time
from sklearn.datasets import load_digits

digits = load_digits(n_class=6)

# 1.读取GloVe向量数据库，取6种类。共400,000个token，每个token是50维。
glove_txt = "./.vector_cache/glove.300d.yelp.finetune.txt"
with open(glove_txt) as f:
    lines = f.readlines()
    lines = lines[15:65]
    X = []
    y = []
    for line in lines:
        now = line.split(' ')
        y.append(now[0])
        X.append(now[1:])
    X = np.array(X)
    y = np.array(y)

n_samples, n_features = X.shape


# 2.编写绘画函数，对输入的数据X进行画图。
def plot_embedding(X, title, ax):
    X = MinMaxScaler().fit_transform(X)

    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        ax.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            # color=plt.cm.Dark2(y[i]),
            fontdict={"weight": "bold", "size": 9},
        )
        '''
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        ax.add_artist(imagebox)
        '''
    ax.set_title(title)
    ax.axis("off")


# 3.选择要用那种方式对原始数据编码(Embedding),这里选择TSNE。
#  n_components = 2表示输出为2维，learning_rate默认是200.0,
embeddings = {
    "t-SNE embeedding": TSNE(
        n_components=4, init='pca', learning_rate=200.0, random_state=0
    ),
}


# 4.根据字典里（这里只有TSNE）的编码方式，生成压缩后的编码矩阵
# 即把每个样本生成了2维的表示。维度由原来的50位变成了2位。
# Input: (n_sample, n_dimension)
# Output: (n_sample, 2)

projections, timing = {}, {}
for name, transformer in embeddings.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()
    print(data.shape, type(data.shape))
    projections[name] = transformer.fit_transform(data, y)
    timing[name] = time() - start_time

# 1.读取GloVe向量数据库，取6种类。共400,000个token，每个token是50维。
glove_txt_1 = "./.vector_cache/glove.6B.300d.txt"
with open(glove_txt_1,encoding="utf-8") as f_1:
    lines = f_1.readlines()
    lines = lines[15:65]
    X = []
    y = []
    for line in lines:
        now = line.split(' ')
        y.append(now[0])
        X.append(now[1:])
    X = np.array(X)
    y = np.array(y)

n_samples_1, n_features_1 = X.shape


# 2.编写绘画函数，对输入的数据X进行画图。
def plot_embedding_1(X, title, ax):
    X = MinMaxScaler().fit_transform(X)

    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        ax.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            color='red',
            fontdict={"weight": "bold", "size": 9},
        )
        '''
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        ax.add_artist(imagebox)
        '''
    ax.set_title(title)
    ax.axis("off")


# 3.选择要用那种方式对原始数据编码(Embedding),这里选择TSNE。
#  n_components = 2表示输出为2维，learning_rate默认是200.0,
embeddings_1 = {
    "t-SNE embeedding": TSNE(
        n_components=4, init='pca', learning_rate=200.0, random_state=0
    ),
}


# 4.根据字典里（这里只有TSNE）的编码方式，生成压缩后的编码矩阵
# 即把每个样本生成了2维的表示。维度由原来的50位变成了2位。
# Input: (n_sample, n_dimension)
# Output: (n_sample, 2)

projections_1, timing_1 = {}, {}
for name, transformer in embeddings_1.items():
    if name.startswith("Linear Discriminant Analysis"):
        data = X.copy()
        data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
    else:
        data = X

    print(f"Computing {name}...")
    start_time = time()
    print(data.shape, type(data.shape))
    projections_1[name] = transformer.fit_transform(data, y)
    timing_1[name] = time() - start_time

# 6.把编码矩阵输出到二维图像中来。
fig, ax = plt.subplots()
title = f"{name} (time {timing[name]:.3f}s)"
plot_embedding(projections[name], title, ax)
title_1 = f"{name} (time {timing[name]:.3f}s)"
plot_embedding_1(projections_1[name], title_1, ax)
plt.show()
