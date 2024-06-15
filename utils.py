import random
import logging
import numpy as np
import torch
import pandas as pd
import os
import re


def set_seed(seed):  # set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]


def load_imdb(path):
    df = pd.read_csv(path, sep=',')
    sents = []
    for i in range(len(df)):
        text = df.iloc[i][0]
        text = text.replace('.', ' .')
        text = text.replace(',', ' ,')
        text = re.sub('<br />', ' ', str(text))
        text = text.lower()
        text = re.sub(r'\s+', ' ', text, flags=re.I)  # substituting multiple spaces with single space
        sents.append(text.split())
    return sents


def load_sent(path):
    sents = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            sents.append(line.split())
    return sents


def write_sent(sents, path):
    with open(path, 'w', encoding="utf-8") as f:
        for s in sents:
            f.write(' '.join(s) + '\n')


def write_doc(docs, path):
    with open(path, 'w', encoding="utf-8") as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')


def write_z(z, path):
    with open(path, 'w', encoding="utf-8") as f:
        for zi in z:
            for zij in zi:
                f.write('%f ' % zij)
            f.write('\n')


def logging_config(save_dir):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'running.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True


def lerp(t, p, q):
    return (1 - t) * p + t * q


def slerp(t, p, q):
    o = np.arccos(np.dot(p / np.linalg.norm(p), q / np.linalg.norm(q)))
    so = np.sin(o)
    return np.sin((1 - t) * o) / so * p + np.sin(t * o) / so * q


def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0 * i / (n - 1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)
