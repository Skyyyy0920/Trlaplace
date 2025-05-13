# Trlaplace
This is the code implementation for our paper [*Private Language Models via Truncated Laplacian Mechanism (EMNLP 2024 oral)*](https://aclanthology.org/2024.emnlp-main.231/).

## Cite our work
```
@inproceedings{huang-etal-2024-private,
    title = "Private Language Models via Truncated Laplacian Mechanism",
    author = "Huang, Tianhao  and
      Yang, Tao  and
      Habernal, Ivan  and
      Hu, Lijie  and
      Wang, Di",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.231/",
    doi = "10.18653/v1/2024.emnlp-main.231",
    pages = "3980--3993",
    abstract = "Recently it has been shown that deep learning models for NLP tasks are prone to attacks that can even reconstruct the verbatim training texts. To prevent privacy leakage, researchers have investigated word-level perturbations, relying on the formal guarantees of differential privacy (DP) in the embedding space. However, many existing approaches either achieve unsatisfactory performance in the high privacy regime when using the Laplacian or Gaussian mechanism, or resort to weaker relaxations of DP that are inferior to the canonical DP in terms of privacy strength. This raises the question of whether a new method for private word embedding can be designed to overcome these limitations. In this paper, we propose a novel private embedding method called the high dimensional truncated Laplacian mechanism. Specifically, we introduce a non-trivial extension of the truncated Laplacian mechanism, which was previously only investigated in one-dimensional space cases. Theoretically, we show that our method has a lower variance compared to the previous private word embedding methods. To further validate its effectiveness, we conduct comprehensive experiments on private embedding and downstream tasks using three datasets. Remarkably, even in the high privacy regime, our approach only incurs a slight decrease in utility compared to the non-private scenario."
}
```
