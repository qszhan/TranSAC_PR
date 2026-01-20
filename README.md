# TranSAC_PR

This repo provides the implementation of "TranSAC: An Unsupervised Transferability Metric Based on Task Speciality and Domain Commonality" published in Pattern Recognition.


## Metrics

$$
\text{TranSAC} = H(T) - \frac{1}{K} H(\hat{Z} \mid T)
$$

- **T**: target representations (e.g., embeddings from a pretrained model)
- **p(Ẑ | T)**: soft predictions over K source classes


## Very Simple Usage in your project to evaluate transferability
```python
import numpy as np
from transac import compute_transac

# T: (n, d) embeddings from your model
# tar_prob: (n, K) softmax probabilities from the source classifier head
score = compute_transac(tar_prob, T, eps=10.0, order=5)
print("TranSAC:", score)
```

## Notes 
 

* **eps**: choose a relatively large value (paper commonly uses `eps=10`) to help Taylor-series convergence.

* **order**: small orders like `3` or `5` usually work well; higher orders cost more.

* Complexity is driven mainly by the Taylor expansion over an `n x n` Gram matrix (roughly cubic in `n`).



## Pipeline of transferability estimation 
### Requirements

* Install `PyTorch==1.11.0` and `torchvision==0.12.0` with `CUDA==11.2`:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.2 -c pytorch
```


  - 
### Data Preparation

* Download the downstream datasets to `./data/*`.



* Fine-tune pretrained models with hyper-parameters sweep to obtain ground-truth transferability score
```bash
python finetune.py -m resnet50 -d cifar10
```

* Extract features of target data using pretrained models
```bash
python forward_feature.py -m resnet50 -d cifar10
```

* Extract features of softmax probabilities using pretrained models
```bash
python cal_tar_probs_CNN.py -m resnet50 -d cifar10
```

* Compute transferability scores using TranSAC
```bash
python evaluate_metric.py -me TranSAC  
```

## Acknowledgement

This code is based on [SFDA (ECCV 2022)](https://github.com/TencentARC/SFDA/tree/main?tab=readme-ov-file#not-all-models-are-equal-predicting-model-transferability-in-a-self-challenging-fisher-space). We thank the authors for releasing their excellent work.
