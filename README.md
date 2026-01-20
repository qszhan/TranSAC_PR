# TranSAC_PR

This repo provides the implementation of "TranSAC, an unsupervised transferability metric based on domain commonality and task speciality" published in Pattern Recognition:


## Metrics
\[
\text{TranSAC} = H(T) - \frac{1}{K} H(\hat Z \mid T)
\]
- **T**: target representations (e.g., embeddings from a pretrained model)
- **p(\hat Z \mid T)**: soft predictions over K source classes




## Usage in your code
```python
import numpy as np
from transac import compute_transac

# T: (n, d) embeddings from your model
# tar_prob: (n, K) softmax probabilities from the source classifier head
score = compute_transac(tar_prob, T, eps=10.0, order=5)
print("TranSAC:", score)
```

## Notes / Practical tips
 

* **eps**: choose a relatively large value (paper commonly uses `eps=10`) to help Taylor-series convergence.

* **order**: small orders like `3` or `5` usually work well; higher orders cost more.

* Complexity is driven mainly by the Taylor expansion over an `n x n` Gram matrix (roughly cubic in `n`).
