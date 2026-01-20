# TranSAC_PR

This repo provides the implementation of "TranSAC, an unsupervised transferability metric based on domain commonality and task speciality" published in Pattern Recognition:


## Metrics
\[
\text{TranSAC} = H(T) - \frac{1}{K} H(\hat Z \mid T)
\]
- **T**: target representations (e.g., embeddings from a pretrained model)
- **p(\hat Z \mid T)**: soft predictions over **K** source classes
