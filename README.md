# Deep AUC Maximization on OGB-Molhiv Dataset
This repo is our implementation for reproducing the our result in leaderboard. 


## Results on ogbg-molhiv
Here, we demonstrate the following performance on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.3.0)

| Model              |Test AUCROC    |Validation AUCROC  | Parameters    | Hardware |
| ------------------ |-------------------   | ----------------- | -------------- |----------|
| DeepGCN            |     0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 101,011| Tesla V100 (32GB) |
| Random Forest      |     0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 101,011| Tesla V100 (32GB) |
| DeepAUC            |     0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 101,011| Tesla V100 (32GB) |
| DeepAUC+RFs        |     0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 101,011| Tesla V100 (32GB) |


### Requirements

1. Install base packages:
    ```bash
    pip install tensorflow>=2.0.0         
    pip install pytorch > 1.8.0     
    ```   
2. Install `ogb` for datasets, evaluation:
    ```bash
    pip install ogb
    ```
3. Install `LibAUC` for training, optimization :
    ```bash
    pip install LibAUC
    ```
    
### Training

1. extract neural figures for 


2. Train model using LibAUC


3. Jointly train with Neural Fingers



### Reference 
- https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv
- https://github.com/lightaime/deep_gcns_torch/
- 
