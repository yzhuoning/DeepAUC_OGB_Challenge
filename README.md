# Deep AUC Maximization on OGB-Molhiv Dataset
This repo contains the implementation of our AUC optimization method on [ogb-molhiv](https://ogb.stanford.edu/docs/leader_graphprop/). Our AUC optimizatin improves original baseline (DeepGCN) from 0.7858 to 0.81555 and achieves SOTA performance, **0.8351**, jointly training with Neural FingerPrints.

## Results on ogbg-molhiv
Here, we show the performance on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.3.2). 

| Model              |Test AUCROC    |Validation AUCROC  | Parameters    | Hardware |
| ------------------ |------------------- | ----------------- | -------------- |----------|
| DeepGCN            | 0.7858 ± 0.0117 | 0.8427 ± 0.0063 | 531,976   | Tesla V100 (32GB) |
| Neural FingerPrints| 0.8232 ± 0.0047 | 0.8331 ± 0.0054 | 2,425,102 | Tesla V100 (32GB) |
| Graphormer         | 0.8051 ± 0.0053 | 0.8310 ± 0.0089 | 47,183,04 | Tesla V100 (16GB) |
| **DeepAUC (Ours)**           | 0.8155±0.0057 | 0.8064±0.0072 | 101,011   | Tesla V100 (32GB) |
| **DeepAUC+NFP (Ours)**     | **0.8351±0.0048** | 0.8236±0.0055 | 101,011   | Tesla V100 (32GB) |


### Requirements

1. Install base packages:
    ```bash
    Python>=3.7
    Pytorch>=1.9.0
    tensorflow>=2.0.0
    pytorch_geometric>=1.6.0
    ogb>=1.3.2 
    dgl>=0.5.3 
    numpy
    pandas
    ```   
2. Install our **`LibAUC`** for training, optimization :
    ```bash
    pip install LibAUC
    ```
    
### Training

1. Random Forest with FingerPrints

First 

First, you should generate fingerprints and train with random forest as mentioned in 《Extended-Connectivity Fingerprints》 and 《GMAN and bag of tricks for graph classification》.

```
python extract_fingerprint.py
python random_forest.py
```
The prediction results will be saved in ../../rf_preds_hiv/rf_final_pred.npy.


2. Train model using LibAUC


3. Jointly train with Neural Fingers


Citation
---------
If you find this work useful, please cite the following paper for our library:
```
@inproceedings{yuan2021robust,
	title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
	author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	year={2021}
	}
```


Reference 
---------
- https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv
- https://github.com/lightaime/deep_gcns_torch/
- 
