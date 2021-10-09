# Deep AUC Maximization on Graph Property Prediction
This repo contains code for OGB submission. Here, we focus on [**ogb-molhiv**](https://ogb.stanford.edu/docs/leader_graphprop/), which is a binary classification task to predict target molecular property, e.g, whether a molecule inhibits HIV virus replication or not. The evaluation metric is **AUROC**. To our best knowledge, this is the first solution to directly optimize AUC score in this task. Our [**AUC-Margin loss**](https://arxiv.org/pdf/2012.03173.pdf) improves baseline (DeepGCN) to **0.8155** and achieves SOTA performance **0.8351** when jointly training with Neural FingerPrints. Our approaches are included in **[LibAUC](https://github.com/Optimization-AI/ICCV2021_DeepAUC)**, which is a ML library for AUC optimization.

## Results on ogbg-molhiv
We present our results on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.3.2) with some strong baselines. 

| Method             |Test AUROC    |Validation AUROC  | Parameters    | Hardware |
| ------------------ |------------------- | ----------------- | -------------- |----------|
| DeepGCN            | 0.7858±0.0117 | 0.8427±0.0063 | 531,976   | Tesla V100 (32GB) |
| Neural FingerPrints| 0.8232±0.0047 | 0.8331±0.0054 | 2,425,102 | Tesla V100 (32GB) |
| Graphormer         | 0.8051±0.0053 | 0.8310±0.0089 | 47,183,04 | Tesla V100 (16GB) |
| **DeepAUC (Ours)**           | 0.8155±0.0057 | 0.8064±0.0072 | ???  | Tesla V100 (32GB) |
| **DeepAUC+FPs (Ours)**     | **0.8351±0.0048** | 0.8236±0.0055 | ???   | Tesla V100 (32GB) |


## Requirements
1. Install base packages:
    ```bash
    Python>=3.7
    Pytorch>=1.9.0
    tensorflow>=2.0.0
    pytorch_geometric>=1.6.0
    ogb>=1.3.2 
    dgl>=0.5.3 
    numpy==1.20.3
    pandas==1.2.5
    scikit-learn==0.24.2
    ```   
2. Install [**LibAUC**](https://github.com/Optimization-AI/LibAUC) for using our loss and optimizer:
    ```bash
    pip install LibAUC
    ```
    
## Training
The training process has two steps: 1) We train a DeepGCN model using our **[AUC-margin loss](https://arxiv.org/abs/2012.03173)** from scratch. 2) We jointly finetuning the pretrained model from (1) with **FingerPrints** models. 
### Training DeepGCN from scratch using **[AUC-margin loss](https://arxiv.org/abs/2012.03173)**:
```
python main_auc.py --use_gpu --conv_encode_edge --num_layers $NUM_LAYERS --block res+ --gcn_aggr softmax_sg --t 1.0 --learn_t --dropout 0.2 --loss auroc \
            --dataset ogbg-molhiv \
            --batch_size 512 \
	    --lr 0.1 \
            --optimizer pesg \
            --gamma 500 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 300
```

### Jointly traininig with FingerPrints Model
Extract fingerprints and train Random Forest by following [PaddleHelix](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv)
```
python extract_fingerprint.py
python random_forest.py
```
Finetuning Our pretrained model with FingerPrints Models using **[AUC-margin loss](https://arxiv.org/abs/2012.03173)**
```
python main_auc.py --use_gpu --conv_encode_edge --num_layers $NUM_LAYERS --block res+ --gcn_aggr softmax_sg --t 1.0 --learn_t --dropout 0.2 --loss auroc \
            --dataset ogbg-molhiv \
            --batch_size 512 \
	    --lr 0.01 \
            --optimizer pesg \
            --gamma 500 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 100
```

## Results
The results (1) improves the original baseline (DeepGCN) from **0.7858 to 0.8155**, which is ~**3%** improvement. The result (2) achieves a higher SOTA performance from **0.8351**, which is ~**1%** improvement over previous baseline. For each stage, we train model by 10 times using different random seeds, e.g., 0 to 9. 

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
- https://github.com/Optimization-AI/LibAUC
- https://github.com/PaddlePaddle/PaddleHelix/tree/dev/competition/ogbg_molhiv
- https://github.com/lightaime/deep_gcns_torch/
- https://ogb.stanford.edu/

