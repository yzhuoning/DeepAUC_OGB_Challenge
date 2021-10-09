# Deep AUC Maximization on OGB-Molhiv Dataset
This repo contains the implementation of our AUC optimization method on [ogb-molhiv](https://ogb.stanford.edu/docs/leader_graphprop/). Our AUC optimizatin improves original baseline (DeepGCN) from 0.7858 to 0.81555 and achieves SOTA performance, **0.8351**, jointly training with Neural FingerPrints.

## Results on ogbg-molhiv
Here, we show the performance on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.3.2). 

| Model              |Test AUROC    |Validation AUROC  | Parameters    | Hardware |
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
The training process contains two steps: 1) We train a [DeepGCN]() model using our **[AUC-margin loss](https://arxiv.org/abs/2012.03173)** from scratch. 2) We jointly finetuning the pretrained model from (1) with [FingerPrints]() models.  The results from (1) improves the original baseline from **0.7858 to 0.8155**, which is ~3% improvement. The result (2) improves the SOTA from 0.8244 to 0.8351, which is ~1% improvement. 

1. Training DeepGCN from scratch using **[LibAUC](https://github.com/Optimization-AI/LibAUC)**:
```
python main_auc.py --use_gpu --conv_encode_edge --num_layers $NUM_LAYERS --block res+ --gcn_aggr softmax_sg --t 1.0 --learn_t --dropout 0.2 --loss auroc \
            --dataset $DATA \
            --batch_size 512 \
	    --lr 0.1 \
            --optimizer pesg \
            --gamma 500 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 300
```

2. Jointly traininig with FingerPrints Model
```
python extract_fingerprint.py
python random_forest.py
python main_auc.py --use_gpu --conv_encode_edge --num_layers $NUM_LAYERS --block res+ --gcn_aggr softmax_sg --t 1.0 --learn_t --dropout 0.2 --loss auroc \
            --dataset $DATA \
            --batch_size 512 \
	    --lr 0.01 \
            --optimizer pesg \
            --gamma 500 \
            --margin 1.0 \
            --weight_decay 1e-5 \
            --random_seed 0 \
            --epochs 100
```

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
