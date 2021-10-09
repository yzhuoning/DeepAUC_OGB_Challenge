# Deep AUC Maximization on OGB-Molhiv Dataset
This repo is our implementation for reproducing the our result in leaderboard. 


## Results on ogbg-molhiv
Here, we demonstrate the following performance on the ogbg-molhiv dataset from Stanford Open Graph Benchmark (1.3.0)

| Model              |Test AUCROC    |Validation AUCROC  | Parameters    | Hardware |
| ------------------ |------------------- | ----------------- | -------------- |----------|
| DeepGCN            | 0.7858 ± 0.0117  | 0.8427 ± 0.0063 | 531,976 | Tesla V100 (32GB) |
| Neural FingerPrints| 0.8232 ± 0.0047   | 0.8331 ± 0.0054 | 2,425,102 | Tesla V100 (32GB) |
| Graphormer         | 0.8051 ± 0.0053 | 0.8310 ± 0.0089 | 47,183,04 | Tesla V100 (16GB) |
| DeepAUC            |     0.xxxx ± 0.xxxx | 0.xxxx ± 0.xxxx | 101,011| Tesla V100 (32GB) |
| DeepAUC+RFs        |     0.8351 ± 0.xxxx | 0.8236 ± 0.xxxx | 101,011| Tesla V100 (32GB) |

	
		

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
