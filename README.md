# Person Re-identification Benchmark
This repository hosts the codebase for the following work:
Karanam, S., Gou, M., Wu, Z., Rates-Borras, A., Camps, O., & Radke, R. J. (2018). 
[A Systematic Evaluation and Benchmark for Person Re-Identification: Features, Metrics, and Datasets.](https://arxiv.org/abs/1605.09653) IEEE Transactions on Pattern Analysis and Machine Intelligence, accepted February 2018.

Tested on Windows Server 2012 with MATLAB 2016b

### Quick Start
* Clone this repository
* Run a quick example in run_experiment_benchmark.m
* Read the results for VIPeR dataset with WHOS feature and XQDA

### Run other experiments
* Download supported dataset, unzip it and put it under the folder ./Data 
* Download corresponding partition file and put it under the folder ./TrainTestSplits
* Run corresponding prepare_DATANAME.m inside the folder ./Data (if avaliable)
* Change the parameters in run_experiment_benchmark.m 

### Check List for supported/tested feature
* HistLBP
* WHOS
* gBiCov
* LDFV
* ColorTexture\ELF
* LOMO (Windows)
* GOG (Windows)

### Check List for supported/tested metric learning
* FDA
* LFDA
* kLFDA-linear/chi2/chi2-rbf/exp
* XQDA
* MFA
* kMFA-linear/chi2/chi2-rbf/exp
* NFST
* KISSME
* PCCA-linear/chi2/chi2-rbf/exp
* rPCCA-linear/chi2/chi2-rbf/exp
* kPCCA-linear/chi2/chi2-rbf/exp
* PRDC
* SVMML
* kCCA

### Check List for supported/tested multi-shot ranking method
* rnp
* srid
* ahisd

### Check List for supported/tested dataset
* [VIPeR](http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip) Parition included in repo
* [Airport](http://www.northeastern.edu/alert/transitioning-technology/alert-datasets/alert-airport-re-identification-dataset/) Partition comes with dataset
* [DukeMTMC4ReID](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/dataset/DukeMTMC4ReID.zip) Partition comes with dataset
* [Market1501](http://www.liangzheng.org/Project/project_reid.html) [Partition](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/code/reid_benchmark/partition/Partition_market.mat)
* CAVIAR (WHOS feature only) Parition included in repo

### Reference
Please cite the work appropriately for each used feature/metric learning/ranking/dataset 
```
@ARTICLE{8294254, 
author={S. Karanam and M. Gou and Z. Wu and A. Rates-Borras and O. Camps and R. J. Radke}, 
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={A Systematic Evaluation and Benchmark for Person Re-Identification: Features, Metrics, and Datasets}, 
year={2018}, 
keywords={Benchmark testing;Cameras;Feature extraction;Histograms;Image color analysis;Measurement;Probes}, 
doi={10.1109/TPAMI.2018.2807450}, 
ISSN={0162-8828}, 
} 
```

