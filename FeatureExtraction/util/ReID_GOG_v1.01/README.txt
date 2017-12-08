INFO:
This is the matlab code of the paper
T. Matsukawa, T. Okabe, E. Suzuki, Y. Sato, 
"Hierarchical Gaussian Descriptor for Person Re-Identification", 
In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.1363-1372, 2016.
 
ver1.01 (2016_9_10)
        --    Added comments to apply_normalization.m

Table of Contents
=================
- Installation
- Usage
- Demo
- DB
- Reference
================
 
Installation
============
Unzip GOG.zip in the folder you like. 
 
If necesarry, compile mex functions by running './GOG/mex/compile.m'
 
Usage
=====
You can extract GOG feature vector for an image as follows. 
 
addpath('GOG/mex');

f = 1; % 1 -- GOG_RGB, 2 -- GOG_Lab, 3 -- GOG_HSV, 4 -- GOG_nRnG
param = set_default_parameter(f); 

I = imread('X.bmp'); % load image 
feature_vec = GOG(I, param); 
 
Note that output of the feature vector is not normalized.
(Because the normalization requires a mean vector of traing dataset). 
Normalization of the feature vector can be performed as follows.  
 
meanX = mean(Xtrain, 1)'; % mean vector( Xtrain: trainig features. Size.[n_train, dim])
feature_vec = (feature_vec - meanX)./norm(feature_vec - meanX, 2); % Mean removal + L2 norm normalization

Demo
=====
Run 'demo_GOG.m' to reproduce the results of the paper. 
You can change dataset by varying sys.database in 'config.m'. 
 
If you run demo, the function of XQDA is required. 
Download LOMO_XQDA.zip from the author's web page http://www.cbsr.ia.ac.cn/users/scliao/projects/lomo_xqda/
Unzip LOMO_XQDA.zip
Copy the .m file of LOMO_XQDA (EvalsCMC.m, MahDist.m, XQDA.m) under the './XQDA/'
 
To run the demo, it is required to download datasets and 
modify the 'datadirname_root' and 'featuredirname_root' in 'set_database.m'.
 
The datasets can be downloaded below. 
    - VIPeR: https://vision.soe.ucsc.edu/node/178
    - CUHK01: http://www.ee.cuhk.edu.hk/~rzhao/
    - PRID450_S: http://lrs.icg.tugraz.at/download.php
    - GRID: http://www.eecs.qmul.ac.uk/~ccloy/downloads_qmul_underground_reid.html
    - CUHK03: http://www.ee.cuhk.edu.hk/~rzhao/
 
CUHK03 is distributed by the MAT file "cuhk-03.mat". We convert the dataset into image files. 
Run "convert_cuhk03.m" if you use CUHK03 dataset. (modify the path in the script) 
 
WARNING: CUHK03 dataset requires large memory (about 30GB) and training time of XQDA (about 4 hours) 
for the evaluation. 
 
DB
==
'./DB' contains datasets information.
    - allimagenames -- all image names of the dataset
    - traininds_set/testinds_set -- index of the training/test images for each division
                                    (index order is the same as allimagenames)
    - trainimagenames_set/testimagenames_set -- image names of the each traing/test division
    - trainlabels_set/testlabels_set -- person IDs of the trainig/test images for each division
    - traincamIDs_set/testcamIDs_set -- camera IDs of the trainng/test images for each division
 
Reference
=========
If you use this code, please cite it as
 
T. Matsukawa, T. Okabe, E. Suzuki, Y. Sato, 
"Hierarchical Gaussian Descriptor for Person Re-Identification", 
In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp.1363-1372, 2016.

