%%%%%%%%

run demo.m



%%%%%%%%

We use kCCA feature from [2]: 
https://github.com/glisanti/KCCAReId. 

They have fixed split for VIPeR.

You can also find kCCA feature and LOMO feature with same split from "code_cvpr16/data/"
For further score-level fusion, we set our LOMO feature split same as kCCA feature split.

For LOMO feature, we can get reported result 42.28% on VIPeR. (RBF kernel)
For kCCA feature, we can get 46.68%(CHI2 kernel), 45.92% (RBF kernel).
We can get reported score-level fusion result 51% on VIPeR. 

P.S. Performance on kCCA is higher than LOMO, why we haven't report results on kCCA feature?

Because we can only get original kCCA feature for VIPeR and PRID2011 from https://github.com/glisanti/KCCAReId
So we implement kCCA feature extraction by ourselves to extract kCCA feature for CUHK01, CUHK03, Market1501, but results are lower than XQDA+LOMO on CUHK03, so we choose to report results on LOMO and fusion results on LOMO+kCCA for all datasets. (People don't like you use different feature for different datasets)



Reference:
[1]Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler, "Kernel Null Space Methods for Novelty Detection". In CVPR, 2013
[2]G. Lisanti, I. Masi, and A. Del Bimbo. Matching people across camera views using kernel canonical correlation analysis, In Proceedings of the International Conference on Distributed Smart Cameras. ACM, 2014.
