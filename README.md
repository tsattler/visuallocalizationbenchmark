# Long-Term Visual Localization Benchmark
This is the Github repository for the long-term localization benchmark hosted at http://visuallocalization.net/. 
The main purpose of the repository is to allow easy discussion and the reporting of issues.

We also use this repository to point to publicly available implementations of visual localization methods:
* Active Search: https://www.graphics.rwth-aachen.de/software/image-localization
* DenseVLAD: http://www.ok.ctrl.titech.ac.jp/~torii/project/247/
* NetVLAD: https://www.di.ens.fr/willow/research/netvlad/
* DenseSFM: http://www.ok.sc.e.titech.ac.jp/res/DenseSfM/index.html
* InLoc: http://www.ok.sc.e.titech.ac.jp/INLOC/

You can find information on how to use custom features inside a retrieval + pose estimation pipeline in the `local_feature_evaluation/` subdirectory. More specifically, the section `Using COLMAP for Visual Localization with Custom Features` in that directory's README file provides the information necessary to implement such a pipeline using COLMAP and an external retrieval approach.
