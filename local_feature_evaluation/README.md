# Visual Localization Benchmark - Local Features

This repository provides a code for evaluating local features in the context of long-term visual localization as part of the [Long-term Visual Localization Benchmark](https://visuallocalization.net/). Given features extracted for images, the code provides functionality to match them. The matches are then used to first triangulate the 3D structure of the scene from a set of database images. The matches between a set of query images and the database images then provide a set of 2D-3D matches that are used to estimate the camera poses of the query images. The output is a submission file that can be uploaded directly to the benchmark's website.

The steps of the reconstruction pipeline are the following:
* generate an empty reconstruction with the parameters of database images, 
* import the features into the database, 
* match the features and import the matches into the database, 
* geometrically verify the matches, 
* triangulate the database observations in the 3D model at **fixed intrinsics**, 
* register the query images at **fixed intrinsics**.

We provide a mutual nearest neighbors matcher implemented in PyTorch as an example - you can replace it by your favorite matching method (e.g. one way nearest neighbors + ratio test).

**Apart from the matcher, please keep the rest of the pipeline intact in order to have as fair a comparison as possible.**

## Usage
### Prerequisites 
The provided code requires that [COLMAP](https://colmap.github.io/), a state-of-the-art Structure-from-Motion pipeline, is available on the system. 

### Dataset Preparation
This code currently supports **only the Aachen Day-Night** dataset. Further datasets might be supported in the future. 
For the dataset, we provide two files ``database.db`` and ``image_pairs_to_match.txt``, which are in the ``data/aachen-day-night/`` sub-directory of this repository. You will need to move them to directory where you are storing the Aachen Day-Night dataset. In order for the script to function properly, the directory should have the following structure:

```
.
├── database.db
├── image_pairs_to_match.txt
├── images
├── model
│  ├── database_intrinsics.txt
│  └── aachen_cvpr2018_db.nvm
└── queries/night_time_queries_with_intrinsics.txt
```
Here,
- `database.db` is an COLMAP database containing all images and intrinsics (this file is **provided in this repository**).

- `image_pairs_to_match.txt` is a list of images to be matched (one pair per line) (this file is **provided in this repository**).

- `images` contains both the database and query images. For Aachen Day-Night, it should contain one sub-folder `images_upright`, which in turn has two subfolders - `db` and `query`. This data is provided by the dataset.

- `3D-models` contains the reference database model in `nvm` format and a list of database images with their COLMAP intrinsics. This data is provided by the dataset.

- `night_time_queries_with_intrinsics.txt` contains the list of query images with their COLMAP intrinsics (the intrinsics are not used since they are supposed to be part of the database already - a list of query images should suffice). This data is provided by the dataset.

In order to run the script, you will first need to extract your local features (see below). Then call the script, e.g., via 
```
python reconstruction_pipeline.py 
	--dataset_path /local/aachen 
	--colmap_path /local/colmap/build/src/exe
	--method_name d2-net
```

## Local Features Format

Our script tries to load local features per image from files. It is your responsibility to create these files. 

The local features should be stored in the [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format with two fields:

- `keypoints` - `N x 2` matrix with `x, y` coordinates of each keypoint in COLMAP format (the `X` axis points to the right, the `Y` axis to the bottom)

- `descriptors` - `N x D` matrix with the descriptors (L2 normalized if you plan on using the provided mutual nearest neighbors matcher)

Moreover, they are supposed to be saved alongside their corresponding images with an extension corresponding to the `method_name` (e.g. if `method_name = d2-net` the features for the image `/local/aachen/images/images_upright/db/1000.jpg` should be in the file `/local/aachen/images/images_upright/db/1000.jpg.d2-net`).

## BibTeX

If you use this code in your project, please cite the following paper:

```
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
```
