# Visual Localization Benchmark - Local Features

This repository provides a sample script for testing local features on the [
Long-term Visual Localization Benchmark](https://visuallocalization.net/). The final output is a submission file that can be uploaded directly to the benchmark's website.

The steps of the reconstruction pipeline are the following: generate an empty reconstruction with the parameters of database images, import the features into the database, match the features and import the matches into the database, geometrically verify the matches, triangulate the database observations in the 3D model at **fixed intrinsics**, and, finally, register the query images at **fixed intrinsics**.

We provide a mutual nearest neighbors matcher implemented in PyTorch as an example - you can replace it by your favorite matching method (e.g. one way nearest neighbors + ratio test).

**Apart from the matcher, please keep the rest of the pipeline intact in order to have as fair a comparison as possible.**

#### Usage example

```
python reconstruction_pipeline.py 
	--dataset_path /local/aachen 
	--colmap_path /local/colmap/build/src/exe
	--method_name d2-net
```

## Dataset Format

In order for the stript to function properly, the dataset should have the following format:

```
.
├── database.db
├── image_pairs_to_match.txt
├── images
├── model
│  ├── database_intrinsics.txt
│  └── database_model.nvm
└── queries_with_intrinsics.txt
```

- `database.db` is an COLMAP database containing all images and intrinsics.

- `image_pairs_to_match.txt` is a list of images to be matched (one pair per line).

- `images` contains both the database and query images (e.g. for Aachen Day-Night, `images` has two subfolders - `db` and `query`).

- `model` contains the reference database model in `nvm` format and a list of database images with their COLMAP intrinsics.

- `queries with intrinsics.txt` contains the list of query images with their COLMAP intrinsics (the intrinsics are not used since they are supposed to be part of the database already - a list of query images should suffice).

## Local Features Format

The local features should be in [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format with two fields:

- `keypoints` - `N x 2` matrix with `x, y` coordinates of each keypoint in COLMAP format (the `X` axis points to the right, the `Y` axis to the bottom)

- `descriptors` - `N x D` matrix with the descriptors (L2 normalized if you plan on using the provided mutual nearest neighbors matcher)

Moreover, they are supposed to be saved alongside their corresponding images with an extension corresponding to the `method_name` (e.g. if `method_name = d2-net` the features for the image `/local/aachen/images/db/1000.jpg` should be in the file `/local/aachen/images/db/1000.jpg.d2-net`).

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
