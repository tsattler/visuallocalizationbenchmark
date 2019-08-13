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

The code was tested on Python 3.7; it should work without issues on Python 3+. [Conda](https://docs.conda.io/en/latest/) can be used to install the missing packages:

```bash
conda install numpy tqdm
conda install pytorch cudatoolkit=9.0 -c pytorch # required for the provided mutual NN matcher
```

### Dataset Preparation
This code currently supports **only the Aachen Day-Night** dataset. Further datasets might be supported in the future. 
For the dataset, we provide two files ``database.db`` and ``image_pairs_to_match.txt``, which are in the ``data/aachen-day-night/`` sub-directory of this repository. You will need to move them to directory where you are storing the Aachen Day-Night dataset. In order for the script to function properly, the directory should have the following structure:

```
.
├── database.db
├── image_pairs_to_match.txt
├── images
│  └── images_upright
├── 3D-models
│  ├── database_intrinsics.txt
│  └── aachen_cvpr2018_db.nvm
└── queries/night_time_queries_with_intrinsics.txt
```
Here,
- `database.db` is an COLMAP database containing all images and intrinsics (this file is **provided in this repository**).

- `image_pairs_to_match.txt` is a list of images to be matched (one pair per line) (this file is **provided in this repository**).

- `images` contains both the database and query images. For Aachen Day-Night, it should contain one sub-folder `images_upright`, which in turn has two subfolders - `db` and `query`. This data is provided by the dataset.

- `3D-models` contains the reference database model in `nvm` format and a list of database images with their COLMAP intrinsics. This data is provided by the dataset.

- `night_time_queries_with_intrinsics.txt` contains the list of query images with their COLMAP intrinsics (the intrinsics are not used since they are supposed to be part of the database already - a list of query images should suffice). **Only the night-time images are currently used**. This data is provided by the dataset.

In order to run the script, you will first need to extract your local features (see below). Then call the script, e.g., via 
```
python reconstruction_pipeline.py 
	--dataset_path /local/aachen 
	--colmap_path /local/colmap/build/src/exe
	--method_name d2-net
```

### Local Features Format

Our script tries to load local features per image from files. It is your responsibility to create these files. 

The local features should be stored in the [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format with two fields:

- `keypoints` - `N x 2` matrix with `x, y` coordinates of each keypoint in COLMAP format (the `X` axis points to the right, the `Y` axis to the bottom)

- `descriptors` - `N x D` matrix with the descriptors (L2 normalized if you plan on using the provided mutual nearest neighbors matcher)

Moreover, they are supposed to be saved alongside their corresponding images with an extension corresponding to the `method_name` (e.g. if `method_name = d2-net` the features for the image `/local/aachen/images/images_upright/db/1000.jpg` should be in the file `/local/aachen/images/images_upright/db/1000.jpg.d2-net`).

**Important information**: In order to work, our script requires that the local features are extracted at the **original image resolutions**. If you downscale the images before feature extraction, you will need to scale the keypoint positions to the original resolutions. **Otherwise, the camera pose estimation stage will fail**.

### Citing

If you use this code in your project, please cite the following paper:

```
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
```

## Using COLMAP with Custom Features
For convenience, and separate from the functionality of the local features benchmark, we also provide functionality to simply import custom local features and matches into any COLMAP database. This allows to use custom features inside COLMAP for arbitrary datasets.

The provided approach assumes that there is a directory ```data_directory/``` that contains the images, an existing COLMAP database (for example ```db.db```), and text file containing pairs of image names to be matched (for example ```match_list.txt```). The database can be created by running COLMAP's feature detection on the images in the database (it cannot be empty). In order to use custom local features, ```.npz``` files containing these features (in the format from above) need to be saved alongside their corresponding images with an extension corresponding to the ```method_name```.

Given these pre-requisites, ```modify_database_with_custom_features_and_matches.py``` can be used to import the features into the database, match images, and run geometric verification. The resulting database can then be used with COLMAP to reconstruct the scene.

An example call to ```modify_database_with_custom_features_and_matches.py```, using D2-Net features, is
```
python modify_database_with_custom_features_and_matches.py
	--dataset_path data_directory/
	--colmap_path /local/colmap/build/src/exe
	--method_name d2-net
	--database_name db.db
	--image_path images/
	--match_list match_list.txt
```
The call assumes that there is a directory ```data_directory/``` that contains an existing COLMAP database ```db.db```, an folder ```images/``` that contains all images (for each image with name ```XX.jpg```, there is assume to be a file ```XX.jpg.d2-net``` containing the D2-Net features for that image), and a text file ```match_list.txt``` that contains the image pairs to be matched. The call will create a new database file ```d2-net.db``` in the ```data_directory/``` directory.

**Important information:**
* The provided approach can currently not be used to extend an existing database where new images were added as it deletes all features and matches in the database. A way around this would be to first export and then re-import the existing matches in the database.
* The call will abort with an error if there is already a database with the name ```d2-net.db``` (or in general ```your_method_name.db```).
* We are currently not providing support for this functionality. It is provided simply for convenience for experienced COLMAP users.
