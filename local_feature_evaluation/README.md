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

# Using COLMAP with Custom Features
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
* If the existing database contains feature matches, especially verified matches, then this will cause problems. Make sure that the database does not contain such matches, e.g., by clearing them inside COLMAP's gui.

# Using COLMAP for Visual Localization with Custom Features
The above pipeline can be used to implement the following visual localization pipeline in COLMAP using custom features:
1. For each query image, find the top-k most similar database images.
2. Match custom features between each query and its top-k retrieved images.
3. Estimate the poses of the query images with respect to a 3D model based on the matches from step 2.

The following describes how to implement steps 2 and 3 in COLMAP.

### Prerequisites
* We assume that image retrieval is implemented elsewhere, e.g., using DenseVLAD or NetVLAD, and that text file containing pairs of query file names and database file names (one pair per line, see COLMAP's "Custom Matching" format for image pairs [here](https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification)) is available. In the following, we will refer to this file as `retrieval_list.txt`.
* A COLMAP database (e.g., `db.db`) containing the database and query images as well as their intrinsics. This database can be constructed by extracting SIFT features. **Important**: Make sure that the correct camera intrinsics that should be used for pose estimation for the query images are stored inside the database.
* A COLMAP sparse reconstruction consistent with the database, i.e., the IMAGE_IDs stored in `images.bin` / `images.txt` are consistent with the IMAGE_IDs stored in the database. The sparse model can be either a complete reconstruction, e.g., a 3D model obtained from SIFT features, or an "empty" reconstruction, i.e., a model that only contains camera poses and camera intrinsics but no 3D points.
* A text file containing pairs of database images that should be matched. For example, this could be an exhaustive list of pairs or the pairs of database images you matched when building a 3D model (see above). The format is the same as for `retrieval_list.txt`. In the following, we will refer to this list as `database_pairs_list.txt`.

### Localization
1. *Extract custom features*: Run your feature extractor and store the features in the local feature format described above. You should extract features for both database and query features.
2. *Import all features and match database images*: This is done using the `modify_database_with_custom_features_and_matches.py` tool provided in this repository (please see above for details) by calling (as an example):
```
python modify_database_with_custom_features_and_matches.py
	--dataset_path data_directory/
	--colmap_path /local/colmap/build/src/exe
	--method_name your_feature_name
	--database_name db.db
	--image_path images/
	--match_list database_pairs_list.txt
```
Here, `data_directory/` is the directory containing all the data and `db.db` is the name of the existing database (i.e., `data_directory/db.db` is the path of this file). `your_feature_name` is the name of your features (see above) and the directory `data_directory/images/` contains the query and database images (potentially in subdirectories). It is assumed that `database_pairs_list.txt` is located in `data_directory/`, i.e., that the file is `data_directory/database_pairs_list.txt`. This call will create a new database `data_directory/your_feature_name.db` from `db.db` (without altering `db.db`), will imported the custom features to `your_feature_name.db`, match the pairs of database images specified in `database_pairs_list.txt`, perform spatial verification, and store the matches in `your_feature_name.db`. This step can take a long time. **Important**: Make sure that `data_directory/your_feature_name.db` does not exist prior to the call.

3. *Build a database 3D model*: In the next step, we build the 3D model from the custom features and the matches that will be used for localization. This is done by calling 
```
colmap point_triangulator --database_path data_directory/your_feature_name.db --image_path data_directory/images/ --input_path data_directory/existing_model/ --output_path data_directory/model_your_feature_name/ --clear_points 1
```
Here, `data_directory/existing_model/` is the existing 3D model mentioned in the prerequisites above. The resulting 3D model will be stored in `data_directory/model_your_feature_name/`. Make sure that this directory exists before the call. Note that this call does not run Structure-from-Motion from scratch but rather only triangulates the 3D structure of the scene from known poses for the database images. The poses and intrinsics of the database images are not changed.

4. *Perform feature matching between the query images and the retrieved database images*: This can be done by calling
```
python modify_database_with_custom_features_and_matches.py
	--dataset_path data_directory/
	--colmap_path /local/colmap/build/src/exe
	--method_name your_feature_name
	--database_name db.db
	--image_path images/
	--match_list retrieval_list.txt
	--matching_only True

```
This will perform feature matching between the query images and the top-k retrieved database images (as specified by the pairs in `retrieval_list.txt`) and import the results after geometric verification into `data_directory/your_feature_name.db`. **Important**: The actual custom feature descriptors are not stored in the COLMAP database. For matching, the features are loaded from the files generated in step 1, so make sure that you did not delete them before this step. After this step, you can delete them.

5. *Estimate the camera poses of the query images*: This final step uses the model build in step 3 and the matches from step 4 for camera pose estimation in COLMAP. This is achieved by calling
```
colmap image_registrator --database_path data_directory/your_feature_name.db --input_path data_directory/model_your_feature_name/ --output_path data_directory/model_your_feature_name_with_queries/ 
```
This will register the images into the 3D model without altering it. The result is a 3D model containing the original 3D structure and database poses from `data_directory/model_your_feature_name/` as well as the poses of the query images (and their inlier 2D-3D matches) that could be localized. The result is stored as a COLMAP binary model in `data_directory/model_your_feature_name_with_queries/` (the path has to exist before the call). Afterwards, you can extract the camera poses from `data_directory/model_your_feature_name_with_queries/images.bin` (see [here](https://colmap.github.io/format.html#binary-file-format) for details). You might want to adjust some of COLMAP's parameters to improve performance. For example, for D2-Net on the Extended CMU Seasons dataset, we used the following settings: `--Mapper.min_num_matches 4 --Mapper.init_min_num_inliers 4 --Mapper.abs_pose_min_num_inliers 4 --Mapper.abs_pose_min_inlier_ratio 0.05 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0 --Mapper.ba_local_max_num_iterations 50 --Mapper.abs_pose_max_error 20 --Mapper.filter_max_reproj_error 12`.
