import argparse

import numpy as np

import os

import shutil

import subprocess

import sqlite3

import torch

import types

from tqdm import tqdm

from matchers import mutual_nn_matcher

from camera import Camera

from utils import quaternion_to_rotation_matrix, camera_center_to_translation

import sys
IS_PYTHON3 = sys.version_info[0] >= 3

def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def recover_database_images_and_ids(paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cameras = {}
    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[2]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images, cameras


def preprocess_reference_model(paths, args):
    print('Preprocessing the reference model...')
    
    # Recover intrinsics.
    with open(os.path.join(paths.reference_model_path, 'database_intrinsics.txt')) as f:
        raw_intrinsics = f.readlines()
    
    camera_parameters = {}

    for intrinsics in raw_intrinsics:
        intrinsics = intrinsics.strip('\n').split(' ')
        
        image_name = intrinsics[0]
        
        camera_model = intrinsics[1]

        intrinsics = [float(param) for param in intrinsics[2 :]]

        camera = Camera()
        camera.set_intrinsics(camera_model=camera_model, intrinsics=intrinsics)

        camera_parameters[image_name] = camera
    
    # Recover poses.
    with open(os.path.join(paths.reference_model_path, 'aachen_cvpr2018_db.nvm')) as f:
        raw_extrinsics = f.readlines()

    # Skip the header.
    n_cameras = int(raw_extrinsics[2])
    raw_extrinsics = raw_extrinsics[3 : 3 + n_cameras]

    for extrinsics in raw_extrinsics:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[0]

        # Skip the focal length. Skip the distortion and terminal 0.
        qw, qx, qy, qz, cx, cy, cz = [float(param) for param in extrinsics[2 : -2]]

        qvec = np.array([qw, qx, qy, qz])
        c = np.array([cx, cy, cz])
        
        # NVM -> COLMAP.
        t = camera_center_to_translation(c, qvec)

        camera_parameters[image_name].set_pose(qvec=qvec, t=t)
    
    return camera_parameters


def generate_empty_reconstruction(images, cameras, camera_parameters, paths, args):
    print('Generating the empty reconstruction...')

    if not os.path.exists(paths.empty_model_path):
        os.mkdir(paths.empty_model_path)
    
    with open(os.path.join(paths.empty_model_path, 'cameras.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s\n' % (
                camera_id, 
                camera.camera_model, 
                ' '.join(map(str, camera.intrinsics))
            ))

    with open(os.path.join(paths.empty_model_path, 'images.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            try:
                camera = camera_parameters[image_name]
            except:
                continue
            f.write('%d %s %s %d %s\n\n' % (
                image_id, 
                ' '.join(map(str, camera.qvec)), 
                ' '.join(map(str, camera.t)), 
                camera_id,
                image_name
            ))

    with open(os.path.join(paths.empty_model_path, 'points3D.txt'), 'w') as f:
        pass


def import_features(images, paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Import the features.
    print('Importing features...')
    
    for image_name, image_id in tqdm(images.items(), total=len(images.items())):
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))
        
        keypoints = np.load(features_path)['keypoints']
        n_keypoints = keypoints.shape[0]
        
        # Keep only x, y coordinates.
        keypoints = keypoints[:, : 2]
        # Add placeholder scale, orientation.
        keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)
        
        keypoints_str = keypoints.tostring()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))
        connection.commit()
    
    # Close the connection to the database.
    cursor.close()
    connection.close()


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def match_features(images, paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()

    # Match the features and insert the matches in the database.
    print('Matching...')

    with open(paths.match_list_path, 'r') as f:
        raw_pairs = f.readlines()
    
    image_pair_ids = set()
    for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
        image_name1, image_name2 = raw_pair.strip('\n').split(' ')
        
        features_path1 = os.path.join(paths.image_path, '%s.%s' % (image_name1, args.method_name))
        features_path2 = os.path.join(paths.image_path, '%s.%s' % (image_name2, args.method_name))

        descriptors1 = torch.from_numpy(np.load(features_path1)['descriptors']).to(device)
        descriptors2 = torch.from_numpy(np.load(features_path2)['descriptors']).to(device)
        matches = mutual_nn_matcher(descriptors1, descriptors2).astype(np.uint32)

        image_id1, image_id2 = images[image_name1], images[image_name2]
        image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]
        
        matches_str = matches.tostring()
        cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
        connection.commit()
    
    # Close the connection to the database.
    cursor.close()
    connection.close()


def geometric_verification(paths, args):
    print('Running geometric verification...')

    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'matches_importer',
                     '--database_path', paths.database_path,
                     '--match_list_path', paths.match_list_path,
                     '--match_type', 'pairs'])


def reconstruct(paths, args):
    if not os.path.isdir(paths.database_model_path):
        os.mkdir(paths.database_model_path)
    
    # Reconstruct the database model.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'point_triangulator',
                     '--database_path', paths.database_path,
                     '--image_path', paths.image_path,
                     '--input_path', paths.empty_model_path,
                     '--output_path', paths.database_model_path,
                     '--Mapper.ba_refine_focal_length', '0',
                     '--Mapper.ba_refine_principal_point', '0',
                     '--Mapper.ba_refine_extra_params', '0'])


def register_queries(paths, args):
    if not os.path.isdir(paths.final_model_path):
        os.mkdir(paths.final_model_path)
    
    # Register the query images.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'image_registrator',
                     '--database_path', paths.database_path,
                     '--input_path', paths.database_model_path,
                     '--output_path', paths.final_model_path,
                     '--Mapper.ba_refine_focal_length', '0',
                     '--Mapper.ba_refine_principal_point', '0',
                     '--Mapper.ba_refine_extra_params', '0'])


def recover_query_poses(paths, args):
    print('Recovering query poses...')

    if not os.path.isdir(paths.final_txt_model_path):
        os.mkdir(paths.final_txt_model_path)

    # Convert the model to TXT.
    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'model_converter',
                     '--input_path', paths.final_model_path,
                     '--output_path', paths.final_txt_model_path,
                     '--output_type', 'TXT'])
    
    # Recover query names.
    query_image_list_path = os.path.join(args.dataset_path, 'queries/night_time_queries_with_intrinsics.txt')
    
    with open(query_image_list_path) as f:
        raw_queries = f.readlines()
    
    query_names = set()
    for raw_query in raw_queries:
        raw_query = raw_query.strip('\n').split(' ')
        query_name = raw_query[0]
        query_names.add(query_name)

    with open(os.path.join(paths.final_txt_model_path, 'images.txt')) as f:
        raw_extrinsics = f.readlines()

    f = open(paths.prediction_path, 'w')

    # Skip the header.
    for extrinsics in raw_extrinsics[4 :: 2]:
        extrinsics = extrinsics.strip('\n').split(' ')

        image_name = extrinsics[-1]

        if image_name in query_names:
            # Skip the IMAGE_ID ([0]), CAMERA_ID ([-2]), and IMAGE_NAME ([-1]).
            f.write('%s %s\n' % (image_name.split('/')[-1], ' '.join(extrinsics[1 : -2])))

    f.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--colmap_path', required=True, help='Path to the COLMAP executable folder')
    parser.add_argument('--method_name', required=True, help='Name of the method')
    args = parser.parse_args()

    # Torch settings for the matcher.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(args.dataset_path, 'database.db')
    paths.database_path = os.path.join(args.dataset_path, args.method_name + '.db')
    paths.image_path = os.path.join(args.dataset_path, 'images', 'images_upright')
    paths.features_path = os.path.join(args.dataset_path, args.method_name)
    paths.reference_model_path = os.path.join(args.dataset_path, '3D-models')
    paths.match_list_path = os.path.join(args.dataset_path, 'image_pairs_to_match.txt')
    paths.empty_model_path = os.path.join(args.dataset_path, 'sparse-%s-empty' % args.method_name)
    paths.database_model_path = os.path.join(args.dataset_path, 'sparse-%s-database' % args.method_name)
    paths.final_model_path = os.path.join(args.dataset_path, 'sparse-%s-final' % args.method_name)
    paths.final_txt_model_path = os.path.join(args.dataset_path, 'sparse-%s-final-txt' % args.method_name)
    paths.prediction_path = os.path.join(args.dataset_path, 'Aachen_eval_[%s].txt' % args.method_name)
    
    # Create a copy of the dummy database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('The database file already exists for method %s.' % args.method_name)
    shutil.copyfile(paths.dummy_database_path, paths.database_path)
    
    # Reconstruction pipeline.
    camera_parameters = preprocess_reference_model(paths, args)
    images, cameras = recover_database_images_and_ids(paths, args)
    generate_empty_reconstruction(images, cameras, camera_parameters, paths, args)
    import_features(images, paths, args)
    match_features(images, paths, args)
    geometric_verification(paths, args)
    reconstruct(paths, args)
    register_queries(paths, args)
    recover_query_poses(paths, args)
