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


def import_features(images, paths, args):
    # Connect to the database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()
    
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    connection.commit()

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
        
        #print(features_path1, features_path2)
        D1 = np.load(features_path1)['descriptors'];
        D2 = np.load(features_path2)['descriptors'];
        D1 = D1[:min(D1.shape[0],25000),:];
        D2 = D2[:min(D2.shape[0],25000),:];
        #print(D1.shape, D2.shape)

        #descriptors1 = torch.from_numpy(np.load(features_path1)['descriptors']).to(device)
        #descriptors2 = torch.from_numpy(np.load(features_path2)['descriptors']).to(device)
        descriptors1 = torch.from_numpy(D1).to(device)
        descriptors2 = torch.from_numpy(D2).to(device)
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
    
    # Close the connection to the database.
    connection.commit()
    cursor.close()
    connection.close()


def geometric_verification(paths, args):
    print('Running geometric verification...')

    subprocess.call([os.path.join(args.colmap_path, 'colmap'), 'matches_importer',
                     '--database_path', paths.database_path,
                     '--match_list_path', paths.match_list_path,
                     '--match_type', 'pairs'])



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--colmap_path', required=True, help='Path to the COLMAP executable folder')
    parser.add_argument('--method_name', required=True, help='Name of the method')
    parser.add_argument('--database_name', required=True, help='Name of the COLMAP database relative to dataset_path')
    parser.add_argument('--image_path', required=True, help='Name of the image directory relative to dataset_path')
    parser.add_argument('--match_list', required=True, help='Name of the text file containing image pairs to be matched, relative to dataset_path')
    parser.add_argument('--matching_only', type=bool, default=False, help='Only performs feature matching without creating a new database or importing features')
    args = parser.parse_args()

    ## Torch settings for the matcher.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.dummy_database_path = os.path.join(args.dataset_path, args.database_name)
    paths.database_path = os.path.join(args.dataset_path, args.method_name + '.db')
    paths.image_path = os.path.join(args.dataset_path, args.image_path)
    paths.features_path = os.path.join(args.dataset_path, args.method_name)
    paths.match_list_path = os.path.join(args.dataset_path, args.match_list)
    
    if args.matching_only == False:
      # Create a copy of the dummy database.
      if os.path.exists(paths.database_path):
        raise FileExistsError('The database file already exists for method %s.' % args.method_name)
      shutil.copyfile(paths.dummy_database_path, paths.database_path)
    
    # Reconstruction pipeline.
    images, cameras = recover_database_images_and_ids(paths, args)
    if args.matching_only == False:
      import_features(images, paths, args)
      
    match_features(images, paths, args)
    geometric_verification(paths, args)
