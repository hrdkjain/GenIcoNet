import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat
from scipy.sparse import coo_matrix
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../IcosahedralCNN'))
from icocnn.utils.ico_geometry import get_icosahedral_grid
sys.path.append(os.path.join(os.path.dirname(__file__), '../PythonFunctions'))
from mesh.utils import compute_adjacency_matrix_sparse, compute_laplacian
import python_utils

normalization_mode = ['none', 'unitsphere']
out_features = ['pos', 'pos_nor', 'pos_nor_lap']


def mesh_vertexnormals(vertices, faces, weight_face_area=True, eps=1e-10):
    # get face vertices
    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]

    # compute unnormalized (area weighted) face normals
    f_normals = np.cross(v1 - v0, v2 - v0, axis=1)
    if not weight_face_area:
        # normalize
        magnitude = np.clip(np.sqrt(np.sum(f_normals**2, axis=1)), a_min=eps, a_max=None)[:, np.newaxis]
        f_normals = np.divide(f_normals, magnitude)

    # accumulate vertex normals from face normals
    v_normals = np.zeros_like(vertices)
    v_normals[faces[:, 0], :] += f_normals
    v_normals[faces[:, 1], :] += f_normals
    v_normals[faces[:, 2], :] += f_normals

    # normalize
    magnitude = np.clip(np.sqrt(np.sum(v_normals ** 2, axis=1)), a_min=eps, a_max=None)[:, np.newaxis]
    v_normals = np.divide(v_normals, magnitude)

    return v_normals


def get_normalize_unitsphere(points_in):
    centroid = np.mean(points_in, axis=0)
    furthest_distance = np.max(np.sqrt(np.sum(np.abs(points_in - centroid) ** 2, axis=1)))
    return centroid, furthest_distance


def read_sparseweights(mat_file):
    matdict = loadmat(mat_file)
    sparse_indices = matdict["sparse_indices"]
    sparse_indices = sparse_indices.astype(dtype=np.int32) - 1

    sparse_weights = matdict["sparse_weights"]
    sparse_weights = sparse_weights.astype(dtype=np.float32)

    return sparse_indices, sparse_weights


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        help="folder for loading data models. Expects off files in that folder",
                        type=str,
                        required=True)
    parser.add_argument("--samp_weights_dir",
                        help="folder for loading sampling weights.Expects sparse matrix as .mat file",
                        type=str,
                        required=True)
    parser.add_argument("--out_dir",
                        help="folder for saving the ground truth models. Will be npz files in that folder",
                        type=str,
                        required=True)
    parser.add_argument("--normalization_mode",
                        help="the normalization mode (default: {})".format(normalization_mode[0]),
                        choices=normalization_mode,
                        type=str,
                        default=normalization_mode[0],
                        required=False)
    parser.add_argument("--nested_dir",
                        help="the nested folder level (default: 0)",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--subdivision",
                        help="the icosahedral grid subdivision (default: 5)",
                        type=int,
                        default=5,
                        required=False)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir), 'data_dir does not exist'
    assert os.path.exists(args.samp_weights_dir), 'samp_weights_dir does not exist'

    if args.nested_dir == 2:
        # ModelNet
        tmp_data_dir, tmp_samp_weights_dir = args.data_dir, args.samp_weights_dir
        args.data_dir, args.samp_weights_dir = [],[]
        for f1 in os.listdir(tmp_data_dir):
            for f2 in os.listdir(os.path.join(tmp_data_dir,f1)):
                args.data_dir.append(os.path.join(tmp_data_dir,f1,f2))
                args.samp_weights_dir.append(os.path.join(tmp_samp_weights_dir,f1,f2))
    elif args.nested_dir == 1:
        # SHREC
        tmp_data_dir, tmp_samp_weights_dir = args.data_dir, args.samp_weights_dir
        args.data_dir, args.samp_weights_dir = [],[]
        for f in os.listdir(tmp_data_dir):
            args.data_dir.append(os.path.join(tmp_data_dir,f))
            args.samp_weights_dir.append(os.path.join(tmp_samp_weights_dir,f))
    else:
        # ShapeNetCore Class-wise
        args.data_dir = [args.data_dir]
        args.samp_weights_dir = [args.samp_weights_dir]

    for data_dir, samp_weights_dir in zip(args.data_dir, args.samp_weights_dir):
        out_dir = args.out_dir

        # get file lists
        data_files = os.listdir(data_dir)
        if not data_files:
            print('no files found in {0}'.format(data_dir))
            continue
        data_files = [f for f in data_files if f.endswith('.off')]
        print("Found {0} off files in data_dir ({1}) to process".format(len(data_files), data_dir))

        weights_files = os.listdir(samp_weights_dir)
        if not weights_files:
            print('no files found in {0}'.format(samp_weights_dir))
            continue
        weights_files = [f for f in weights_files if f.endswith('.mat')]
        print("Found {0} mat files in samp_weights_dir ({1}) to process".format(len(weights_files), samp_weights_dir))

        # compose list of sampled meshes
        id_data = [int(os.path.splitext(os.path.split(f)[1])[0].split('_')[1]) for f in data_files]
        id_sw = [int(os.path.splitext(os.path.split(f)[1])[0].split('_')[1]) for f in weights_files]
        _, idx_sw, idx_data = np.intersect1d(id_sw, id_data, return_indices=True)

        # make out dir
        if args.nested_dir == 2:
            out_dir = os.path.join(args.out_dir, '{0}'.format(os.path.basename(os.path.dirname(samp_weights_dir))), '{0}'.format(os.path.basename(samp_weights_dir)))
        elif args.nested_dir == 1:
            out_dir = os.path.join(args.out_dir, '{0}'.format(os.path.basename(samp_weights_dir)))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # create icosahedral data
        ico_v, ico_f = get_icosahedral_grid(args.subdivision)
        adj_sparse = compute_adjacency_matrix_sparse(ico_v.shape[0], torch.from_numpy(ico_f))

        for i in tqdm(range(idx_sw.size)):
            # compose file paths
            f_data = data_files[idx_data[i]]
            f_data_abs = os.path.join(data_dir, f_data)

            f_sw = weights_files[idx_sw[i]]
            f_sw_abs = os.path.join(samp_weights_dir, f_sw)

            f_out_absolute = os.path.join(out_dir, f_sw[:-4] + '.npz')
            if os.path.exists(f_out_absolute):
                continue

            # load files and convert no numpy
            data_v, _ = python_utils.read_off(f_data_abs)
            data_v = np.asarray(data_v, dtype=np.float32)

            sparse_indices, sparse_weights = read_sparseweights(f_sw_abs)
            sparse = coo_matrix((sparse_weights.flatten(),
                                 (sparse_indices[:, 0].flatten(), sparse_indices[:, 1].flatten())),
                                shape=(ico_v.shape[0], data_v.shape[0]))

            # sample meshes (by multiplying with sampling weights matrix)
            data_samp_v = sparse.dot(data_v)

            # sanity check
            if np.any(np.isnan(data_samp_v)):
                print('ERROR file {0}, NaNs in the data'.format(f_data_abs))
                continue

            # normalize vertices if requested
            if args.normalization_mode == normalization_mode[1]:
                # normalize to unit sphere
                data_centroid, data_scale = get_normalize_unitsphere(data_samp_v)
                gt_centroid, gt_scale = get_normalize_unitsphere(gt_samp_v)
                scale = max(data_scale, gt_scale)

                data_samp_v = (data_samp_v - data_centroid) / scale
                gt_samp_v = (gt_samp_v - gt_centroid) / scale

            # compute normals
            data_samp_vnormals = mesh_vertexnormals(data_samp_v, ico_f)

            # compute laplacian
            data_samp_lap = compute_laplacian(torch.from_numpy(data_samp_v).type(torch.float32), adj_sparse).numpy()

            # concatenate vertices, normals, laplacian of groundtruth
            data_samp = np.concatenate((data_samp_v, data_samp_vnormals, data_samp_lap), axis=1)

            # store data & groundtruth
            np.savez(f_out_absolute, data=data_samp.transpose())

if __name__ == '__main__':
    main()
