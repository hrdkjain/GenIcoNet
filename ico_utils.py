import kaolin
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../PythonFunctions'))
import python_utils

def output2vertices(subdivisions, output):
    # compute x y corners
    base_height = 2 ** subdivisions
    top_corner_src_y = torch.arange(5) * base_height
    top_corner_src_x = torch.tensor([0])
    bottom_corner_src_y = torch.arange(1, 6) * base_height - 1
    bottom_corner_src_x = torch.tensor([-1])
    corner_src_y = torch.stack((top_corner_src_y, bottom_corner_src_y))
    corner_src_x = torch.stack((top_corner_src_x, bottom_corner_src_x))

    v_tmp = output.view(output.shape[0], output.shape[1], -1)
    # compute pole vertices
    v_poles = torch.mean(output[:, :, corner_src_y, corner_src_x], -1)
    data_v = torch.cat((v_tmp, v_poles), dim=2).transpose(1, 2).contiguous()
    return data_v

def computeDistance(outvertices, refvertices, reffaces, f, mode ='point2point', write_mesh = False, outfaces=None):
    # input should be torch tensor, so treat accordingly
    if write_mesh:
        # create kaolin Meshes
        if outfaces is None:
            outfaces = reffaces
        python_utils.writeOffMesh(f, outvertices, outfaces)
    if mode == 'point2mesh':
        if kaolin.__version__ == '0.1.0':
            refmesh = kaolin.rep.TriangleMesh(refvertices, reffaces)
            if outvertices.is_cuda:
                refmesh.to('cuda')
            dist = kaolin.metrics.mesh.point_to_surface(outvertices, refmesh)
        elif kaolin.__version__ == '0.9.1':
            dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(outvertices[None,:,:], refvertices[None,:,:], reffaces)
            dist = torch.mean(dist).cpu().numpy()
    else:
        dist = None
    return dist

def saveDistance(nameDistPair, path):
    #save the distances as csv
    names = []
    distances = []
    f = open(path+'.csv', 'w')
    f.write('Name,Distance\n')
    for name,dist in nameDistPair:
        f.write('%s,%f\n' %(name,dist))
        names.append(name)
        distances.append(dist)

    plt.figure()
    plt.hist(distances, label=names)
    plt.xlabel('Distance')
    plt.xticks(rotation=30)
    plt.ylabel('Frequency (total=%d)' %len(distances))
    plt.title('Histogram of %s\n(%0.8f \u00B1 %0.8f) (Median: %0.8f))' %(os.path.basename(path),np.mean(distances),np.std(distances), np.median(distances)))
    plt.savefig(path+'.png')
    print('%s: %0.8f +- %0.8f, Median: %0.8f' %(os.path.basename(path),np.mean(distances),np.std(distances), np.median(distances)))


def getEpochNumber(epoch):
    if type(epoch) is int:
        return epoch
    elif type(epoch) is str:
        return int(epoch[1:])
    else:
        ValueError('epoch type not specified')

def get_input_shape(dataset):
    # outputs the data/input shape without the batch size
    data = dataset.__getitem__(0)
    return data[0].shape

def get_output_shape(model, dataset):
    if next(model.parameters()).is_cuda:
        model_is_cuda = True
    else:
        model_is_cuda = False

    input,_ = dataset.__getitem__(0)
    input = input.unsqueeze(0)
    if model_is_cuda:
        input = input.to('cuda')
    else:
        input = input.to('cpu')

    output = model(input)
    return output.shape[1:]

def save_to_file(file, *args, **kwds):
    _, ext = os.path.splitext(file)
    if ext == '.npz':
        np.savez_compressed(file, *args, **kwds)
    elif ext == '.pt':
        torch.save(*args, file)
    else:
        raise ValueError('File format %s not specified for save_to_file' %ext)
