import natsort
import numpy as np
import os
import scipy
import torch

def listFiles(params, data_type, data_instance):
    fullFileList = []
    if params['ico']['dataPthLvl'] == 1:
        # SHREC
        # OR ELSE list from the directory
        if data_type == 'enc' or data_type == 'ftr':
            dataPth = os.path.join(params[data_type]['dataPth'], data_instance)
        else:
            dataPth = params[data_type]['dataPth']
        print('Listing files from %s dir' %dataPth)
        fileList = []
        for file in natsort.natsorted(os.listdir(dataPth)):
            if file.endswith(params[data_type]['ext']):
                fileList.append(file)
        fullFileList = [os.path.join(dataPth, f) for f in fileList]
    elif params['ico']['dataPthLvl'] == 2:
        # ModelNet
        dataPths = os.listdir(params[data_type]['dataPth'])
        if data_instance == 'trn':
            data_instance = 'train'
        elif data_instance == 'val':
            data_instance = 'test'
        for dataPth in dataPths:
            tmp = os.path.join(params[data_type]['dataPth'], dataPth, data_instance)
            fileList = []
            for file in natsort.natsorted(os.listdir(tmp)):
                if file.endswith(params[data_type]['ext']):
                    fileList.append(file)
            fullFileList += [os.path.join(tmp, f) for f in fileList]
    return fullFileList

def loadEncFile(params, inFile):
    _, ext = os.path.splitext(inFile)
    if ext == '.npz':
        tmp = np.load(inFile)
    else:
        raise ValueError('File format %s not specified for loadEncFile' %ext)
    return torch.tensor(tmp['arr_0'])

def loadIcoFile(params, inFile):
    if params['ico']['ext'] =='.mat':
        # loads the icosahedron mesh saved as mat
        lbl = scipy.io.loadmat(inFile)
        if 'variable' in lbl:
            lbl = lbl['variable']
            lbl = np.swapaxes(lbl, 0, 2)
            lbl = np.swapaxes(lbl, 1, 2)
            lbl[0:3, :, :] /= 255.0
            lbl[3:6, :, :] = lbl[0:3, :, :]
            lbl = lbl.astype(np.float32)
            assert not np.isnan(lbl.all()), inFile
            return lbl, lbl
        elif 'sparse_weights' in lbl:
            raise ValueError('mat file with sparse_weights and sparse_vertices cannot be handled here, use generate.py')
        else:
            raise ValueError('content of mat file unhandleable')

    elif params['ico']['ext']=='.npz':
        # shape (3+3+3,10242)
        lbl2 = np.load(inFile)["data"]  # Load the target with vertices, normals, lap (if available)
        lbl1 = lbl2[:3, :-2]    # Only the vertices without poles
        lbl1 = lbl1.reshape(lbl1.shape[0], -1, params['ico']['width'])
        return lbl1, lbl2
    else:
        return ValueError('ico loader for %s not specified' %params['ico']['ext'])


class createico2icoDataset(torch.utils.data.Dataset):
    def __init__(self, params, data_instance):
        self.params = params
        self.icoList = listFiles(self.params, 'ico', data_instance)
        self.icoPair = []
        for icoFile in self.icoList:
            self.icoPair.append(loadIcoFile(self.params, icoFile))
        if self.params['process_name'] == 'test':
            self.outIcoPth = os.path.join(self.params['out']['dataPth'], self.params[self.params['model_name']]['data_instance'])
            if not os.path.exists(self.outIcoPth):
                os.makedirs(self.outIcoPth)
                print('created dir: %s' % self.outIcoPth)

        print('Listed %d ico2ico pairs for data_instance: %s' %(len(self.icoList), data_instance))

    def __getitem__(self, idx):
        ico, outIco = self.icoPair[idx]
        if self.params['process_name'] == 'train':
            return ico, outIco
        elif self.params['process_name'] == 'test':
            outIco = os.path.join(self.outIcoPth, os.path.basename(self.icoList[idx]).split('.')[0])
            return ico, outIco, ico
        else:
            raise ValueError('%s process on %s model with %s data_instance not defined, HOW DID YOU ENTER HERE??' % (
                self.params['process_name'], self.params['model_name'], self.params['data_instance_name']))

    def __len__(self):
        return len(self.icoList)

class createico2encDataset(torch.utils.data.Dataset):
    def __init__(self, params, data_instance):
        self.params = params
        self.icoList = listFiles(self.params, 'ico', data_instance)
        self.encDataPth = os.path.join(params['enc']['dataPth'], data_instance)
        if not os.path.exists(self.encDataPth):
            os.makedirs(self.encDataPth)
            print('created dir: %s' % self.encDataPth)
        print('Loaded %d ico pairs for data_instance: %s' %(len(self.icoList), data_instance))

    def __getitem__(self, idx):
        ico, _ = loadIcoFile(self.params, self.icoList[idx])
        enc = os.path.join(self.encDataPth, os.path.basename(self.icoList[idx]).split('.')[0]+self.params['enc']['ext'])
        return ico, enc

    def __len__(self):
        return len(self.icoList)

class createenc2icoDataset(torch.utils.data.Dataset):
    def __init__(self, params, data_instance):
        self.params = params
        encList = listFiles(self.params, 'enc', data_instance)
        icoList = listFiles(self.params, 'ico', data_instance)
        encBaseNameList = [os.path.basename(f) for f in encList]
        icoBaseNameList = [os.path.basename(f) for f in icoList]
        idList = []
        for id, ico in enumerate(icoBaseNameList):
            if ico in encBaseNameList:
                idList.append(id)
        self.encList = encList
        self.icoList = [icoList[id] for id in idList]
        self.outDataPth = os.path.join(params['out']['dataPth'], data_instance)
        if not os.path.exists(self.outDataPth):
            os.makedirs(self.outDataPth)
            print('created dir: %s' %self.outDataPth)

        print('Loaded %d enc files for data_instance: %s' %(len(self.encList), data_instance))

    def __getitem__(self, idx):
        enc = loadEncFile(self.params, self.encList[idx])
        icoPath = os.path.join(self.outDataPth, os.path.basename(self.encList[idx]).split('.')[0])
        ico, _ = loadIcoFile(self.params, self.icoList[idx])
        return enc, icoPath, ico

    def __len__(self):
        return len(self.icoList)

class createico2ico_vaeDataset(createico2icoDataset):
    def __init__(self, params, data_instance):
        super(createico2ico_vaeDataset, self).__init__(params, data_instance)

class createico2enc_vaeDataset(createico2encDataset):
    def __init__(self, params, data_instance):
        super(createico2enc_vaeDataset, self).__init__(params, data_instance)

class createenc2ico_vaeDataset(createenc2icoDataset):
    def __init__(self, params, data_instance):
        super(createenc2ico_vaeDataset, self).__init__(params, data_instance)
