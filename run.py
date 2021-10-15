# Title: GenIcoNet
# Author: Hardik Jain
# Date: 2020
# Link: https://github.com/hrdkjain/GenIcoNet

import argparse
import datetime
import glob
import multiprocessing
import natsort
import numpy as np
import os
import sys
import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm

import ico_utils
import data
import losses
import models
sys.path.append(os.path.join(os.path.dirname(__file__),'../PythonFunctions'))
import torch_utils
import python_utils
import torchsummary
sys.path.append(os.path.join(os.path.dirname(__file__),'../IcosahedralCNN'))
import icocnn

ENC = {}
MU = {}
LOGVAR = {}
REPARAMETERIZE = {}

def load_data(params, model_name, data_instance, batch_size=None):
    # Provides a data loader which contains pairs based on process_name
    # train: file-file
    # encode: file-fullFilePth
    # test: file-fullFilePth-reference(optional)
    if params['quickLearn']:
        print('Quick Learn Mode')

    dataset = eval('data.create'+params['model_name']+'Dataset(params, data_instance)')
    if params['quickLearn']:
        dataset = torch.utils.data.Subset(dataset, range(params['quickLearn']))

    shuffle = False
    if data_instance=='trn':
        shuffle = True
    if not batch_size:
        batch_size = params[model_name]['batch_size']
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=params['num_workers'], pin_memory=params['device'] == 'cuda')
    assert loader.__len__()!=0
    params[data_instance + '_dataset_len'] = dataset.__len__()
    params[data_instance + '_iter_per_epoch'] = loader.__len__()
    return loader

def load_both_data(params, model_name, train_size, batch_size=None):
    # outputs train and validation dataloader obtained from single dataset
    dataset = eval('data.create'+params['model_name']+'Dataset(params, train_size)')
    # dataset.performSanityChecks()
    params['classes'] = dataset.classes
    assert len(params['classes']) == params[params['model_name']]['num_classes']
    params['dataset_len'] = len(dataset)
    assert params['dataset_len'] != 0
    if not batch_size:
        batch_size = params[model_name]['batch_size']

    trn_dataset = torch.utils.data.Subset(dataset, dataset.trn_idx)
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=params['num_workers'], pin_memory=params['device'] == 'cuda')
    params['trn_iter_per_epoch'] = trn_loader.__len__()
    params['trn_dataset_len'] = len(dataset.trn_idx)

    val_dataset = torch.utils.data.Subset(dataset, dataset.val_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=params['num_workers'], pin_memory=params['device'] == 'cuda')
    params['val_iter_per_epoch'] = val_loader.__len__()
    params['val_dataset_len'] = len(dataset.val_idx)

    print('Split into %d trn and %d val files for %d classes' %(params['trn_dataset_len'], params['val_dataset_len'], len(params['classes'])))
    return trn_loader, val_loader

def load_log_data(params, val_loader, shuffle = False):
    num_log_samples = 3
    log_loader = {}
    if shuffle:
        data, targets = next(iter(val_loader))
        rand_indices = np.random.choice(len(data), num_log_samples, replace=False)
        log_loader['data_subset'] = data[rand_indices]
        log_loader['targets_subset'] = targets[rand_indices]
    else:
        log_dataset = torch.utils.data.Subset(val_loader.dataset, range(num_log_samples))
        loader = torch.utils.data.DataLoader(log_dataset, batch_size=num_log_samples, pin_memory=params['device'] == 'cuda')
        log_loader['data_subset'], log_loader['targets_subset'] = next(iter(loader))
    log_loader['max_dist_subset'] = 0.1
    return log_loader

def log_mesh(params, log_loader, model, writer, epoch, model_name, model2=None):
    if params[model_name]['log_mesh_epoch'] and epoch % params[model_name]['log_mesh_epoch'] == 0:
        # log meshes
        data = log_loader['data_subset']
        lbl = log_loader['targets_subset']

        data = data.to(params['device'], non_blocking=True)
        lbl = lbl.to(params['device'], non_blocking=True)
        if epoch:
            outputs = model(data)
            if 'vae' in model_name:
                outputs, _, _ = outputs
            if model2 is not None:
                outputs = model2(outputs)
            name = model_name
        else:
            name = model_name + '_ref'
            if 'ico2ico' in model_name:
                outputs = data  # ico2ico has ico loaded as input with 3 (vertices)
            else:
                raise ValueError('%s model not specified for log_mesh for epoch=0' %model_name)

        data_v = ico_utils.output2vertices(params['ico']['subdivisions'], outputs)
        data_colors = data_v.new_zeros(data_v.shape)

        if lbl.dim()==3:    #if lbl is icosahedron
            # prepare target labels
            tmp_lbl = lbl.transpose(1, 2).contiguous()
            lbl_v = tmp_lbl[:, :, :3]
            data_nn_v = lbl_v

            # create colors from distances
            dist = torch.sqrt(torch.sum((data_v - data_nn_v) ** 2, dim=2))
            dist[(dist > log_loader['max_dist_subset'])] = log_loader['max_dist_subset']
            avg_dist = dist.mean(1)
            dist /= log_loader['max_dist_subset']
            dist *= 255
            data_colors[:, :, 0] = dist
            if epoch:
                for i in range(data.shape[0]):
                    writer.add_scalars(model_name+'_mesh', {str(i): avg_dist[i]}, epoch)
        elif lbl.dim()==4:  # if lbl is encoding
            data_colors[:, :, 0] = 0
        else:
            raise ValueError('lbl shape: ' + str(lbl.shape) + ' not handable')
        # adapt shapes
        data_colors = data_colors.expand([-1, -1, 3])
        icofaces = torch.from_numpy(icocnn.utils.ico_geometry.get_ico_faces(params['ico']['subdivisions']))
        icofaces = icofaces.unsqueeze(0).expand([data_v.shape[0], -1, -1])

        writer.add_mesh(name, vertices=data_v, colors=data_colors, faces=icofaces, global_step=epoch)
        writer.flush()

def log_image(params, log_loader, model, writer, epoch, model_name, model2=None):
    # To log output/reconstructed images for celebA dataset
    if params[model_name]['log_image_epoch'] and epoch % params[model_name]['log_image_epoch'] == 0:
        # log images
        data = log_loader['data_subset']
        data = data.to(params['device'], non_blocking=True)
        if epoch:
            outputs = model(data)
            outputs = outputs[0]
            name = model_name
        else:
            name = model_name + '_ref'
            outputs = data
        outputs = torch_utils.tanh2sigmoid(outputs)
        writer.add_images(name, outputs, global_step=epoch)
        writer.flush()

def log_encoding(params, log_loader, model, writer, epoch, model_name):
    # log meshes
    data = log_loader['data_subset']
    lbl = log_loader['targets_subset']

    if epoch:
        data = data.to(params['device'], non_blocking=True)
        hooks = []
        # Based on the model, either the model output or the forward hook output is used
        if model_name == 'ico2ico' or model_name == 'img2ico':
            hooks.append(model.enc.register_forward_hook(hook))
        elif 'vae' in model_name:
            data = data[0:1]
            hooks.append(model.mu_hook.register_forward_hook(muHook))
            hooks.append(model.logvar_hook.register_forward_hook(logvarHook))
            hooks.append(model.reparameterize_hook.register_forward_hook(reparameterizeHook))
        outputs = model(data)
        if model_name == 'ico2ico':
            Outputs = ENC
            Model_name = model_name
        elif 'vae' in model_name:
            Outputs = MU, LOGVAR, REPARAMETERIZE
            Model_name = 'mu','logvar','reparam'
        if not hooks==[]:
            for h in hooks:
                h.remove()
    elif 'ico2ico' in model_name or model_name == 'img2ico':
        return
    else:
        lbl = lbl.to(params['device'], non_blocking=True)
        Outputs = lbl

    if 'log_encoding-hist' in params[params['model_name']] and params[params['model_name']]['log_encoding-hist']:
        for outputs,model_name in zip(Outputs,Model_name):
            writer.add_histogram(model_name, outputs, global_step=epoch,)
    else:
        #sample the outputs to contains only one out of the six rotations
        idx = range(0,Outputs.shape[1],int(Outputs.shape[1]/6))
        if len(idx) == 6:
            pass
        elif len(idx) > 6:
            idx = idx[:6]
        else:
            raise ValueError('idx should have 6 elements')
        Outputs = Outputs[:,idx,:,:]

        for i in range(Outputs.shape[0]):
            writer.add_images(model_name+'_'+str(i), Outputs[i].unsqueeze(3), global_step=epoch,  dataformats='NHWC')
    writer.flush()

def hook(module, input, output):
    global ENC
    ENC = output

def muHook(module, input, output):
    global MU
    MU = output

def logvarHook(module, input, output):
    global LOGVAR
    LOGVAR = output

def reparameterizeHook(module, input, output):
    global REPARAMETERIZE
    REPARAMETERIZE = output

def train(params, trn_loader, model, criterion, writer, epoch, model_name, optimizer, scheduler=None, cdata=None):
    misc = {}
    output = []
    model.train()
    with torch.autograd.detect_anomaly():
        if params['debug']:
            start = datetime.datetime.now()
        for i, (img, lbl) in enumerate(trn_loader):
            img = img.to(params['device'],non_blocking=True)
            lbl = lbl.to(params['device'],non_blocking=True)
            ## Forward
            output = model(img)
            loss = criterion(output, lbl)
            loss_posmse, loss_norcos, loss_lap, loss_vol, logLoss = criterion.get_last_losses()
            ## backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler step if there is an scheduler that needs update after each batch
            if scheduler:
                scheduler.step()

            iter = epoch * params['trn_iter_per_epoch'] + i
            if iter % params[model_name]['log_freq'] == 0:
                if params[model_name]['loss'] == 'vae' or params[model_name]['loss'] == 'p2pkld':
                    writer.add_scalars(model_name + '_recon', {'trn': loss_posmse}, iter)
                    writer.add_scalars(model_name + '_KLD', {'trn': loss_vol}, iter)
                writer.add_scalars(model_name, {'trn': logLoss}, iter)
                writer.flush()

            if params[model_name]['log_grad_freq']:
                if iter % params[model_name]['log_grad_freq'] == 0:
                    image = torch_utils.image_grad_flow(model.named_parameters(), epoch, i, cdata)
                    writer.add_image('Grad', image, iter, dataformats='NCHW')

            if params['debug']:
                end = datetime.datetime.now()
                print('iter %d: %f sec' %(i,(end-start).seconds))
                start = datetime.datetime.now()

    if params[model_name]['loss'] == 'vae' or params[model_name]['loss'] == 'p2pkld':
        misc['trn_mean'] = output[1]
        misc['trn_logvar'] = output[2]

    return misc

def validate(params, val_loader, model, criterion, writer, epoch, model_name, optimizer):
    model.eval()
    if params[model_name]['loss'] in params['vae_loss']:
        reconLoss = []
        otherLoss = []
    logValLoss = []
    with torch.no_grad():
        for img, lbl in val_loader:
            img = img.to(params['device'],non_blocking=True)
            lbl = lbl.to(params['device'],non_blocking=True)
            ## Forward
            output = model(img)
            loss = criterion(output, lbl)
            loss_posmse, loss_norcos, loss_lap, loss_vol, logLoss = criterion.get_last_losses()
            if params[model_name]['loss'] in params['vae_loss']:
                reconLoss.append(loss_posmse)
                otherLoss.append(loss_vol)
            logValLoss.append(logLoss)

    if params[model_name]['loss'] in params['vae_loss']:
        loss_recon = np.average(reconLoss)
        loss_other = np.average(otherLoss)
        if params[model_name]['loss'] == 'vae' or params[model_name]['loss'] == 'p2pkld':
            writer.add_scalars(model_name + '_recon', {'val': loss_recon}, epoch * params['trn_iter_per_epoch'])
            writer.add_scalars(model_name + '_KLD', {'val': loss_other}, epoch * params['trn_iter_per_epoch'])
        elif params[model_name]['loss'] == 'vqvae' or params[model_name]['loss'] == 'p2pvq':
            writer.add_scalars(model_name + '_recon', {'val': loss_recon}, epoch * params['trn_iter_per_epoch'])
            writer.add_scalars(model_name + '_VQ', {'val': loss_other}, epoch * params['trn_iter_per_epoch'])
    loss = np.average(logValLoss)
    writer.add_scalars(model_name, {'val': loss}, epoch * params['trn_iter_per_epoch'])
    writer.flush()
    if params[model_name]['loss'] == 'ce':
        tqdm.tqdm.write('Epoch: {epoch}, Val Loss: {loss:.6f}, Val Acc: {acc:.3f}'.format(epoch=epoch, loss=loss, acc=acc))
    else:
        tqdm.tqdm.write('Epoch: {epoch}, Val Loss: {loss:.6f}'.format(epoch=epoch, loss=loss))
    return loss

def saveBestModel(params, model, optimizer, epoch, modelName, last_best_loss, last_loss, misc=None):
    if last_loss[0] <= last_best_loss[0]:
        # delete old model
        lastBestModelPath = os.path.join(params['logDir'], 'savedModel', modelName + '_EB*[0-9]*.pt')
        lastBestModelPaths = glob.glob(lastBestModelPath)
        lastBestModelPaths = natsort.natsorted(lastBestModelPaths)
        for i in range(len(lastBestModelPaths)-5):  # keep last five+one best models
            os.remove(lastBestModelPaths[i])

        # save new last_loss
        saveModel(params, model, optimizer, 'B' + str(epoch), modelName, last_loss[0], misc)
        last_best_loss[0] = last_loss[0]

def saveModel(params, model, optimizer, epoch, modelName, val_loss, misc):
    modelPath = os.path.join(params['logDir'], 'savedModel', modelName + '_E' + str(epoch) + '.pt')
    if not os.path.exists(os.path.dirname(modelPath)):
        os.makedirs(os.path.dirname(modelPath))

    if not os.path.exists(modelPath):
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': ico_utils.getEpochNumber(epoch), 'loss': val_loss, 'misc': misc}, modelPath)
        print('Saved %s model with %s epochs (%s)' %(modelName,str(epoch), datetime.datetime.now()))
    else:
        print('%s model with %s epochs, already exists at %s, aborting saving !!' %(modelName,str(epoch),modelPath))

def loadModel(params, model, savedEpoch, modelName, optimizer=[], last_best_loss=[np.Inf], misc=[]):
    if savedEpoch == [0]:
        modelPath = os.path.join(params['logDir'], 'savedModel', modelName + '_EB*[0-9]*.pt')
        modelPaths = glob.glob(modelPath)
        modelPaths = natsort.natsorted(modelPaths)
        if len(modelPaths):
            # take the latest model
            modelPath = modelPaths[-1]
    else:
        modelPath = os.path.join(params['logDir'], 'savedModel', modelName + '_E' + str(savedEpoch[0]) + '.pt')

    if not os.path.exists(modelPath):
        print('No saved model exists at %s' %modelPath)
        return False

    checkpoint = torch.load(modelPath, map_location=lambda storage, loc: storage)

    # get model and saved model dictionary
    model_dict = model.state_dict()
    saved_dict = checkpoint['model_state_dict']
    saved_dict_len = len(saved_dict)
    # 1. filter out unnecessary keys
    saved_dict = {k: v for k, v in saved_dict.items() if k in model_dict}
    # 2. load the filtered saved dict
    model.load_state_dict(saved_dict)
    print('Selected %d dict keys out of %d keys' %(len(saved_dict),saved_dict_len))

    if not optimizer==[]:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    savedEpoch[0] = checkpoint['epoch']
    if 'loss' in checkpoint:
        last_best_loss[0] = checkpoint['loss']

    misc.append(checkpoint['misc'])
    print('Loaded %s model with %d epochs' %(modelName,savedEpoch[0]))
    # re-set the dataPth, if find best model as set
    if 'dataPth' in params['out']: params['out']['dataPth'] = params['out']['dataPth'].replace('E0', 'EB'+str(savedEpoch[0]))
    if 'dataPth' in params['enc']: params['enc']['dataPth'] = params['enc']['dataPth'].replace('E0', 'EB'+str(savedEpoch[0]))
    if 'dataPth' in params['ftr']: params['ftr']['dataPth'] = params['ftr']['dataPth'].replace('E0', 'EB'+str(savedEpoch[0]))
    return True

def loadMultiModel(params, model, savedEpochs, modelNames):
    # this is a special case which requires loading two models
    modelPath = {}
    saved_dict = {}
    new_saved_dict = {}
    common_saved_dict = {}
    # get model and saved model dictionary
    model_dict = model.state_dict()
    print('Model dict len %d' %len(model_dict))
    for id, (savedEpoch, modelName) in enumerate(zip(savedEpochs, modelNames)):
        new_saved_dict[modelName] = {}
        modelPath[modelName] = os.path.join(params['logDir'], 'savedModel', modelName + '_E' + str(savedEpoch) + '.pt')
        if not os.path.exists(modelPath[modelName]):
            raise ValueError('No saved model exists at %s' %modelPath[modelName])
        checkpoint = torch.load(modelPath[modelName], map_location=lambda storage, loc: storage)
        saved_dict[modelName] = checkpoint['model_state_dict']
        # 1. filter out unnecessary keys
        for k, v in saved_dict[modelName].items():
            if k in model_dict:
                new_saved_dict[modelName][k] = v
                del model_dict[k]
        common_saved_dict.update(new_saved_dict[modelName])
        print('Selected %d dict keys out of %d keys from %s' % (len(new_saved_dict[modelName]), len(saved_dict[modelName]), modelPath[modelName]))

    # 2. load the filtered saved dict
    model.load_state_dict(common_saved_dict)
    return True


def experiment_train(params):
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=params['logDir'])
    model = eval('models.' + params['model_name'] + '(params).to(params[\'device\'],non_blocking=True)')

    if params['quickLearn']:
        # To speed up quick checks
        trn_loader = load_data(params, params['model_name'],'val')
        params['trn_iter_per_epoch'] = params['val_iter_per_epoch']
        params['trn_dataset_len'] = params['val_dataset_len']
    else:
        trn_loader = load_data(params, params['model_name'],'trn')
    val_loader = load_data(params, params['model_name'],'val')


    log_loader = load_log_data(params, val_loader)
    summ = torchsummary.summary_string(model, input_size=ico_utils.get_input_shape(trn_loader.dataset), device=params['device'])
    imagePath = python_utils.get_new_name(os.path.join(params['logDir'], 'train' + '_' + params['model_name']), '.jpg')
    torchsummary.save_summary(summ, imagePath)
    torchsummary.draw_graph(model, ico_utils.get_input_shape(trn_loader.dataset), params['device'], imagePath.replace('.jpg',''))

    if params[params['model_name']]['loss'] == 'p2p':
        # pos: vertex positions
        # nor: vertex normals, precomputed reference used
        # lap: laplacian batch, precomputed reference used
        criterion = losses.P2P_Loss(params['ico']['subdivisions'], params['ico']['factor_pos'],
                                    params['ico']['factor_nor'], params['ico']['factor_lap'])
    elif params[params['model_name']]['loss'] == 'p2pkld':
        factor_KL = 1.
        criterion = losses.P2PKLD_Loss(params['ico']['subdivisions'], params['ico']['factor_pos'],
                                       params['ico']['factor_nor'], params['ico']['factor_lap'], factor_KL)
    else:
        raise ValueError('loss for %s model not specified' %params['model_name'])

    criterion = criterion.to(params['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params[params['model_name']]['lr'])
    print('Optimizable Parameters %d' %sum(p.numel() for p in model.parameters() if p.requires_grad == True))
    if 'lr_base' in params[params['model_name']] and 'lr_max' in params[params['model_name']]:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, params[params['model_name']]['lr_base'], params[params['model_name']]['lr_max'],
                                                      cycle_momentum=False)
        print('Using CyclicLR scheduler with lr_base: %f, lr: %f, lr_max: %f' %(params[params['model_name']]['lr_base'],
                                                                                params[params['model_name']]['lr'],params[params['model_name']]['lr_max']))
    else:
        scheduler = None

    savedEpoch = [0]
    model_loaded = False
    last_best_loss = [np.Inf]
    if params[params['model_name']]['load_pretrained_model']:
        savedEpoch = [params[params['model_name']]['load_epoch']]
        print('Asked to load pre-trained model')
        model_loaded = loadModel(params, model, savedEpoch, params['model_name'], optimizer, last_best_loss)
    if not params['quickLearn']:
        if not model_loaded:
            ico_utils.get_input_shape(trn_loader.dataset)
            input = torch.rand((1,) +  ico_utils.get_input_shape(trn_loader.dataset))
            input = input.to(params['device'],non_blocking=True)
            try:
                writer.add_graph(model, input)
                writer.flush()
            except:
                print('Couldn\'t print model to tensorboard, probably because model.__init__() has an additional submodel as class')


    last_loss = []
    writer.add_text(params['model_name'] + '_trn_iter_per_epoch', str(params['trn_iter_per_epoch']), savedEpoch[0])
    writer.add_text(params['model_name'] + '_val_iter_per_epoch', str(params['val_iter_per_epoch']), savedEpoch[0])
    print(datetime.datetime.now())
    misc = None
    for epoch in tqdm.trange(savedEpoch[0],params[params['model_name']]['train_epoch']):
        if params['device'] == 'cuda':
            log_mesh(params, log_loader, model, writer, epoch, params['model_name'])
        if 'log_encoding_epoch' in params[params['model_name']] and params[params['model_name']]['log_encoding_epoch'] and epoch % params[params['model_name']]['log_encoding_epoch'] == 0:
            log_encoding(params, log_loader, model, writer, epoch, params['model_name'])
        misc = train(params, trn_loader, model, criterion, writer, epoch, params['model_name'], optimizer, scheduler, summ[2])
        last_loss = [validate(params, val_loader, model, criterion, writer, epoch+1, params['model_name'], optimizer)]
        saveBestModel(params, model, optimizer, epoch+1, params['model_name'], last_best_loss, last_loss, misc)
        if (epoch+1) % params[params['model_name']]['save_epoch_freq'] == 0:
            saveModel(params, model, optimizer, epoch+1, params['model_name'], last_loss, misc)

        if 'factor_step_size' in params[params['model_name']] and 'factor_gamma' in params[params['model_name']]:
            writer.add_scalar(params['model_name'] + '_factor', criterion.get_factor(), epoch * params['trn_iter_per_epoch'])
            criterion.update_factor(epoch+1, params[params['model_name']]['factor_step_size'], params[params['model_name']]['factor_gamma'])

    if params[params['model_name']]['train_epoch']-savedEpoch[0]:
        saveModel(params, model, optimizer, params[params['model_name']]['train_epoch'], params['model_name'], last_loss, misc)
    print(datetime.datetime.now())

def experiment_test(params):
    model = eval('models.' + params['model_name'] + '(params).to(params[\'device\'],non_blocking=True)')
    if 'enc2ico' in params['model_name']:
        load_model_name = params['model_name'].replace('enc2ico','ico2ico')
    else:
        load_model_name = params['model_name']

    if not loadModel(params, model, [params[params['model_name']]['test_epoch']], load_model_name):
        raise ValueError('Unable to load model')

    model2 = None
    batch_size = params[params['model_name']]['batch_size'] if 'batch_size' in params[params['model_name']] else 1
    test_loader = load_data(params, params['model_name'], params[params['model_name']]['data_instance'], batch_size)
    summ = torchsummary.summary_string(model, input_size=ico_utils.get_input_shape(test_loader.dataset), device=params['device'])
    torchsummary.save_summary(summ, python_utils.get_new_name(os.path.join(params['logDir'], 'test' + '_' + params['model_name']), '.jpg'))

    model = model.to(params['device'],non_blocking=True)
    #model.eval()
    nameDistPair = []
    with torch.no_grad():
        for ip, op, ref in tqdm.tqdm(test_loader):
            ip = ip.to(params['device'],non_blocking=True)
            ref = ref.to(params['device'],non_blocking=True)
            output = model(ip)
            if 'vae' in params['model_name']:
                output, _, _ = output
            if model2 is not None:
                output = model2(output)
            outvertices = ico_utils.output2vertices(params['ico']['subdivisions'], output)
            refvertices = ico_utils.output2vertices(params['ico']['subdivisions'], ref)
            icofaces = torch.from_numpy(icocnn.utils.ico_geometry.get_ico_faces(params['ico']['subdivisions']))
            icofaces = icofaces.unsqueeze(0).expand([outvertices.shape[0], -1, -1]).to(params['device'],non_blocking=True)
            for id, f in enumerate(op):
                dist = ico_utils.computeDistance(outvertices[id], refvertices[id], icofaces[id], f, params[params['model_name']]['test_mode'], params[params['model_name']]['write_output_mesh'])
                if dist is not None:
                    nameDistPair.append([os.path.splitext(os.path.basename(f))[0], dist])
    if params[params['model_name']]['test_mode'] is not None:
        ico_utils.saveDistance(nameDistPair, os.path.join(os.path.dirname(f) + '_' + params[params['model_name']]['test_mode']))

def get_args(params):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="Architecture of the GenIcoNet i.e. AE: ico2ico or VAE: ico2ico_vae",
                        type=str, default='ico2ico', required=False)
    parser.add_argument("--process", help="Whether to train or test the network",
                        type=str, default='train', required=False)
    parser.add_argument("--data_instance", help="data instance to use for the process either trn or val",
                        type=str, default='val', required=False)
    parser.add_argument("--batch_size", help="batch size for training",
                        type=int, default=12, required=False)
    parser.add_argument("--quickLearn", help="data subset size to use quick learning",
                        type=int, default=0, required=False)
    parser.add_argument("--debug", help="debugging",
                        action='store_true', required=False)
    parser.add_argument("--logDir", help="folder to store the train experiment output and to read test experiment models",
                        type=str, default='log/test1', required=False)
    parser.add_argument("--dataPth", help="folder containing dataset in .mat",
                        type=str, default='/home/user/Dataset/ModelNet10/V128A_AHSO_I5', required=False)
    parser.add_argument("--subdivision", help="Icosahedral subdvision level of the data",
                        type=int, default=5, required=False)
    parser.add_argument("--suffix", help="suffix of the ico files",
                        type=str, default='ahs_I5', required=False)
    parser.add_argument("--train_epoch", help="No. of Epoch to train",
                        type=int, default='0', required=False)
    parser.add_argument("--test_epoch", help="Epoch to use for testing",
                        type=str, default='0', required=False)
    parser.add_argument("--test_mode", help="Metric to compute during testing point2mesh/None",
                        type=str, default=None, required=False)
    parser.add_argument("--write_output_mesh", help="Write the output mesh when testing",
                        action='store_true', required=False)

    args = parser.parse_args()
    for arg in vars(args):
        if arg in ['model', 'process']:
            params[arg][args.__dict__[arg]] = True
        elif arg in ['data_instance', 'batch_size', 'train_epoch', 'test_mode', 'write_output_mesh']:
            params[args.model][arg] = args.__dict__[arg]
        elif arg in ['test_epoch']:
            try:
                params[args.model]['test_epoch'] = int(args.__dict__[arg])
            except ValueError:
                params[args.model]['test_epoch'] = args.__dict__[arg]
        elif arg in ['dataPth', 'subdivision', 'suffix']:
            params['ico'][arg] = args.__dict__[arg]
        else:
            params[arg] = args.__dict__[arg]
    params['model_name'] = args.model
    params['process_name'] = args.process
    params['data_instance_name'] = args.data_instance

def set_paths(params):
    # Derived Paths
    if 'ico2ico' in params['model_name']:
        out_model_name = params['model_name']
        out_epoch = params[params['model_name']]['test_epoch']

    if 'enc_model_name' not in locals():
        enc_model_name= 'NA'
    if 'enc_epoch' not in locals():
        enc_epoch = np.inf
    if 'ftr_model_name' not in locals():
        ftr_model_name = 'NA'
    if 'out_model_name' not in locals():
        out_model_name = 'NA'
    if 'out_epoch' not in locals():
        out_epoch = 'np.inf'

    params['enc']['dataPth'] =  os.path.join(params['enc']['intrPth'],
                                             enc_model_name,
                                             'E' + str(enc_epoch))
    params['ftr']['dataPth'] =  os.path.join(params['ftr']['intrPth'],
                                             ftr_model_name)
    params['out']['dataPth'] = os.path.join(params['out']['intrPth'],
                                            out_model_name,
                                            'E' + str(out_epoch))

if __name__ == '__main__':
    params = {}
    params['model'] = {'ico2ico': False, 'ico2ico_vae': False }
    params['process'] = {'train': False, 'test': False}

    # train_epoch: #epochs to train the model
    # save_epoch_freq: model saving frequency
    # test_epoch: epoch at which to test the model
    # log_mesh_epoch: mesh logging frequency
    # load_pretrained_model: to load pre-trained model
    # load_epoch: epoch to load model for inference or re-train
    # load_enc_epoch: #epochs to load trained submodel for either computing encoding or using to compute output from encoding
    # log_freq: frequency at which to log the scalars
    # model: 'simple' 'residual' 'simpleS2S' 'residualS2S' 'identity'
    # loss: p2p, p2pFnLb, CustomChamfer, mse, p2pVol, vae, p2pkld, vqvae, p2pvq, ce
    # test_mode: point2point, point2mesh, chamfer, f_<radius>
    # M1 Specifications
    params['ico2ico'] = {'model': 'residualS2S', #'simple'
                         'loss': 'p2p',
                         'lr': 1e-6,
                         'lr_base': 1e-9,
                         'lr_max': 1e-3,
                         'batch_size': 12,
                         'train_epoch': 600,
                         'save_epoch_freq': 100,
                         'log_freq': 10,
                         'log_mesh_epoch': 50,
                         'log_grad_freq': 1000,
                         'log_encoding_epoch': 0,
                         'data_instance': 'trn',
                         'load_pretrained_model': False,
                         'load_epoch': 0,
                         'test_epoch': 0, # to take the last best or else string: 'B44'
                         'test_mode': 'point2mesh',
                         }

    params['ico2ico_vae'] = {'model': 'residualS2S',
                             'loss': 'p2pkld',
                             'factor_step_size': 25,
                             'factor_gamma': 0.9,
                             'lr': 1e-6,
                             'lr_base': 1e-9,
                             'lr_max': 1e-3,
                             'batch_size': 12,
                             'train_epoch': 600,
                             'save_epoch_freq': 50,
                             'log_freq': 20,
                             'log_mesh_epoch': 25,
                             'log_grad_freq': 1000,
                             'log_encoding_epoch': 50,
                             'log_encoding-hist': True,
                             'data_instance': 'trn',
                             'load_pretrained_model': False,
                             'load_epoch': 0,
                             'test_epoch': 0, # to take the last best or else string: 'B44'
                             'test_mode': 'point2mesh',
                             }

    # 4 Data types Paths (ico, enc, img, out)
    # filePth: path to the file which contains the list of files
    # dataPth: path to the data folder which contains the files
    # intrPth: any other intermediate path used to derive the dataPth

    # ico
    params['ico'] = {
        'ext': '.npz',
        'subdivisions': 5,
        'width': None,
        'corner_mode': 'average',
        'dataPthLvl': 2  # for ModelNet10 dataset
    }
    # Update params from arg input
    get_args(params)

    if params['model_name'] == 'ico2ico':
        params['ico']['factor_pos'] = 1.
        params['ico']['factor_nor'] = 0.
        params['ico']['factor_lap'] = 0.
    elif params['model_name'] == 'ico2ico_vae':
        params['ico']['factor_pos'] = 0.6
        params['ico']['factor_nor'] = 0.2
        params['ico']['factor_lap'] = 0.2
    params['ico']['width'] = 2 ** (params['ico']['subdivisions'] + 1)
    params['ico']['suffix'] = 'ahs_I'+str(params['ico']['subdivisions'])

    params['enc'] = {'intrPth': os.path.join(params['logDir'],'data'),
                     'suffix': 'ahs_I'+str(params['ico']['subdivisions']),
                     'ext': '.npz'}
    params['ftr'] = {'intrPth': os.path.join(params['logDir'],'data'),
                     'ext': '.npz',
                    }
    params['out'] = {'intrPth': os.path.join(params['logDir'],'data'), #,'Output'
                     }
    # Variational Autoencoders
    params['vae_loss'] = ['p2pkld']
    set_paths(params)

    # Device and Repo
    params['device'] = torch_utils.selectDevice(torch.cuda)
    params['num_workers'] = multiprocessing.cpu_count()*2
    params['GenIcoNetGitID'] = torch_utils.get_git_info('.')
    params['IcosahedronCNNGitID'] = torch_utils.get_git_info('../IcosahedralCNN')

    print('****************************************************************************************')
    print('Using %s for %s process on %s model with %s data_instance, logging at %s'
          %(params['device'], params['process_name'], params['model_name'],params['data_instance_name'],params['logDir']))

    if params['process']['train']:
        torch_utils.save_params(params, params['logDir'])
        experiment_train(params)
    elif params['process']['test']:
        experiment_test(params)
