import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_table.Format
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import scipy.spatial
import sys
import torch
import torch.utils.data as data_utils
import tqdm
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from functools import partial
from natsort import natsorted
from plyfile import PlyData
from sklearn.decomposition import PCA

import run
sys.path.append(os.path.join(os.path.dirname(__file__), '../PythonFunctions'))
import python_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'Visualizer'))
import visualizer_utils
import dash_reusable_components as drc
sys.path.append(os.path.join(os.path.dirname(__file__), '../IcosahedralCNN/'))
from data import createico2icoDataset, createico2ico_vaeDataset
from icocnn.utils.ico_geometry import get_ico_faces


# 1. Initialize dash
app = dash.Dash(
    __name__,
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
    ],
    assets_folder='Visualizer/assets'
)
server = app.server


# List saved model epochs
@app.callback(
    [
        Output('dropdown-test-epoch','options'),
        Output('dropdown-test-epoch','value')
    ],
    [
        Input('dropdown-select-model', 'value')
    ]
)
def update_epoch_list(model_name):
    if not os.path.exists(os.path.join(params['logDir'], 'savedModel')):
        raise PreventUpdate
    models = natsorted([f.split('.')[0].replace(model_name+'_E','') for f in os.listdir(os.path.join(params['logDir'],'savedModel'))])
    options = [{'label': f, 'value': f} for f in models if '_' not in f]
    if len(options) > 0:
        defaultOption = options[-1]['value']
    else:
        defaultOption = None
    return options, defaultOption

# Callback: Based on the dataset, model and data_instance,
# load model, get Dataset, list files and print status
@app.callback(
    [
        Output('logging-state', 'children'),
        Output('dropdown-select-file11', 'options'),
        Output('dropdown-select-file12', 'options'),
        Output('dropdown-select-file13', 'options'),
        Output('dropdown-select-file14', 'options'),
        Output('dropdown-select-file31', 'options'),
        Output('dropdown-select-file41', 'options'),
        Output('dropdown-select-file51', 'options'),
        Output('dropdown-select-file52', 'options'),
        Output('dropdown-select-file53', 'options'),
        Output('dropdown-select-file61', 'options'),
        Output('dropdown-select-file62', 'options'),
        Output('dropdown-select-file71', 'options'),
        Output('dropdown-select-file72', 'options'),
        Output('dropdown-select-file73', 'options'),
        Output('dropdown-select-file74', 'options'),
        Output('viewpoint-input','value')
    ],
    [
        Input('button-listFiles','n_clicks')
    ],
    [
        State('dropdown-select-model', 'value'),
        State('dropdown-select-data_instance', 'value'),
        State('dropdown-test-epoch', 'value'),
    ]
)
def update_model_dataset_files(n_clicks, model_name, data_instance, test_epoch):
    global pca
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate

    global ENCODER_MODEL, DATASET
    params['model_name'] = model_name
    params['dataDir'] = os.path.join(params['logDir'],'data', model_name, 'E'+test_epoch, 'MeshVisualizer')
    if not os.path.exists(params['dataDir']):
        os.makedirs(params['dataDir'])
    string = loadModelnDataset(model_name, data_instance, test_epoch)
    string += 'Listed ' + str(len(DATASET.fileList)) +' files, '
    pca = None
    return string, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, \
           DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, DATASET.fileList, \
           params['dataDir']

def update_input_graph(file_id, id):
    if file_id is None:
        raise PreventUpdate

    v_input = get_vinput_from_idx(DATASET, file_id)
    input_figure = visualizer_utils.create_mesh_figure(v_input, params['ico']['faces'], 'Input{} ({})'.format(id,DATASET.fList[file_id]), 350, 350, False)
    return input_figure
for i in [11,12,13,14,31,51,52,53]:
    app.callback(
        Output('input%i-figure' %i, 'data'),
        [
            Input('dropdown-select-file%i' %i, 'value')
        ]
    )(partial(update_input_graph, id=i))

for i in [11,12,13,14,31,51,52,53]:
    @app.callback(
        [
            Output('input%i-dialog' %i, 'message'),
            Output('input%i-dialog' %i, 'displayed'),
        ],
        [
            Input('button-input%i' %i, 'n_clicks')
        ],
        [
            State('dropdown-select-file%i' %i, 'value'),
            State('input%i-figure' %i, 'data'),
            State('radio-input%i' %i, 'value')
        ]
    )
    def save_input_graph(n_clicks, file_id, input_figure, ext):
        if n_clicks is None or n_clicks <= 0 or file_id is None:
            raise PreventUpdate
        fileName = 'GT_{}'.format(DATASET.fList[file_id])
        return visualizer_utils.save_figure(params['dataDir'], fileName, input_figure, ext)

def update_output_graph(file_id, id):
    if file_id is None:
        raise PreventUpdate
    encoding, _ = run_encoder(file_id)
    if 'vae' in ENCODER_MODEL._get_name():
        encoding, _ = encoding
    v_output = run_decoder(encoding)
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Reconstructed Output{} ({})'.format(id,DATASET.fList[file_id]), 350, 350, False)
    return output_figure, encoding.numpy()
for i in [61,62]:
    app.callback(
        [
            Output('output%i-figure' %i, 'data'),
            Output('output%i-mu' % i, 'data'),
        ],
        [
            Input('dropdown-select-file%i' %i, 'value')
        ]
    )(partial(update_output_graph, id=i))

for i in [61,62]:
    @app.callback(
        [
            Output('output%i-dialog' %i, 'message'),
            Output('output%i-dialog' %i, 'displayed'),
        ],
        [
            Input('button-output%i' %i, 'n_clicks')
        ],
        [
            State('dropdown-select-file%i' %i, 'value'),
            State('output%i-figure' %i, 'data'),
            State('radio-output%i' %i, 'value')
        ]
    )
    def save_output6_graph(n_clicks, file_id, output_figure, ext):
        if n_clicks is None or n_clicks <= 0 or file_id is None:
            raise PreventUpdate
        fileName = 'OP_{}'.format(DATASET.fList[file_id])
        return visualizer_utils.save_figure(params['dataDir'], fileName, output_figure, ext)

for i in [71,72,73,74]:
    @app.callback(
        [
            Output('output%i-dialog' %i, 'message'),
            Output('output%i-dialog' %i, 'displayed'),
        ],
        [
            Input('button-output%i' %i, 'n_clicks')
        ],
        [
            State('dropdown-select-file%i' %i, 'value'),
            State('output%i-figure' %i, 'data'),
            State('radio-output%i' %i, 'value')
        ]
    )
    def save_output7_graph(n_clicks, file_id, output_figure, ext):
        if n_clicks is None or n_clicks <= 0 or file_id is None:
            raise PreventUpdate
        fileName = 'PA_{}'.format(DATASET.fList[file_id])
        return visualizer_utils.save_figure(params['dataDir'], fileName, output_figure, ext)
# Slider: to select the interpolation
@app.callback(
    Output('text-interpolation', 'value'),
    [
        Input('slider-interpolate', 'value')
    ],
    [
        State('dropdown-select-file11', 'value'),
        State('dropdown-select-file12', 'value'),
    ]
)
def update_output1_text(value, file11_id, file12_id):
    if file11_id is None or file12_id is None:
        raise PreventUpdate
    return '{:0.1f}*{}+{:0.1f}*{}'.format(1-value[0],DATASET.fList[file11_id], value[0], DATASET.fList[file12_id])

@app.callback(
    [
        Output('output21-figure', 'data'),
        Output('output22-figure', 'data'),
        Output('topk-table', 'data'),
        Output('input24-figure', 'data'),
        Output('output25-figure', 'data'),
    ],
    [
        Input('slider-interpolate', 'value'),
        Input('radio-operation', 'value'),
        Input('dropdown-select-file11', 'value'),
        Input('dropdown-select-file12', 'value'),
        Input('dropdown-select-file13', 'value'),
        Input('dropdown-select-file14', 'value'),
    ],
    [
        State('dropdown-select-model', 'value'),
        State('output21-figure', 'data'),
        State('output22-figure', 'data'),
        State('topk-table', 'data')
    ],
)
def compute_output2_graph(interpolate, operation, file11_id, file12_id, file13_id, file14_id, model_name, output_figure, pca_figure, pca_dict):
    global pca, PCA_DATASET, enc_pca_tree
    trigger = []
    v_interpolate = None
    for item in dash.callback_context.triggered:
        trigger.append(item['prop_id'])
    if operation == 'pca' and pca is None:
        pca_file = os.path.join(params['dataDir'],'pca.pkl')
        if os.path.exists(pca_file):
            with open(pca_file, 'rb') as f:
                print('Reading pca from file:' + pca_file)
                pca, PCA_DATASET, enc_pca, enc_pca_tree = pickle.load(f)
        else:
            enc1 = run_model_on_dataset(ENCODER_MODEL, DATASET)
            OTHER_DATASET, PCA_DATASET = loadPCADataset(model_name, DATASET)
            enc2 = run_model_on_dataset(ENCODER_MODEL, OTHER_DATASET)
            enc = np.concatenate((enc1,enc2), axis=0)
            pca = PCA(n_components=3)
            ### fit PCA on the complete dataset
            pca.fit(enc.reshape(enc.shape[0], -1))
            enc_pca = pca.transform(enc.reshape(enc.shape[0], -1))
            enc_pca_tree = scipy.spatial.KDTree(enc_pca)
            with open(pca_file, 'wb') as f:
                pickle.dump([pca, PCA_DATASET, enc_pca, enc_pca_tree], f)
                print('Writing pca to file:' + pca_file)

        pca_figure = visualizer_utils.create_3d_figure(PCA_DATASET, enc_pca, 'PCA Output22', 450, 450)
        if pca_dict is None:
            pca_dict = {}
        for idx, f in enumerate(PCA_DATASET.fList):
            pca_dict[f] = enc_pca[idx]
    elif operation == 'interpolate':
        if 'dropdown-select-file11.value' in trigger or 'dropdown-select-file12.value' in trigger or \
            'slider-interpolate.value' in trigger:
            if file11_id is not None and file12_id is not None:
                interpolate = interpolate[0]
                # run encoder
                v_enc, v_enc1 = run_encoder(file11_id, file12_id)
                if 'vae' in ENCODER_MODEL._get_name():
                    v_enc, _ = v_enc
                    v_enc1, _ = v_enc1
                # manipulate encoding based
                v_interpolate = explore_enc(v_enc, v_enc1, interpolate)
                x_interpolate = v_interpolate.cpu().numpy().reshape(v_interpolate.shape[0], -1)
                if pca_figure:
                    # apply pca on v_enc
                    x_interpolate_pca = pca.transform(x_interpolate)
                    [dist, nn_id] = enc_pca_tree.query(x_interpolate_pca)
                    nn_id = nn_id[0]
                    v_input = get_vinput_from_idx(PCA_DATASET, nn_id)
                    input24_figure = visualizer_utils.create_mesh_figure(v_input, params['ico']['faces'],
                                                        'NN Input24 ({})-[{:.4f}]'.format(PCA_DATASET.fList[nn_id],
                                                                                          dist[0]), 350, 350, False)
                    v_NN_input, _ = PCA_DATASET[nn_id]
                    v_NN_encoding = run_model(ENCODER_MODEL, v_NN_input)
                    if 'vae' in ENCODER_MODEL._get_name():
                        v_NN_encoding, _ = run_model(ENCODER_MODEL, v_NN_input)
                    v_NN_output = run_decoder(v_NN_encoding)
                    output25_figure = visualizer_utils.create_mesh_figure(v_NN_output, params['ico']['faces'], 'NN Output25', 350, 350, False)
                # run decoder
                v_output = run_decoder(v_interpolate)
                output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Interpolation Output21', 350, 350, False)
    if pca_figure:
        points = []
        hoverText = []
        for id in [file11_id, file12_id, file13_id, file14_id]:
            if id is not None:
                points.append([pca_figure['data'][0]['x'][id], pca_figure['data'][0]['y'][id],
                               pca_figure['data'][0]['z'][id]])
                hoverText.append(pca_figure['data'][0]['text'][id])
            else:
                points.append([None, None, None])
                hoverText.append(None)
        trace1 = pca_figure['data'][1]
        trace2 = pca_figure['data'][2]
        trace3 = pca_figure['data'][3]
        trace4 = pca_figure['data'][4]
        # update trace1
        if 'dropdown-select-file11.value' in trigger or 'dropdown-select-file12.value' in trigger or \
            'dropdown-select-file13.value' in trigger or 'dropdown-select-file14.value' in trigger:
            trace1 = create_points_trace(points, hoverText)
        # update trace2
        if 'dropdown-select-file11.value' in trigger or 'dropdown-select-file12.value' in trigger:
            if file11_id is not None and file12_id is not None:
                trace2 = create_line_trace(points)
        # update trace3 and trace4
        if "x_interpolate_pca" in locals():
            trace3 = create_point_trace(x_interpolate_pca[0], 'interpolate', 'diamond')
            point = [pca_figure['data'][0]['x'][nn_id], pca_figure['data'][0]['y'][nn_id],
                               pca_figure['data'][0]['z'][nn_id]]
            trace4 = create_point_trace(point, 'NN', 'cross')
        # update pca_figure
        pca_figure = go.Figure(data=[pca_figure['data'][0],
                                 trace1,
                                 trace2,
                                 trace3,
                                 trace4], layout=pca_figure['layout'])

    if "input24_figure" not in locals():
        input24_figure = None
    if "output25_figure" not in locals():
        output25_figure = None
    return output_figure, pca_figure, pca_dict, input24_figure, output25_figure

@app.callback(
    [
        Output('output21-dialog', 'message'),
        Output('output21-dialog', 'displayed'),
    ],
    [
        Input('button-output21', 'n_clicks')
    ],
    [
        State('text-interpolation', 'value'),
        State('output21-figure', 'data'),
        State('radio-output21', 'value')
    ]
)
def save_output21_graph(n_clicks, text, output21_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output21_figure is None or text is None:
        raise PreventUpdate
    fileName = 'IT_{}'.format(text.replace('*','x'))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output21_figure, ext)

# compute top-k pairs
@app.callback(
    Output('table-topk', 'data'),
    [
        Input('button-topk','n_clicks')
    ],
    [
        State('topk-table', 'data')
    ]
)
def compute_topk_table(n_clicks, pca_dict):
    global pca, PCA_DATASET
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate
    topk = 5
    data = []
    distance={}
    if pca:
        # compute distance between each pair
        for idx1 in tqdm.tqdm(range(len(PCA_DATASET.fList))):
            for idx2 in range(idx1, len(PCA_DATASET.fList)):
                f1 = PCA_DATASET.fList[idx1]
                f2 = PCA_DATASET.fList[idx2]
                if not f1 == f2:
                    # print(pca_dict[f1])
                    distance[f1+'-'+f2] = scipy.spatial.distance.euclidean(pca_dict[f1],pca_dict[f2])
        #distance = {k: v for k, v in sorted(distance.items(), key=lambda item: item[1])}
        distance = sorted(distance.items(), key=lambda item: item[1])

        # top-k closest
        for k in range(topk):
            data.append({'column1': distance[k][0], 'column2': distance[k][1]})
        data.append({'column1': '...', 'column2': '...'})
        # top-k farthest
        for k in range(topk,0,-1):
            data.append({'column1': distance[-k][0], 'column2': distance[-k][1]})
    return data

@app.callback(
    [
        Output('output22-dialog', 'message'),
        Output('output22-dialog', 'displayed'),
    ],
    [
        Input('button-output22', 'n_clicks')
    ],
    [
        State('output22-figure', 'data'),
        State('radio-output22', 'value')
    ]
)
def save_output22_graph(n_clicks, output22_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output22_figure is None or ext == '.off':
        raise PreventUpdate
    fileName = 'pca'
    return visualizer_utils.save_figure(params['dataDir'], fileName, output22_figure, ext)

@app.callback(
    [
        Output('input24-dialog', 'message'),
        Output('input24-dialog', 'displayed'),
    ],
    [
        Input('button-input24', 'n_clicks')
    ],
    [
        State('text-interpolation', 'value'),
        State('input24-figure', 'data'),
        State('radio-input24', 'value')
    ]
)
def save_input24_graph(n_clicks, text, input24_figure, ext):
    if n_clicks is None or n_clicks <= 0 or input24_figure is None or text is None:
        raise PreventUpdate
    fileName = 'INN_{}'.format(text.replace('*','x'))
    return visualizer_utils.save_figure(params['dataDir'], fileName, input24_figure, ext)


@app.callback(
    [
        Output('output25-dialog', 'message'),
        Output('output25-dialog', 'displayed'),
    ],
    [
        Input('button-output25', 'n_clicks')
    ],
    [
        State('text-interpolation', 'value'),
        State('output25-figure', 'data'),
        State('radio-output25', 'value')
    ]
)
def save_input25_graph(n_clicks, text, output25_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output25_figure is None or text is None:
        raise PreventUpdate
    fileName = 'INNR_{}'.format(text.replace('*', 'x'))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output25_figure, ext)


# Callback: to update camera-mesh-plot containing camera
@app.callback(
    Output('camera-mesh-plot', 'data'),
    [
        Input('graph-input11', 'relayoutData'),
        Input('graph-input12', 'relayoutData'),
        Input('graph-input13', 'relayoutData'),
        Input('graph-input14', 'relayoutData'),
        Input('graph-output21', 'relayoutData'),
        Input('graph-input24', 'relayoutData'),
        Input('graph-output25', 'relayoutData'),
        Input('graph-input31', 'relayoutData'),
        Input('graph-output32', 'relayoutData'),
        Input('graph-output33', 'relayoutData'),
        Input('graph-output41', 'relayoutData'),
        Input('graph-output42', 'relayoutData'),
        Input('graph-output43', 'relayoutData'),
        Input('graph-output44', 'relayoutData'),
        Input('graph-input51', 'relayoutData'),
        Input('graph-input52', 'relayoutData'),
        Input('graph-input53', 'relayoutData'),
        Input('graph-output54', 'relayoutData'),
        Input('graph-input55', 'relayoutData'),
        Input('graph-output56', 'relayoutData'),
        Input('graph-output61', 'relayoutData'),
        Input('graph-output62', 'relayoutData'),
        Input('graph-output63', 'relayoutData'),
        Input('graph-output71', 'relayoutData'),
        Input('graph-output72', 'relayoutData'),
        Input('graph-output73', 'relayoutData'),
        Input('graph-output74', 'relayoutData'),
        Input('graph-output81', 'relayoutData'),
        Input('graph-output82', 'relayoutData'),
        Input('graph-output83', 'relayoutData'),
        Input('graph-output84', 'relayoutData'),
        Input('graph-output91', 'relayoutData'),
        Input('graph-output92', 'relayoutData'),
        Input('graph-output93', 'relayoutData'),
        Input('graph-output94', 'relayoutData'),
        Input('dropdown-select-viewpoint', 'value')
    ],
    [
        State('camera-mesh-plot', 'data')
    ]
)
def update_mesh_state_data(input11_relData, input12_relData, input13_relData, input14_relData, output21_relData, input24_relData, output25_relData,
                           input31_relData, output32_relData, output33_relData,
                           output41_relData, output42_relData, output43_relData, output44_relData,
                           input51_relData, input52_relData, input53_relData, output54_relData, input55_relData, output56_relData,
                           output61_relData, output62_relData, output63_relData,
                           output71_relData, output72_relData, output73_relData, output74_relData,
                           output81_relData, output82_relData, output83_relData, output84_relData,
                           output91_relData, output92_relData, output93_relData, output94_relData,
                           viewpoint_path, state_data):
    did_update = False
    ctx = dash.callback_context
    for item in ctx.triggered:
        if item['prop_id'] == 'graph-input11.relayoutData' or \
            item['prop_id'] == 'graph-input12.relayoutData' or \
            item['prop_id'] == 'graph-input13.relayoutData' or \
            item['prop_id'] == 'graph-input14.relayoutData' or \
            item['prop_id'] == 'graph-output21.relayoutData'or \
            item['prop_id'] == 'graph-input24.relayoutData' or \
            item['prop_id'] == 'graph-output25.relayoutData' or \
            item['prop_id'] == 'graph-input31.relayoutData' or \
            item['prop_id'] == 'graph-output32.relayoutData' or \
            item['prop_id'] == 'graph-output33.relayoutData' or \
            item['prop_id'] == 'graph-output41.relayoutData' or \
            item['prop_id'] == 'graph-output42.relayoutData' or \
            item['prop_id'] == 'graph-output43.relayoutData' or \
            item['prop_id'] == 'graph-output44.relayoutData' or \
            item['prop_id'] == 'graph-input51.relayoutData' or \
            item['prop_id'] == 'graph-input52.relayoutData' or \
            item['prop_id'] == 'graph-input53.relayoutData' or \
            item['prop_id'] == 'graph-output54.relayoutData' or \
            item['prop_id'] == 'graph-input55.relayoutData' or \
            item['prop_id'] == 'graph-output56.relayoutData' or \
            item['prop_id'] == 'graph-output61.relayoutData' or \
            item['prop_id'] == 'graph-output62.relayoutData'or \
            item['prop_id'] == 'graph-output63.relayoutData'or \
            item['prop_id'] == 'graph-output71.relayoutData'or \
            item['prop_id'] == 'graph-output72.relayoutData'or \
            item['prop_id'] == 'graph-output73.relayoutData'or \
            item['prop_id'] == 'graph-output74.relayoutData'or \
            item['prop_id'] == 'graph-output81.relayoutData'or \
            item['prop_id'] == 'graph-output82.relayoutData'or \
            item['prop_id'] == 'graph-output83.relayoutData'or \
            item['prop_id'] == 'graph-output84.relayoutData'or \
            item['prop_id'] == 'graph-output91.relayoutData'or \
            item['prop_id'] == 'graph-output92.relayoutData'or \
            item['prop_id'] == 'graph-output93.relayoutData'or \
            item['prop_id'] == 'graph-output94.relayoutData':
            if 'scene.camera' in item['value'] and \
                ('scene.camera' not in state_data or state_data['scene.camera'] != item['value']['scene.camera']):
                state_data['scene.camera'] = item['value']['scene.camera']
                did_update = True
        elif viewpoint_path is not None:
            with open(viewpoint_path, 'rb') as f:
                state_data['scene.camera'] = pickle.load(f)
                did_update = True

        if not did_update:
            raise PreventUpdate
    return state_data

# callback to donwload viewpoint
@app.callback(
    [
        Output('viewpoint-dialog', 'message'),
        Output('viewpoint-dialog', 'displayed'),
    ],
    [
        Input('button-viewpoint-download', 'n_clicks')
    ],
    [
        State('camera-mesh-plot', 'data'),
        State('dropdown-select-file11', 'value'),
        State('dropdown-select-file12', 'value'),
    ]
)
def download_view_point(n_clicks, state_data, file11_id, file12_id):
    if n_clicks is None or n_clicks <= 0 or 'scene.camera' not in state_data:
        raise PreventUpdate
    fileName = 'fileName'
    if file11_id is not None:
        fileName = DATASET.fList[file11_id]
    if file12_id is not None:
        fileName += '+' + DATASET.fList[file12_id]
    return visualizer_utils.save_figure(params['dataDir'], fileName, state_data['scene.camera'], ext='.pkl')

# callback to list viewpoint
@app.callback(
    Output('dropdown-select-viewpoint', 'options'),
    [
        Input('button-viewpoint-list', 'n_clicks')
    ],
    [
        State('viewpoint-input', 'value')
    ]
)
def update_viewpoint_list(n_clicks, viewpoint_dir):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate
    fList = [{'label': f, 'value': os.path.join(viewpoint_dir,f)} for f in os.listdir(viewpoint_dir) if os.path.isfile(os.path.join(viewpoint_dir,f)) and f.endswith('.pkl')]
    return fList

# Callback: to update camera-3d-plot containing camera
@app.callback(
    Output('camera-3d-plot', 'data'),
    [
        Input('graph-output22', 'relayoutData')
    ],
    [
        State('camera-3d-plot', 'data')
    ]
)
def update_3d_state_data(output22_relData, state_data):
    did_update = False
    ctx = dash.callback_context
    for item in ctx.triggered:
        if item['prop_id'] == 'graph-output22.relayoutData':
            if 'scene.camera' in item['value'] and \
                ('scene.camera' not in state_data or state_data['scene.camera'] != item['value']['scene.camera']):
                state_data['scene.camera'] = item['value']['scene.camera']
                did_update = True

    if not did_update:
        raise PreventUpdate
    return state_data

# Clientside js callback: for quick updating the camera viewpoint
for i in [11,12,13,14,24,31,51,52,53,55]:
    app.clientside_callback(
        ClientsideFunction('clientside', 'update_io_figure'),
        Output('graph-input%i' %i, 'figure'),
        [
            Input('input%i-figure' %i, 'data'),
            Input('camera-mesh-plot', 'data')
        ],
    )

for i in [21,25,32,33,41,42,43,44,54,56,61,62,63,71,72,73,74,81,82,83,84,91,92,93,94]:
    app.clientside_callback(
        ClientsideFunction('clientside', 'update_io_figure'),
        Output('graph-output%i' %i, 'figure'),
        [
            Input('output%i-figure'%i, 'data'),
            Input('camera-mesh-plot', 'data')
        ],
    )

app.clientside_callback(
    ClientsideFunction('clientside', 'update_3d_figure'),
    Output('graph-output22', 'figure'),
    [
        Input('output22-figure', 'data'),
        Input('camera-3d-plot', 'data')
    ]
)

# Callback to encode input31 and update radio-encoding-dim
@app.callback(
    [
        Output('radio-encoding-dim','options'),
        Output('output32-figure', 'data'),
        Output('output32-mu', 'data'),
        Output('output32-std', 'data')
    ],
    [
        Input('dropdown-select-file31', 'value'),
        Input('exploration-color', 'value')
    ],
)
def compute_output32_graph(file_id, is_colored):
    if file_id is None:
        raise PreventUpdate
    output32_encoding, _ = run_encoder(file_id)
    if 'vae' in ENCODER_MODEL._get_name():
        output32_mu, output32_logvar = output32_encoding
    else:
        output32_mu, output32_logvar = output32_encoding, output32_encoding
    options = [
                  {'label': output32_mu.shape[1], 'value': 1},
                  {'label': output32_mu.shape[2], 'value': 2},
                  {'label': output32_mu.shape[3], 'value': 3}
              ]
    v_output = run_decoder(output32_mu)
    v_patch = get_patch_info(params['ico']['subdivisions'])
    if is_colored:
        patch_info = v_patch
    else:
        patch_info = None
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Reconstructed Output32', 350, 350, False, patch_info = patch_info)
    return options, output_figure, output32_mu.numpy(), np.exp(output32_logvar/2.0)

@app.callback(
    [
        Output('output32-dialog', 'message'),
        Output('output32-dialog', 'displayed'),
    ],
    [
        Input('button-output32', 'n_clicks')
    ],
    [
        State('dropdown-select-file31', 'value'),
        State('output32-figure', 'data'),
        State('radio-output32', 'value')
    ]
)
def save_output32_graph(n_clicks, file31_id, output32_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output32_figure is None:
        raise PreventUpdate
    fileName = 'OP_{}'.format(DATASET.fList[file31_id])
    return visualizer_utils.save_figure(params['dataDir'], fileName, output32_figure, ext)

# Callback to select dimension of the input-encoding-channel
@app.callback(
    [
        Output('input-encoding-channel1', 'max'),
        Output('input-encoding-channel2', 'max'),
        Output('input-encoding-channel3', 'max'),
        Output('input-encoding-channel4', 'max'),

    ],
    [
        Input('radio-encoding-dim','value')
    ],
    [
        State('output32-mu', 'data')
    ]
)
def update_output3_channel(dim, output32_mu):
    if dim is None:
        raise PreventUpdate
    max = torch.tensor(output32_mu).shape[dim] - 1
    return max, max, max, max

# callback to print slider-nosie-level value
@app.callback(
    Output('text-exploration','value'),
    [
        Input('slider-exploration-level1', 'value'),
        Input('slider-exploration-level2', 'value'),
        Input('slider-exploration-level3', 'value'),
        Input('slider-exploration-level4', 'value'),
        Input('button-exploration', 'n_clicks')
    ],
    [
        State('exploration-noise', 'value'),
        State('radio-encoding-dim','value'),
        State('input-encoding-channel1', 'value'),
        State('input-encoding-channel2', 'value'),
        State('input-encoding-channel3', 'value'),
        State('input-encoding-channel4', 'value'),
    ],
)
def update_output3_text(level1, level2, level3, level4, n_clicks,
                        noise, dim, channel1, channel2, channel3, channel4):
    if dim is None or channel1 is None:
        raise PreventUpdate
    std = ''
    if noise:
        std += 'E*'
    if 'vae' in ENCODER_MODEL._get_name():
        std += 'std'
    else:
        std += 'mu'
    text = ''
    if channel1 is not None:
        text += '{:+0.1f}*{}_{}_{}'.format(level1[0], std, dim, channel1)
    if channel2 is not None:
        text += '{:+0.1f}*{}_{}_{}'.format(level2[0], std, dim, channel2)
    if channel3 is not None:
        text += '{:+0.1f}*{}_{}_{}'.format(level3[0], std, dim, channel3)
    if channel4 is not None:
        text += '{:+0.1f}*{}_{}_{}'.format(level4[0], std, dim, channel4)
    return text


# callback to perform decode on the new manipulated encoding
@app.callback(
    Output('output33-figure', 'data'),
    [
        Input('slider-exploration-level1', 'value'),
        Input('slider-exploration-level2', 'value'),
        Input('slider-exploration-level3', 'value'),
        Input('slider-exploration-level4', 'value'),
        Input('exploration-color', 'value'),
        Input('button-exploration', 'n_clicks')
    ],
    [
        State('radio-encoding-dim','value'),
        State('input-encoding-channel1', 'value'),
        State('input-encoding-channel2', 'value'),
        State('input-encoding-channel3', 'value'),
        State('input-encoding-channel4', 'value'),
        State('output32-figure', 'data'),
        State('output32-mu', 'data'),
        State('output32-std', 'data'),
        State('exploration-noise', 'value')
    ],
)
def compute_output33_graph(level1, level2, level3, level4, is_colored, n_clicks, dim, channel1, channel2, channel3, channel4,
                           input12_figure, output32_mu, output32_std, noise):
    if dim is None:
        raise PreventUpdate
    new_encoding = torch.tensor(output32_mu)
    output32_std = torch.tensor(output32_std)
    if channel1 is not None:
        noise1 = level1[0] * output32_std.index_select(dim, torch.tensor(channel1))
        if noise:
            noise1 *= torch.randn_like(output32_std.index_select(dim, torch.tensor(channel1)))
        new_encoding.index_add_(dim, torch.tensor(channel1), noise1)

    if channel2 is not None:
        noise2 = level2[0] * output32_std.index_select(dim, torch.tensor(channel2))
        if noise:
            noise2 *= torch.randn_like(output32_std.index_select(dim, torch.tensor(channel2)))
        new_encoding.index_add_(dim, torch.tensor(channel2), noise2)

    if channel3 is not None:
        noise3 = level3[0] * output32_std.index_select(dim, torch.tensor(channel3))
        if noise:
            noise3 *= torch.randn_like(output32_std.index_select(dim, torch.tensor(channel3)))
        new_encoding.index_add_(dim, torch.tensor(channel3), noise3)

    if channel4 is not None:
        noise4 = level4[0] * output32_std.index_select(dim, torch.tensor(channel4))
        if noise:
            noise4 *= torch.randn_like(output32_std.index_select(dim, torch.tensor(channel4)))
        new_encoding.index_add_(dim, torch.tensor(channel4), noise4)

    # run decoder
    v_output = run_decoder(new_encoding)
    new_encoding = None
    if is_colored:
        ref_figure = input12_figure
    else:
        ref_figure = None
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Exploration Output33', 350, 350, False, ref_figure = ref_figure)
    return output_figure

@app.callback(
    [
        Output('output33-dialog', 'message'),
        Output('output33-dialog', 'displayed'),
    ],
    [
        Input('button-output33', 'n_clicks')
    ],
    [
        State('dropdown-select-file31', 'value'),
        State('text-exploration','value'),
        State('output33-figure', 'data'),
        State('radio-output33', 'value')
    ]
)
def save_output33_graph(n_clicks, file31_id, text, output33_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output33_figure is None:
        raise PreventUpdate
    fileName = 'EX_{}{}'.format(DATASET.fList[file31_id], text.replace('\n', '').replace('*','x'))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output33_figure, ext)

# compute reconstruction for dropdown-select-file41
@app.callback(
    Output('output41-figure', 'data'),
    [
        Input('dropdown-select-file41', 'value')
    ]
)
def compute_output41_figure(file_id):
    if file_id is None:
        raise PreventUpdate
    enc, _ = run_encoder(file_id)
    if 'vae' in ENCODER_MODEL._get_name():
        enc, _ = enc
    v_output = run_decoder(enc)
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Output41 ({})'.format(DATASET.fList[file_id]), 350, 350, False)
    return output_figure

@app.callback(
    [
        Output('output41-dialog', 'message'),
        Output('output41-dialog', 'displayed'),
    ],
    [
        Input('button-output41', 'n_clicks')
    ],
    [
        State('dropdown-select-file41', 'value'),
        State('output41-figure', 'data'),
        State('radio-output41', 'value')
    ]
)
def save_output41_graph(n_clicks, file41_id, output41_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output41_figure is None:
        raise PreventUpdate
    fileName = 'OP_{}'.format(DATASET.fList[file41_id])
    return visualizer_utils.save_figure(params['dataDir'], fileName, output41_figure, ext)

# generate additional shapes from the input shape
@app.callback(
    [
        Output('output42-figure', 'data'),
        Output('output43-figure', 'data'),
        Output('output44-figure', 'data'),
    ],
    [
        Input('button-generate','n_clicks')
    ],
    [
        State('dropdown-select-file41', 'value'),
        State('slider-generation-level2', 'value'),
        State('slider-generation-level3', 'value'),
        State('slider-generation-level4', 'value'),
    ]
)
def compute_output4_graph(n_clicks, file_id, noise2, noise3, noise4):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate

    enc, _ = run_encoder(file_id)
    mu, logvar = enc

    z = torch.add(mu, np.exp(logvar/2.0) * noise2[0] * torch.randn(logvar.shape))
    v_output = run_decoder(z)
    output_figure42 = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Generation Output42', 350, 350, False)

    z = torch.add(mu, np.exp(logvar/2.0) * noise3[0] * torch.randn(logvar.shape))
    v_output = run_decoder(z)
    output_figure43 = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Generation Output43', 350, 350, False)

    z = torch.add(mu, np.exp(logvar/2.0) * noise4[0] * torch.randn(logvar.shape))
    v_output = run_decoder(z)
    output_figure44 = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Generation Output44', 350, 350, False)

    return output_figure42, output_figure43, output_figure44

def save_output4_graph(n_clicks, file41_id, output4_figure, ext, noise2, noise3, noise4, id):
    if n_clicks is None or n_clicks <= 0 or output4_figure is None:
        raise PreventUpdate
    noiseLevel={}
    noiseLevel['42'] = noise2[0]
    noiseLevel['43'] = noise3[0]
    noiseLevel['44'] = noise4[0]
    fileName = 'GN_{}_{}{:+.1f}'.format(DATASET.fList[file41_id],str(id)[1:],noiseLevel[str(id)])
    return visualizer_utils.save_figure(params['dataDir'], fileName, output4_figure, ext)
for i in [42,43,44]:
    app.callback(
        [
            Output('output%i-dialog' %i, 'message'),
            Output('output%i-dialog' %i, 'displayed'),
        ],
        [
            Input('button-output%i' %i, 'n_clicks')
        ],
        [
            State('dropdown-select-file41', 'value'),
            State('output%i-figure' %i, 'data'),
            State('radio-output%i' % i, 'value'),
            State('slider-generation-level2', 'value'),
            State('slider-generation-level3', 'value'),
            State('slider-generation-level4', 'value'),
        ]
    )(partial(save_output4_graph, id=i))

# perform arithmetic on encoding
@app.callback(
    [
        Output('output54-figure','data'),
        Output('input55-figure','data'),
        Output('text-arithmetic', 'value'),
        Output('output56-figure', 'data'),
    ],
    [
        Input('button-arithmetic','n_clicks')
    ],
    [
        State('dropdown-select-file51', 'value'),
        State('dropdown-select-operation512', 'value'),
        State('dropdown-select-file52', 'value'),
        State('dropdown-select-operation523', 'value'),
        State('dropdown-select-file53', 'value'),
        State('output22-figure', 'data'),
    ]
)
def compute_output5_graph(n_clicks, file51_id, operation_512, file52_id, operation_523, file53_id, pca_figure):
    if n_clicks is None or n_clicks <= 0:
        raise PreventUpdate

    text = ''
    if file51_id is None and file52_id is None and operation_512 is None:
        raise PreventUpdate

    input51_enc, input52_enc = run_encoder(file51_id, file52_id)
    if 'vae' in ENCODER_MODEL._get_name():
        input51_enc, _ = input51_enc
        input52_enc, _ = input52_enc

    z = input51_enc + input52_enc*operation_512
    text += DATASET.fList[file51_id]
    if operation_512 == 1:
        text += '+'
    elif operation_512 == -1:
        text += '-'
    text += DATASET.fList[file52_id]


    if file53_id is not None and operation_523 is not None:
        input53_enc, _ = run_encoder(file53_id)
        if 'vae' in ENCODER_MODEL._get_name():
            input53_enc, _ = input53_enc
        z += input53_enc*operation_523
        if operation_523 == 1:
            text += '+'
        elif operation_523 == -1:
            text += '-'
        text += DATASET.fList[file53_id]

    v_output = run_decoder(z)
    output_figure54 = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Arithmetic Output54', 350, 350, False)
    if pca_figure:
        z_pca = pca.transform(z.cpu().numpy().reshape(z.shape[0], -1))
        [dist, nn_id] = enc_pca_tree.query(z_pca)
        nn_id = nn_id[0]
        v_input = get_vinput_from_idx(PCA_DATASET, nn_id)
        input_figure55 = visualizer_utils.create_mesh_figure(v_input, params['ico']['faces'], 'NN Input55 ({})-[{:.4f}]'.format(PCA_DATASET.fList[nn_id], dist[0]), 350, 350, False)
        v_NN_input, _ = PCA_DATASET[nn_id]
        v_NN_encoding = run_model(ENCODER_MODEL, v_NN_input)
        if 'vae' in ENCODER_MODEL._get_name():
            v_NN_encoding, _ = run_model(ENCODER_MODEL, v_NN_input)
        v_NN_output = run_decoder(v_NN_encoding)
        output_figure56 = visualizer_utils.create_mesh_figure(v_NN_output, params['ico']['faces'], 'NN Output56', 350,
                                                              350, False)
    else:
        input_figure55 = None
        output_figure56 = None

    return output_figure54, input_figure55, text, output_figure56

@app.callback(
    [
        Output('output54-dialog', 'message'),
        Output('output54-dialog', 'displayed'),
    ],
    [
        Input('button-output54', 'n_clicks')
    ],
    [
        State('text-arithmetic', 'value'),
        State('output54-figure', 'data'),
        State('radio-output54', 'value')
    ]
)
def save_output54_graph(n_clicks, text, output54_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output54_figure is None:
        raise PreventUpdate
    fileName = 'AR_{}'.format(text.replace('\n',''))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output54_figure, ext)

@app.callback(
    [
        Output('input55-dialog', 'message'),
        Output('input55-dialog', 'displayed'),
    ],
    [
        Input('button-input55', 'n_clicks')
    ],
    [
        State('text-arithmetic', 'value'),
        State('input55-figure', 'data'),
        State('radio-input55', 'value')
    ]
)
def save_input55_graph(n_clicks, text, input55_figure, ext):
    if n_clicks is None or n_clicks <= 0 or input55_figure is None:
        raise PreventUpdate
    fileName = 'ARNN_{}'.format(text.replace('\n',''))
    return visualizer_utils.save_figure(params['dataDir'], fileName, input55_figure, ext)

@app.callback(
    [
        Output('output56-dialog', 'message'),
        Output('output56-dialog', 'displayed'),
    ],
    [
        Input('button-output56', 'n_clicks')
    ],
    [
        State('text-arithmetic', 'value'),
        State('output56-figure', 'data'),
        State('radio-output56', 'value')
    ]
)
def save_output56_graph(n_clicks, text, output56_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output56_figure is None:
        raise PreventUpdate
    fileName = 'ARNNR_{}'.format(text.replace('\n',''))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output56_figure, ext)

# update combination radio dimension
@app.callback(
    Output('radio-combination-dim', 'options'),
    [
        Input('output62-mu','data')
    ]
)
def update_combination_options(output62_mu):

    if output62_mu is None:
        raise PreventUpdate
    output62_mu = torch.tensor(output62_mu)
    options = [
                  {'label': output62_mu.shape[1], 'value': 1},
                  {'label': output62_mu.shape[2], 'value': 2},
                  {'label': output62_mu.shape[3], 'value': 3}
              ]
    return options

# update combination range slider
@app.callback(
    [
        Output('slider-combination', 'max'),
        Output('slider-combination', 'marks'),
    ],
    [
        Input('radio-combination-dim','value')
    ],
    [
        State('output62-mu', 'data')
    ]
)
def update_combination_slider(dim, output62_mu):
    if dim is None:
        raise PreventUpdate
    max = torch.tensor(output62_mu).shape[dim]
    if dim == 1:
        keys = range(0,max+1,6)
    elif dim == 2:
        keys = range(0,max+1,int(max/5))
    elif dim == 3:
        keys = range(0,max+1,max)
    marks = {}
    for key in keys:
        marks[key] = {'label': str(key)}
    return max, marks

# update combination text
@app.callback(
    Output('text-combination', 'value'),
    [
        Input('slider-combination', 'value'),
    ],
    [
        State('dropdown-select-file61', 'value'),
        State('dropdown-select-file62', 'value'),
        State('radio-combination-dim','value'),
        State('slider-combination', 'max'),
    ]
)
def update_combination_text(combination, file61_id, file62_id, dim, max):
    if file61_id is None or file62_id is None:
        raise PreventUpdate
    return '{}_{}_{}:{}\n{}_{}_{}:{}'.format(DATASET.fList[file61_id], dim, combination[0], combination[1],
                                             DATASET.fList[file62_id], dim, combination[1], max)

# update Combination output
@app.callback(
    Output('output63-figure', 'data'),
    [
        Input('slider-combination', 'value'),
    ],
    [
        State('radio-combination-dim','value'),
        State('output61-mu', 'data'),
        State('output62-mu', 'data'),
        State('slider-combination', 'max'),
    ]
)
def compute_output63_graph(combination, dim, output61_mu, output62_mu, max):
    if dim is None:
        raise PreventUpdate
    output61_mu = torch.tensor(output61_mu)
    output62_mu = torch.tensor(output62_mu)
    new_encoding = torch.cat((output61_mu.index_select(dim, torch.tensor(range(combination[0],combination[1]))),
                              output62_mu.index_select(dim, torch.tensor(range(combination[1],max)))),
                             dim)
    # run decoder
    v_output = run_decoder(new_encoding)
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Exploration Output33', 350, 350, False)
    return output_figure

@app.callback(
    [
        Output('output63-dialog', 'message'),
        Output('output63-dialog', 'displayed'),
    ],
    [
        Input('button-output63', 'n_clicks')
    ],
    [
        State('text-combination', 'value'),
        State('output63-figure', 'data'),
        State('radio-output63', 'value')
    ]
)
def save_output63_graph(n_clicks, text, output63_figure, ext):
    if n_clicks is None or n_clicks <= 0 or output63_figure is None:
        raise PreventUpdate
    fileName = 'CM_{}'.format(text.replace('\n','+'))
    return visualizer_utils.save_figure(params['dataDir'], fileName, output63_figure, ext)

def compute_output7_graph(file7_id, id):
    if file7_id is None:
        raise PreventUpdate
    encoding, _ = run_encoder(file7_id)
    if 'vae' in ENCODER_MODEL._get_name():
        encoding, _ = encoding
    v_output = run_decoder(encoding)
    v_patch = get_patch_info(params['ico']['subdivisions'])
    output_figure = visualizer_utils.create_mesh_figure(v_output, params['ico']['faces'], 'Reconstructed Output{} ({})'.format(id,DATASET.fList[file7_id]), 350, 350, False, patch_info = v_patch)
    return output_figure
for i in [71,72,73,74]:
    app.callback(
        Output('output%i-figure' %i, 'data'),
        [
            Input('dropdown-select-file%i' % i, 'value')
        ]
    )(partial(compute_output7_graph, id=i))

# list files from general folder
for i in [8,9]:
    @app.callback(
        [
            Output('dropdown-select-file%i1' %i, 'options'),
            Output('dropdown-select-file%i2' %i, 'options'),
            Output('dropdown-select-file%i3' %i, 'options'),
            Output('dropdown-select-file%i4' %i, 'options'),
        ],
        [
            Input('button-general%i' %i, 'n_clicks')
        ],
        [
            State('general%i-input' %i, 'value')
        ]
    )
    def update_general_list(n_clicks, root_dir):
        if n_clicks is None or n_clicks <= 0:
            raise PreventUpdate
        fList = [{'label': f, 'value': f} for f in sorted(os.listdir(root_dir)) if os.path.isfile(os.path.join(root_dir,f)) and f.endswith(SHAPE_EXT)]
        return fList, fList, fList, fList

# compute general graph
def compute_output89_graph(file, root_dir, showEdges, id):
    if file is None:
        raise PreventUpdate

    fileName = os.path.join(root_dir, file)
    if fileName.endswith('.ply'):
        plydata = PlyData.read(fileName)
        v = np.vstack((plydata['vertex'].data['x'],plydata['vertex'].data['y'],plydata['vertex'].data['z']))
        f = np.vstack(plydata['face'].data['vertex_indices'])
    elif fileName.endswith('.off'):
        v, f = python_utils.read_off(fileName)
    output_figure = visualizer_utils.create_mesh_figure(v, f, 'Output{}'.format(i), 350, 350, len(showEdges))
    return output_figure
for i in [8,9]:
    for j in [1,2,3,4]:
        app.callback(
            Output('output%i%i-figure' %(i,j), 'data'),
            [
                Input('dropdown-select-file%i%i' %(i,j), 'value')
            ],
            [
                State('general%i-input' %i, 'value'),
                State('showEdges', 'value')
            ]
        )(partial(compute_output89_graph, id=i))

for i in [8,9]:
    for j in [1,2,3,4]:
        @app.callback(
            [
                Output('output%i%i-dialog' %(i,j), 'message'),
                Output('output%i%i-dialog' %(i,j), 'displayed'),
            ],
            [
                Input('button-output%i%i' %(i,j), 'n_clicks')
            ],
            [
                State('general%i-input' %i, 'value'),
                State('dropdown-select-file%i%i' %(i,j), 'value'),
                State('output%i%i-figure' %(i,j), 'data'),
                State('radio-output%i%i' %(i,j), 'value'),

            ]
        )
        def save_output89_graph(n_clicks, root_dir, file, output_figure, ext):
            if n_clicks is None or n_clicks <= 0 or file is None:
                raise PreventUpdate
            fileName = '{}'.format(os.path.splitext(os.path.join(root_dir, file))[0])
            return visualizer_utils.save_figure(params['dataDir'], fileName, output_figure, ext, False)

# python functions for repeatative tasks
def loadModelnDataset(model_name, data_instance, test_epoch):
    global ENCODER_MODEL, DECODER_MODEL, DATASET
    return_string = ''

    if 'ico2ico' in model_name:
        encoder_name = model_name.replace('ico2ico','ico2enc')
        decoder_name = model_name.replace('ico2ico','enc2ico')

    if ENCODER_MODEL is None or not ENCODER_MODEL._get_name() == encoder_name:
        # load the encoder model
        ENCODER_MODEL = eval('run.models.' + encoder_name + '(params).to(params[\'device\'],non_blocking=True)')
        return_string += 'Loaded ' + encoder_name
    else:
        return_string += 'already loaded ' + encoder_name
    # load weights to encoder model
    if not hasattr(ENCODER_MODEL, 'test_epoch') or not ENCODER_MODEL.test_epoch == test_epoch:
        if run.loadModel(params, ENCODER_MODEL, [test_epoch], model_name):
            ENCODER_MODEL.test_epoch = test_epoch
            return_string += ' with ' + test_epoch + ' epochs, '
        else:
            return_string += 'Unable to load weights'
            return return_string
    else:
        return_string += ' already with ' + test_epoch + ' epochs, '

    # switch to eval mode
    ENCODER_MODEL.eval()

    if DECODER_MODEL is None or not DECODER_MODEL._get_name() == decoder_name:
        # load the encoder model
        DECODER_MODEL = eval('run.models.' + decoder_name + '(params).to(params[\'device\'],non_blocking=True)')
        return_string += 'Loaded ' + decoder_name
    else:
        return_string += 'already loaded ' + decoder_name
    # load weights to encoder model
    if not hasattr(DECODER_MODEL, 'test_epoch') or not DECODER_MODEL.test_epoch == test_epoch:
        if run.loadModel(params, DECODER_MODEL, [test_epoch], model_name):
            DECODER_MODEL.test_epoch = test_epoch
            return_string += ' with ' + test_epoch + ' epochs, '
        else:
            return_string += 'Unable to load weights'
            return return_string
    else:
        return_string += ' already with ' + test_epoch + ' epochs, '

    # switch to eval mode
    DECODER_MODEL.eval()

    if DATASET is None or not data_instance == DATASET.data_instance:
        DATASET = eval('create'+model_name+'Dataset(params, data_instance)')
        DATASET.data_instance = data_instance

        flist = DATASET.icoList
        DATASET.fileList = [{'label': os.path.basename(flist[i]).replace('_ahs_I5.npz', ''), 'value': i} for i in
                            range(len(flist))]
        DATASET.fList = [os.path.basename(flist[i]).replace('_ahs_I5.npz', '') for i in range(len(flist))]
        DATASET.classes = list(set([f.split('_')[0] for f in DATASET.fList]))
        return_string += 'Loaded ' + data_instance + ', '
    else:
        return_string += 'already loaded ' + data_instance + ', '

    return return_string

def loadPCADataset(model_name, loaded_dataset):
    if loaded_dataset.data_instance == 'val':
        data_instance = 'trn'
    else:
        data_instance = 'val'
    other_dataset = eval('create'+model_name+'Dataset(params, data_instance)')
    other_dataset.data_instance = data_instance

    flist = other_dataset.icoList
    other_dataset.fileList = [{'label': os.path.basename(flist[i]).replace('_ahs_I5.npz', ''), 'value': i} for i in
                        range(len(flist))]
    other_dataset.fList = [os.path.basename(flist[i]).replace('_ahs_I5.npz', '') for i in range(len(flist))]
    other_dataset.classes = list(set([f.split('_')[0] for f in other_dataset.fList]))

    # Combine some attributes of the 2 datasets
    pca_dataset = combineDatasets(loaded_dataset,other_dataset)
    return other_dataset, pca_dataset


class combineDatasets(torch.utils.data.Dataset):
    def __init__(self, loaded_dataset, other_dataset):
        self.fList = loaded_dataset.fList + other_dataset.fList
        self.classes = loaded_dataset.classes
        self.loaded_dataset = loaded_dataset
        self.other_dataset = other_dataset
        self.len_loaded_dataset = self.loaded_dataset.__len__()

    def __getitem__(self, idx):
        if idx < self.len_loaded_dataset:
            return self.loaded_dataset[idx]
        else:
            idx = idx - self.len_loaded_dataset
            return self.other_dataset[idx]

    def __len__(self):
        return len(self.fList)

def get_vinput_from_idx(dataset, file_id):
    _, v_target = dataset[file_id]
    v_input = v_target[:3, :].reshape([3, -1]).transpose()
    return v_input

def run_encoder(mesh_id, mesh2_id=None):
    global ENCODER_MODEL

    # load mesh data
    v_input, _ = DATASET[mesh_id]
    if mesh2_id is not None:
        v_input2, _ = DATASET[mesh2_id]
        return run_model(ENCODER_MODEL, v_input), run_model(ENCODER_MODEL, v_input2)
    return run_model(ENCODER_MODEL, v_input), None

def run_decoder(encoding):
    global DECODER_MODEL
    v_output = run_model(DECODER_MODEL, encoding)
    if 'vae' in DECODER_MODEL._get_name():
        v_output, _, _ = v_output
    v_output = run.ico_utils.output2vertices(params['ico']['subdivisions'], v_output)
    v_output = v_output.squeeze(0).cpu().numpy()
    return v_output

def run_model(model, input):
    with torch.no_grad():
        if isinstance(input, np.ndarray):
            data = torch.from_numpy(input).unsqueeze(0)
        elif isinstance(input, torch.Tensor):
            data = input
        data = data.to(params['device'], non_blocking=True)
        return model(data)

def run_model_on_dataset(model, dataset):
    loader = run.torch.utils.data.DataLoader(dataset, batch_size=params[params['model_name']]['batch_size'], shuffle=False, num_workers=params['num_workers'], pin_memory=params['device'] == 'cuda')
    output = []
    # with torch.no_grad():
    for ip, op in tqdm.tqdm(loader):
        ip = ip.to(params['device'],non_blocking=True)
        out = model(ip)
        if 'vae' in model._get_name():
            out, _ = out
        if output == []:
            output = out.cpu().detach().numpy()
        else:
            output = np.concatenate((output,out.cpu().detach().numpy()), axis = 0)
    return output

def create_points_trace(points, text):
    text = [t.split('_')[1] if t is not None else None for t in text]
    return go.Scatter3d(
        x=[points[0][0], points[1][0], points[2][0], points[3][0]],
        y=[points[0][1], points[1][1], points[2][1], points[3][1]],
        z=[points[0][2], points[1][2], points[2][2], points[3][2]],
        mode='markers+text',
        name='Points',
        # hovertemplate='%{text}',
        text=text,
        textposition='bottom center',
        textfont=dict(size=18,))

def create_line_trace(points):
    return go.Scatter3d(
        x=[points[0][0], points[1][0]],
        y=[points[0][1], points[1][1]],
        z=[points[0][2], points[1][2]],
        mode='lines',
        name='Line')

def create_point_trace(point, name, marker_symbol):
    return go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode='markers',
        marker_symbol=marker_symbol,
        name=name
    )

def explore_enc(v_enc0, v_enc1, interpolate):
    v_enc = v_enc0*(1-interpolate) + v_enc1*interpolate
    return v_enc

def get_patch_info(subdivisions):
    patch_info = np.concatenate((
        0 * np.ones([2 ** subdivisions, 2 ** (subdivisions + 1)], dtype=int),
        1 * np.ones([2 ** subdivisions, 2 ** (subdivisions + 1)], dtype=int),
        2 * np.ones([2 ** subdivisions, 2 ** (subdivisions + 1)], dtype=int),
        3 * np.ones([2 ** subdivisions, 2 ** (subdivisions + 1)], dtype=int),
        4 * np.ones([2 ** subdivisions, 2 ** (subdivisions + 1)], dtype=int)), axis = 0)
    patch_info = patch_info.reshape(-1)
    patch_info = np.concatenate((patch_info, np.zeros((2), dtype=int)), axis=0)
    return patch_info

def write_params_from_run():
    with open('run.py') as f:
        lines = f.readlines()
        start_idx = lines.index('if __name__ == \'__main__\':\n')
        with open('get_params_from_run.py', 'w') as f:
            f.write('import os\n')
            f.write('import multiprocessing\n')
            f.write('from run import get_args\n\n')
            f.write('def get_params_from_run():\n')
            for i in range(start_idx+1, len(lines)):
                if 'get_args(params)' in lines[i]:
                    f.writelines(lines[i])
                elif not 'if params[\'process\']' in lines[i] and \
                    not 'torch_utils.save_params' in lines[i] and \
                    not 'print' in lines[i] and \
                    not '(params)' in lines[i] and \
                    not 'params[\'process_name\']' in lines[i] and \
                    not 'get_git_info' in lines[i] and \
                    not '--process' in lines[i] and \
                    not 'device' in lines[i]:
                    f.writelines(lines[i])
            f.write('    return params\n')

if __name__ == '__main__':
    ENCODER_MODEL = None
    DECODER_MODEL = None
    DATASET = None
    PCA_DATASET = None
    SHAPE_EXT = tuple(['.off', '.ply'])
    pca = None
    MODEL_OPTIONS = [{'label': 'ico2ico', 'value': 'ico2ico'},
                     {'label': 'ico2ico_vae', 'value': 'ico2ico_vae'}]
    DATASUBSET_OPTIONS = [{'label': 'trn', 'value': 'trn'},
                          {'label': 'val', 'value': 'val'}]
    ARITHMETIC_OPTIONS = [{'label': '+', 'value': 1},
                          {'label': '-', 'value': -1}]

    write_params_from_run()
    import get_params_from_run

    params = get_params_from_run.get_params_from_run()
    params['ico']['faces'] = get_ico_faces(params['ico']['subdivisions'])
    params['device'] = 'cpu'

    # 2. Define dash layout
    store = []
    for i in [11,12,13,14,24,31,51,52,53,55]:
        store.append(dcc.Store('input%i-figure' %i, data=None))
        store.append(dcc.ConfirmDialog('input%i-dialog' %i, message=None))
    for i in [21,22,25,32,33,41,42,43,44,54,56,61,62,63,71,72,73,74,81,82,83,84,91,92,93,94]:
        store.append(dcc.Store('output%i-figure' %i, data=None))
        store.append(dcc.ConfirmDialog('output%i-dialog' %i, message=None))

    app.layout = html.Div(children= store + [
        dcc.Store('topk-table', data=None),
        dcc.Store('camera-mesh-plot', data={}),
        dcc.Store('camera-3d-plot', data={}),
        html.Div(id='body',className='container scalable',
                 children=[
                     html.Div(id='app-container',
                              style={'display': 'flex','flex-direction': 'row', 'align-items': 'flex-start'},
                              children=[
                                  html.Div(id='left-column',
                                           style={'width': '260px', 'display': 'flex', 'flex-direction': 'column', 'align-items': 'left',
                                                  'margin': '0px 20px 0px 0px'},
                                           children=[
                                               html.H2('GenIcoNet', style={'margin': '0px'}),
                                               html.Div('Visualizer for Generative Icosahedral Mesh Convolutional Network', style={'margin': '0px'}),
                                               drc.Card(id='first-card',
                                                        children=[
                                                            drc.HorizontalCard(children=[
                                                                drc.NamedDropdown(
                                                                    name='Select Model',
                                                                    id='dropdown-select-model',
                                                                    options=MODEL_OPTIONS,
                                                                    clearable=False,
                                                                    searchable=False,
                                                                    value=MODEL_OPTIONS[1]['value'],
                                                                    style={"width": "110px",'margin': '0px 10px 0px 0px'},
                                                                ),
                                                                drc.NamedDropdown(
                                                                    name='Epochs',
                                                                    id='dropdown-test-epoch',
                                                                    options=[],
                                                                    clearable=False,
                                                                    value=None,
                                                                    style={"width": "70px",'margin': '0px 10px 0px 0px'},
                                                                ),
                                                                drc.NamedDropdown(
                                                                    name='Data',
                                                                    id='dropdown-select-data_instance',
                                                                    options=DATASUBSET_OPTIONS,
                                                                    clearable=False,
                                                                    searchable=False,
                                                                    value=DATASUBSET_OPTIONS[1]['value'],
                                                                    style={"width": "60px", 'margin': '0px 0px 0px 0px'},
                                                                ),
                                                            ]),
                                                            html.Button('Load Model & Files', id='button-listFiles',
                                                                        style={'width': '100%', 'margin': '10px 0px 0px 0px'}),
                                                            drc.NamedDropdown(
                                                                name='Select Input Files',
                                                                id='dropdown-select-file11',
                                                                options=[],
                                                                clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file12',
                                                                options=[],
                                                                clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file13',
                                                                options=[],
                                                                clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file14',
                                                                options=[],
                                                                clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                        ],),
                                               drc.Card(id='view-card',
                                                        children=[
                                                            html.Button('Download View Point', id='button-viewpoint-download',
                                                                        style={'margin': '0px 0px 0px 10px'}),
                                                            store.append(dcc.ConfirmDialog('viewpoint-dialog', message=None)),
                                                            drc.NamedInput('View Point Dir', id='viewpoint-input',
                                                                           style={'color': '#a5acc3', 'width': '100%',
                                                                                  'font-size': 11}),
                                                            drc.HorizontalCard(children=[
                                                                html.Button('List View Point', id='button-viewpoint-list',
                                                                        style={'margin': '0px 0px 0px 0px'}),
                                                                dcc.Checklist(id ='showEdges',
                                                                              options=[{'label': 'Edges', 'value': 1}],
                                                                              value = [],
                                                                              labelStyle={'display': 'inline-block'}),]),
                                                            dcc.Dropdown(id='dropdown-select-viewpoint',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%'}),
                                                        ],),
                                               drc.Card(id='interpolation-card',
                                                        children=[
                                                            html.P('Mesh Interpolation', style= {'text-align':'center', 'font-size': '16px'}),
                                                            drc.HorizontalCard(children=[
                                                                dcc.RadioItems(
                                                                    id='radio-operation',
                                                                    options=[
                                                                        {'label': 'interpolate', 'value': 'interpolate'},
                                                                        {'label': 'pca', 'value': 'pca'},
                                                                    ],
                                                                    value=None,
                                                                    style= {'margin': '0px 0px 0px 0px'}
                                                                ),
                                                                html.Button('List pairs', id='button-topk',
                                                                            style = {'width': '150px', 'margin': '0px 0px 0px 10px'} ),
                                                            ]),
                                                            drc.NamedRangeSlider(
                                                                name='Interpolation',
                                                                id='slider-interpolate',
                                                                min=0,
                                                                max=1.0,
                                                                value=[0],
                                                                marks={
                                                                    0: {'label': 'input11'},
                                                                    1: {'label': 'input12'},
                                                                },
                                                                step=0.1,
                                                                updatemode='mouseup',
                                                            ),
                                                            dcc.Textarea(id='text-interpolation',
                                                                         style={'width': '100%', 'color': '#a5acc3'}, ),
                                                        ], ),
                                               drc.Card(id='exploration-card',
                                                        children=[
                                                            html.P('Mesh Latent Space Exploration', style= {'text-align':'center', 'font-size': '16px'}),
                                                            drc.NamedDropdown(
                                                                name='Select File',
                                                                id='dropdown-select-file31',
                                                                options=[],
                                                                clearable=False,
                                                                value=None, style={'width': '100%'}
                                                            ),
                                                            drc.HorizontalCard(
                                                                style={'margin': '5px 0px 0px 0px'},
                                                                children=[
                                                                    html.P('Select Dimension:'),
                                                                    dcc.RadioItems(
                                                                        id='radio-encoding-dim',
                                                                        labelStyle={'display': 'inline-block'},
                                                                        options = [{'label': 'NA', 'value': 1},
                                                                                   {'label': 'NA', 'value': 2},
                                                                                   {'label': 'NA', 'value': 3}],
                                                                    ),]),
                                                            dcc.RadioItems('exploration-noise',
                                                                           options=[{'label': 'With Noise', 'value': 1},
                                                                                    {'label': 'Without Noise', 'value': 0}],
                                                                           value=0,
                                                                           labelStyle={'display': 'inline-block'}),
                                                            html.P('Select Channels:', style={'margin': '5px 0px 5px 3px'}),
                                                            dcc.Input(id='input-encoding-channel1', type = 'number',
                                                                      style={'width': '25%', 'color': '#a5acc3', 'margin': '0px 0px 0px 0px'},),
                                                            dcc.Input(id='input-encoding-channel2', type = 'number',
                                                                      style={'width': '25%', 'color': '#a5acc3', 'margin': '0px 0px 0px 0px'},),
                                                            dcc.Input(id='input-encoding-channel3', type = 'number',
                                                                      style={'width': '25%', 'color': '#a5acc3', 'margin': '0px 0px 0px 0px'},),
                                                            dcc.Input(id='input-encoding-channel4', type = 'number',
                                                                      style={'width': '25%', 'color': '#a5acc3'},),
                                                            html.P('Select Levels:', style={'margin': '10px 0px -10px 3px'}),
                                                            drc.HorizontalCard(children=[
                                                                dcc.RangeSlider(id='slider-exploration-level1',
                                                                                min=-3.0, max=3.0, value=[0.0], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                dcc.RangeSlider(id='slider-exploration-level2',
                                                                                min=-3.0, max=3.0, value=[0.0], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                dcc.RangeSlider(id='slider-exploration-level3',
                                                                                min=-3.0, max=3.0, value=[0.0], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                dcc.RangeSlider(id='slider-exploration-level4',
                                                                                min=-3.0, max=3.0, value=[0.0], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                                ]),
                                                            dcc.Store(id='output32-mu',data=None),
                                                            dcc.Store(id='output32-std',data=None),
                                                            dcc.RadioItems('exploration-color',
                                                                           options=[{'label': 'With Color', 'value': 1},
                                                                                    {'label': 'Without Color', 'value': 0}],
                                                                           value=1,
                                                                           labelStyle={'display': 'inline-block'}),
                                                            html.Button('Compute', id='button-exploration',
                                                                        style={'width': '100%',
                                                                               'margin': '10px 0px 0px 0px'}),

                                                            dcc.Textarea(id='text-exploration',
                                                                style={'width': '100%', 'color': '#a5acc3'},),
                                                        ], ),
                                               drc.Card(id='generation-card',
                                                        children=[
                                                            html.P('Mesh Re-Generation', style={'text-align': 'center', 'font-size': '16px'}),
                                                            drc.NamedDropdown(
                                                                name='Select File',
                                                                id='dropdown-select-file41',
                                                                options=[],
                                                                clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            html.P('Select Noise Level:', style={'margin': '5px 0px 5px 3px'}),
                                                            drc.HorizontalCard(children=[
                                                                dcc.RangeSlider(id='slider-generation-level2',
                                                                                min=-3.0, max=3.0, value=[-1.5], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                dcc.RangeSlider(id='slider-generation-level3',
                                                                                min=-3.0, max=3.0, value=[-1.0], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                dcc.RangeSlider(id='slider-generation-level4',
                                                                                min=-3.0, max=3.0, value=[1.2], step=0.1,
                                                                                updatemode='mouseup',
                                                                                marks={-3: '-3', 0: '0', 3: '3'},
                                                                                vertical=True, verticalHeight = 200,
                                                                                tooltip = {'always_visible': True, 'placement': 'left'},),
                                                                                ]),
                                                            html.Button('Compute', id='button-generate',
                                                                        style={'width': '100%',
                                                                               'margin': '10px 0px 0px 0px'}),
                                                        ],),
                                               drc.Card(id='arithmetic-card',
                                                        children=[
                                                            html.P('Mesh Arithmetic', style= {'text-align':'center', 'font-size': '16px'}),
                                                            drc.NamedDropdown(
                                                                name='Select Files',
                                                                id='dropdown-select-file51',
                                                                options=[], clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-operation512',
                                                                options=ARITHMETIC_OPTIONS, clearable=False,
                                                                value=None,
                                                                style={'width': '25%',}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file52',
                                                                options=[], clearable=False,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-operation523',
                                                                options=ARITHMETIC_OPTIONS, clearable=True,
                                                                value=None,
                                                                style={'width': '25%',}
                                                            ),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file53',
                                                                options=[], clearable=True,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            html.Button('Compute', id='button-arithmetic',
                                                                        style={'width': '100%',
                                                                               'margin': '10px 0px 0px 0px'}),
                                                            dcc.Textarea(id='text-arithmetic',
                                                                style={'width': '100%', 'color': '#a5acc3'},),
                                                        ], ),
                                               drc.Card(id='combination-card',
                                                        children=[
                                                            html.P('Patch-wise Mesh Combination',
                                                                   style={'text-align': 'center', 'font-size': '16px'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file61',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Store(id='output61-mu', data=None),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file62',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}
                                                            ),
                                                            dcc.Store(id='output62-mu', data=None),
                                                            drc.HorizontalCard(
                                                                style={'margin': '5px 0px 0px 0px'},
                                                                children=[
                                                                    html.P('Select Dimension:'),
                                                                    dcc.RadioItems(
                                                                        id='radio-combination-dim',
                                                                        labelStyle={'display': 'inline-block'},
                                                                        options=[{'label': 'NA', 'value': 1},
                                                                                 {'label': 'NA', 'value': 2},
                                                                                 {'label': 'NA', 'value': 3}],
                                                                    ), ]),
                                                            drc.NamedRangeSlider('Input61',
                                                                id='slider-combination',
                                                                min=0,
                                                                step=1,
                                                                updatemode='mouseup',
                                                                tooltip = {'always_visible': True, 'placement': 'topRight'}
                                                            ),
                                                            dcc.Textarea(id='text-combination',
                                                                         style={'width': '100%', 'color': '#a5acc3'}, ),

                                                        ], ),
                                               drc.Card(id='patch-card',
                                                        children=[
                                                            html.P('Patch Visualization',
                                                                   style={'text-align': 'center', 'font-size': '16px'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file71',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file72',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file73',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file74',
                                                                options=[],
                                                                clearable=False,
                                                                searchable=True,
                                                                value=None,
                                                                style={'width': '100%'}),
                                                        ], ),
                                               drc.Card(id='general8-card',
                                                        children=[
                                                            drc.NamedInput('Input dir to view files', id='general8-input', value='',
                                                                           style={'color': '#a5acc3', 'width': '100%', 'font-size': 11}),
                                                            html.Button('List Files', id='button-general8',
                                                                        style={'width': '100%',
                                                                               'margin': '10px 0px 0px 0px'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file81',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file82',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file83',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file84',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                        ],),
                                               drc.Card(id='general9-card',
                                                        children=[
                                                            drc.NamedInput('Input dir to view files', id='general9-input', value='',
                                                                           style={'color': '#a5acc3', 'width': '100%',
                                                                                  'font-size': 11}),
                                                            html.Button('List Files', id='button-general9',
                                                                        style={'width': '100%',
                                                                               'margin': '10px 0px 0px 0px'}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file91',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file92',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file93',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                            dcc.Dropdown(
                                                                id='dropdown-select-file94',
                                                                options=[],
                                                                value=None,
                                                                style={'width': '100%', 'font-size': 11}),
                                                        ], ),
                                           ],),
                                  html.Div(id='graph-column',
                                           style={'display': 'flex', 'flex-direction': 'column'},
                                           children=[
                                               html.Div(id='predefined-inputs', children=['logDir: ' + params['logDir']]),
                                               html.Div(id='logging-state'),
                                               html.Div(id='graph1-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'input11', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input12', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input13', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input14', style={'width': '350px', 'height': '350px'},),),
                                                                  ],),
                                               html.Div(id='graph2-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'output21', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output22',style={'width': '450px', 'height': '450px'},),),
                                                                  html.Div(children=dash_table.DataTable(id='table-topk',columns=[
                                                                      {'name': 'Shape Pairs', 'id': 'column1'},
                                                                      {'name': 'Distance', 'id': 'column2',
                                                                       'type': 'numeric',
                                                                       'format': dash_table.Format.Format(precision=5,
                                                                                        #scheme=Scheme.fixed
                                                                                        ),}],),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input24', style={'width': '350px','height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output25', style={'width': '350px','height': '350px'}, ), ),
                                                                  ],),
                                               html.Div(id='graph3-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'input31', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output32', style={'width': '350px', 'height': '350px'},),),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output33', style={'width': '350px', 'height': '350px'},),),
                                                                  ], ),
                                               html.Div(id='graph4-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'output41', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output42', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output43', style={'width': '350px','height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output44', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  ], ),
                                               html.Div(id='graph5-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'input51', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input52', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input53', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  ], ),
                                               html.Div(id='graph5A-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'output54', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'input55', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output56', style={'width': '350px', 'height': '350px'}, ), ),
                                                                  ], ),
                                               html.Div(id='graph6-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[html.Div(children=drc.ButtonedGraph('Download', 'output61',
                                                                                                      style={
                                                                                                          'width': '350px',
                                                                                                          'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output62',
                                                                                                      style={
                                                                                                          'width': '350px',
                                                                                                          'height': '350px'}, ), ),
                                                                  html.Div(children=drc.ButtonedGraph('Download', 'output63',
                                                                                                      style={
                                                                                                          'width': '350px',
                                                                                                          'height': '350px'}, ), ),
                                                                  ], ),
                                               html.Div(id='graph7-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output71',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output72',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output73',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output74',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            ], ),
                                               html.Div(id='graph8-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output81',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output82',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output83',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output84',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                        ], ),
                                               html.Div(id='graph9-container',
                                                        style={'display': 'flex', 'flex-direction': 'row'},
                                                        children=[
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output91',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output92',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output93',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                            html.Div(children=drc.ButtonedGraph('Download', 'output94',
                                                                                                style={
                                                                                                    'width': '350px',
                                                                                                    'height': '350px'}, ), ),
                                                        ], ),
                                           ])
                              ],)],),])

    app.run_server(debug=True, port=8050)
