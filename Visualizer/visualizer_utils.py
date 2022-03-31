from dash.exceptions import PreventUpdate
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as io
import pymesh
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import python_utils


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def create_mesh_figure(v, f, name, width, height, showLines=False, ref_figure = None, patch_info = None, \
                       intensity = None, colorscale = None, cmin = None, cmax = None, hovertemplate = ''):
    if v.shape[1] != 3:
        v = np.transpose(v)
    if f.shape[1] != 3:
        f = np.transpose(f)
    if ref_figure is not None:
        reference = np.stack([np.asarray(ref_figure['data'][0]['x']), np.asarray(ref_figure['data'][0]['y']),
                              np.asarray(ref_figure['data'][0]['z'])], axis=0)
        intensity = python_utils.euclideanDistance(v, reference)
        intensitymode = 'vertex'
        colorscale = [[0.0, 'rgb(116,126,152)'], [1, 'rgb(256,0,0)']]
        cmin = 0.0
        cmax = 0.1
    elif patch_info is not None:
        intensity = patch_info
        intensitymode = 'vertex'
        colorscale = px.colors.diverging.Portland
        cmin = 0.0
        cmax = 5.0
    elif intensity is not None and colorscale is not None and cmin is not None and cmax is not None:
        intensitymode = 'vertex'
        colorscale = colorscale
        cmin = cmin
        cmax = cmax
    else:
        mesh = pymesh.meshio.form_mesh(vertices=v,faces=f)
        intensity = np.ones(f.shape[0])
        intersecting_faces = pymesh.detect_self_intersection(mesh)[:,0]
        # v, f, intersecting_faces = get_inverted_faces(mesh)
        intensity[intersecting_faces] = 0.0
        intensitymode = 'cell'
        colorscale = [[0.0, 'rgb(0,0,0)'], [1.0, 'rgb(116,126,152)']]
        cmin = 0.0
        cmax = 1.0

    plot_mesh = go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color='rgb(116,126,152)',
        flatshading=False,
        intensity=intensity,
        intensitymode=intensitymode,
        showscale=True,
        colorscale=colorscale,#[[0.0,'rgb(116,126,152)'],[1,'rgb(107,0,156)']],#'Hot',
        hoverinfo='none',
        hovertemplate=hovertemplate,
        cmin=cmin,  # atrick to get a nice plot (z.min()=-3.31909)
        cmax=cmax,
        lighting=dict(ambient=0.9,
                      diffuse=0.5,
                      fresnel=0,
                      specular=1.5,
                      roughness=1.0,
                      facenormalsepsilon=1e-15,
                      vertexnormalsepsilon=1e-15),
        lightposition=dict(x=3,
                           y=3,
                           z=3),
    )

    layout = go.Layout(
        title=name,
        font=dict(size=10, color='grey'),
        width=width,
        height=height,
        scene_xaxis_visible=False,
        scene_yaxis_visible=False,
        scene_zaxis_visible=False,
        # paper_bgcolor='rgb(50,50,50)',
        margin=dict(l=0,r=0,t=30,b=0),
        scene=dict(aspectmode='data')
    )

    if showLines:
        tri = np.stack((v[f[:, 0]], v[f[:, 1]], v[f[:, 2]], v[f[:, 0]]))
        Xe = []
        Ye = []
        Ze = []
        for i in np.arange(tri.shape[1]):
            Xe.extend(tri[:, i, 0].tolist() + [None])
            Ye.extend(tri[:, i, 1].tolist() + [None])
            Ze.extend(tri[:, i, 2].tolist() + [None])
        lines = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode='lines',
            name='',
            line=dict(color='rgb(70,70,70)', width=1))
        fig = go.Figure(data=[plot_mesh, lines], layout=layout)
    else:
        fig = go.Figure(data=[plot_mesh], layout=layout)
    return fig

def create_3d_figure(dataset, data, name, width, height):
    if len(dataset.classes) > 1:
        color = [dataset.classes.index(f.split('_')[0]) for f in dataset.fList]
    else:
        color = np.zeros((len(data)), dtype=float)
    plot_3d = go.Scatter3d(
        x=data[:,0],
        y=data[:,1],
        z=data[:,2],
        mode='markers',
        marker=dict(
            # size=12,
            color=color,  # set color to an array/list of desired values
            colorscale='curl',  # choose a colorscale
            opacity=0.8
        ),
        hovertemplate = '%{text}',
        text = dataset.fList,
        name = name,
    )
    plot_3d.hoverlabel = {'bgcolor': [plot_3d.marker.colorscale[int(t)][1] for t in color]}
    layout = go.Layout(
        # title=name,
        #font=dict(size=10, color='white'),
        width=width,
        height=height,
        #paper_bgcolor='rgb(50,50,50)',
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend_orientation='h',
        uirevision='true',
    )
    trace1 = go.Scatter3d()  # Four points
    trace2 = go.Scatter3d()  # Interpolation Line
    trace3 = go.Scatter3d()  # Interpolation Point
    trace4 = go.Scatter3d()  # Nearest neighbour PCA

    fig = go.Figure(data=[plot_3d,trace1,trace2,trace3,trace4], layout=layout)
    return fig

def save_figure(dataDir, fileName, figure, ext='.png', isFullPathGiven=False):
    if not fileName.endswith(ext):
        fileName += ext
    if not isFullPathGiven:
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)
        fullFileName = os.path.join(dataDir, fileName)
    else:
        fullFileName = fileName
    tmp = 1
    while os.path.exists(fullFileName):
        if '({})'.format(tmp-1) in fullFileName:
            fullFileName = fullFileName.replace('({}){}'.format(tmp-1,ext), '({}){}'.format(tmp,ext))
        else:
            fullFileName = fullFileName.replace(ext, '('+str(tmp)+')'+ext)
        tmp += 1

    if ext == '.off':
        vertices = np.vstack([figure['data'][0]['x'], figure['data'][0]['y'], figure['data'][0]['z']])
        faces = np.vstack([figure['data'][0]['i'], figure['data'][0]['j'], figure['data'][0]['k']])
        python_utils.writeOffMesh(fullFileName, vertices, faces)
    elif ext == '.png':
        io.write_image(figure, fullFileName, engine='kaleido')
    elif ext=='.pkl':
        with open(fullFileName, 'wb') as f:
            pickle.dump(figure, f)
    else:
        raise PreventUpdate
    text = 'Downloaded {}'.format(fullFileName)
    return text, True
