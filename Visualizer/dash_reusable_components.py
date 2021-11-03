from textwrap import dedent

import dash_core_components as dcc
import dash_html_components as html


# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(className="card", children=children, style={'border-bottom': '1px solid rgba(255, 255, 255, 0.1)',
                                                                    'padding': '10px 0px 10px 0px'},
                        **_omit(["style"], kwargs))

def HorizontalCard(children, **kwargs):
    return html.Section(className="horizontalCard", children=children,
                        style={'display': 'flex', 'flex-direction': 'row',
                               'justifyContent': 'space-around'},
                        **_omit(["style"], kwargs))

def FormattedSlider(**kwargs):
    return html.Div(
        style=kwargs.get("style", {}), children=dcc.Slider(**_omit(["style"], kwargs))
    )


def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "10px 0px 0px 0px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )


def NamedRangeSlider(name, **kwargs):
    return html.Div(
        style={"padding": "10px 0px 0px 0px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "0px"}, children=dcc.RangeSlider(**kwargs)),
        ],
    )


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px 0px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin": "0px 0px 5px 3px"}),
            dcc.Dropdown(**kwargs),
        ],
    )


def NamedInput(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px 0px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin": "0px 0px 5px 3px"}),
            dcc.Input(**kwargs),
        ],
    )

def NamedRadioItems(name, **kwargs):
    return html.Div(
        style={"padding": "10px 0px 0px 0px"},
        children=[html.P(children=f"{name}:"), dcc.RadioItems(**kwargs)],
    )

def ButtonedGraph(button_name, id, style, **kwargs):
    graph_style = {}
    return html.Div(
        children=[
            HorizontalCard(children=[
                dcc.RadioItems('radio-' + id,
                               options=[{'label': '.off', 'value': '.off'},
                                        {'label': '.png', 'value': '.png'}],
                               value='.png',
                               labelStyle={'display': 'inline-block'}),
                html.Button(button_name, id='button-' + id),
            ]),
            dcc.Graph(id='graph-' + id, style={**graph_style, **style}, **kwargs),
        ],
        style={'margin': '5px 0px 0px 0px'}
    )

# Non-generic
def DemoDescription(filename, strip=False):
    with open(filename, "r") as file:
        text = file.read()

    if strip:
        text = text.split("<Start Description>")[-1]
        text = text.split("<End Description>")[0]

    return html.Div(
        className="row",
        style={
            "padding": "15px 30px 27px",
            "margin": "45px auto 45px",
            "width": "80%",
            "max-width": "1024px",
            "borderRadius": 5,
            "border": "thin lightgrey solid",
            "font-family": "Roboto, sans-serif",
        },
        children=dcc.Markdown(dedent(text)),
    )
