from dash import Dash, dcc, html
from flask import Flask
import deta_utils
from plotly.subplots import make_subplots
import plotly.graph_objs as go

app = Flask(__name__)
dash_app = Dash(server=app)

d = deta_utils.download(['get_mood', 'get_tsh', 'get_dosage', 'get_cumulative_complaints', 'get_apple_health_distance',
                         'get_apple_health_weight', 'get_correlation'])

d_mood = {'x': list(d['mood']), 'y': list(d['mood'].values()), 'name': 'mood', 'marker': {'color': 'rgb(55, 83, 109)'}}
d_tsh = {'x': list(d['tsh']), 'y': list(d['tsh'].values()), 'name': 'tsh', 'marker': {'color': 'rgb(55, 83, 109)'}}
d_dosage = {'x': list(d['dosage']), 'y': list(d['dosage'].values()), 'name': 'dosage',
            'marker': {'color': 'rgb(55, 83, 109)'}}

d_complaints = {'x': list(d['complaints']), 'y': list(d['complaints'].values()), 'name': 'complaints',
                'marker': {'color': 'rgb(55, 83, 109)'}}

d_weight = {'x': list(d['weight']), 'y': list(d['weight'].values()), 'name': 'weight',
            'marker': {'color': 'rgb(55, 83, 109)'}}

d_exercice = {'x': list(d['distance']), 'y': list(d['distance'].values()), 'name': 'distance',
              'marker': {'color': 'rgb(55, 83, 109)'}}


d = deta_utils.download(['get_correlation'])
col_name = list(d['correlation'])
d = d['correlation']

L = list()
for k1 in list(d):
    l = list()
    for k, v in d[k1].items():
        l.append(v)
    L.append(l)


# corr_fig = px.imshow(L, x=col_name, y=col_name, text_auto=".2f", aspect="auto")

corr_fig = go.Figure(data=go.Heatmap(
    z=L,
    x=col_name,
    y=col_name,
    texttemplate="%{z:.2f}",
    showscale=True,
    ))
corr_fig.layout.title = 'Pearson Correlation'

fig = make_subplots(rows=6,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    horizontal_spacing=0.009,
                    subplot_titles=("Mood", "TSH", "Dosage", "Complaints", "Weight", "Exercice"),
                    )
fig.append_trace(d_mood, 1, 1)
fig.append_trace(d_tsh, 2, 1)
fig.append_trace(d_dosage, 3, 1)
fig.append_trace(d_complaints, 4, 1)
fig.append_trace(d_weight, 5, 1)
fig.append_trace(d_exercice, 6, 1)

fig['layout']['margin'] = {'l': 100, 'r': 10, 'b': 50, 't': 50}
fig.layout.title = 'Physiological Time-Series'

dash_app.layout = html.Div(children=[
                                    dcc.Graph(
                                    figure=fig,
                                    style={'width': '35%', 'height': '1000px',
                                           'display': 'inline-block',
                                           'vertical-align': 'top'},
                                    id='my-graph'),
                                    dcc.Graph(
                                        figure=corr_fig,
                                        style={'width': '30%', 'height': '30%',
                                               'display': 'inline-block',
                                               'vertical-align': 'top'},
                                        id='my-graph2')
                                    ],)

if __name__ == '__main__':
    dash_app.run_server(debug=False)

