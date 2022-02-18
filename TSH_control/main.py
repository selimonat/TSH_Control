from dash import Dash, dcc, html
from flask import Flask
import deta_utils

app = Flask(__name__)
dash_app = Dash(server=app)

d = deta_utils.download(['get_mood', 'get_tsh', 'get_dosage'])
d_mood = {'x': list(d['mood']), 'y': list(d['mood'].values()), 'name': 'mood', 'marker': {'color': 'rgb(55, 83, 109)'}}
d_tsh = {'x': list(d['tsh']), 'y': list(d['tsh'].values()), 'name': 'tsh', 'marker': {'color': 'rgb(55, 83, 109)'}}
d_dosage = {'x': list(d['dosage']), 'y': list(d['dosage'].values()), 'name': 'dosage', 'marker': {'color': 'rgb(55, 83, '
                                                                                                     '109)'}}

dash_app.layout = html.Div([
    dcc.Input(
        placeholder='Enter a value...',
        type='text',
        value=''
    ),
    dcc.Graph(
        figure=dict(
            data=[
                d_mood, d_tsh, d_dosage
            ],
            layout=dict(
                title='US Export of Plastic Scrap',
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1.0
                ),
                margin=dict(l=40, r=0, t=40, b=30)
            )
        ),
        style={'height': 300},
        id='my-graph'
    )
])

if __name__ == '__main__':
    dash_app.run_server(debug=False)

