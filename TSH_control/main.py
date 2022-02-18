from dash import Dash, dcc, html
from flask import Flask
import deta_utils

app = Flask(__name__)
dash_app = Dash(server=app)

d = deta_utils.download('get_mood')
data = {'x': list(d['mood']), 'y': list(d['mood'].values()), 'name': 'mood', 'marker': {'color': 'rgb(55, 83, 109)'}}

dash_app.layout = html.Div([
    dcc.Input(
        placeholder='Enter a value...',
        type='text',
        value=''
    ),
    dcc.Graph(
        figure=dict(
            data=[
                data
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

