# imports
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/user/Downloads/2014_World_Power_Consumption.csv')
# Choropleth Plot of the Power Consumption for Countries
data = dict(
        type = 'choropleth',
        locations = df['Country'],
        locationmode = "country names",
        z = df['Power Consumption KWH'],
        text = df['Country'],
        colorbar = {'title' : 'Power Consumption KWH	'},
      )
layout = dict(
    title = '2Power Consumption KWH',
    geo = dict(
        showframe = False,
        projection = {'type':'mercator'}
    )
)
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
plt.show()
df = pd.read_csv("C:/Users/user/Downloads/2012_Election_Data.csv")
data = dict(type = 'choropleth',
            locations = df['State Abv'],
            locationmode = 'USA-states',
            colorscale= 'Portland',
            text= ['text1','text2','text3'],
            z=df['Voting-Age Population (VAP)'],
            colorbar = {'title':'Colorbar Title'})

layout = dict(geo = {'scope':'usa'})
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
plt.show()