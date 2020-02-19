# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # datasetssets processing, CSV file I/O (e.g. pd.read_csv)
# from framework import DATASET_PATH
# Input datasets files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../datasets"))

import matplotlib.pyplot as plt
from PIL import Image

import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20, .2f}'.format

import plotly.io as pio
pio.renderers.default = "browser"

# %matplotlib inline
import numpy as np # linear algebra
import pandas as pd # datasets processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
#from collections import Counter
import networkx as nx 
# Input datasets files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../datasets/train_relationships.csv")
print(train_df.head(),train_df.shape)

files = [os.path.join(dp, f) 
         for dp, dn, fn in os.walk(os.path.expanduser("../datasets/train")) for f in fn]
train_images_df = pd.DataFrame({
    'files': files,
    'familyId': [file.split('/')[3] for file in files],
    'kinId': [file.split('/')[4] for file in files],
    'uniqueId': [file.split('/')[3] + '/' + file.split('/')[4] for file in files]
})

print(train_images_df.head())
print()

print("Total number of members in the dataset: {0}".format(train_images_df["uniqueId"].nunique()))
print("Total number of families in the dataset: {0}".format(train_images_df["familyId"].nunique()))


family_with_most_pic = train_images_df["familyId"].value_counts()
kin_with_most_pic = train_images_df["uniqueId"].value_counts()
print("Family with maximum number of images: {0}, Image Count: {1}".format(family_with_most_pic.index[0], family_with_most_pic[0]))
print("Member with maximum number of images: {0}, Image Count: {1}".format(kin_with_most_pic.index[0], kin_with_most_pic[0]))

family_series = family_with_most_pic[:25]
labels = (np.array(family_series.index))
sizes = (np.array((family_series / family_with_most_pic.sum()) * 100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Pic Count by Families')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='Families')
# fig.show(renderer="svg")


most_pic_members = train_images_df[train_images_df["uniqueId"] == kin_with_most_pic.index[0]].files.values
fig, ax = plt.subplots(4, 6, figsize=(50, 40))
row = 0
col = 0
for index in range(len(most_pic_members[:24])):
    with open(most_pic_members[index], 'rb') as f:
        img = Image.open(f)
        ax[row][col].imshow(img)

        if(col < 5):
            col = col + 1
        else: 
            col = 0
            row = row + 1
fig.show()

family_with_most_members = train_images_df.groupby("familyId")["kinId"].nunique().sort_values(ascending=False)
print("Family with maximum number of members: {0}, Member Count: {1}".format(family_with_most_members.index[0], family_with_most_members[0]))
print("Family with least number of members: {0}, Member Count: {1}".format(
    family_with_most_members.index[len(family_with_most_members)-1], 
    family_with_most_members[len(family_with_most_members)-1]))

large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[0]]
print(large_family_df.head())

def render_bar_chart(datasets_df, column_name, title, filename):
    series = datasets_df[column_name].value_counts()
    count = series.shape[0]
    
    trace = go.Bar(x = series.index, y=series.values, marker=dict(
        color=series.values,
        showscale=True
    ))
    layout = go.Layout(title=title)
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=filename)
    
    
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')

def render_images(large_family_df):
    large_family_pics = [large_family_df.loc[large_family_df.loc[large_family_df["uniqueId"] == aKin].index[0]]["files"] for aKin in large_family_df["uniqueId"].unique()]
    nrows = round(len(large_family_pics) / 6) + 1


    fig, ax = plt.subplots(nrows, 6, figsize=(50, 40))
    row = 0
    col = 0
    for index in range(len(large_family_pics)):
        with open(large_family_pics[index], 'rb') as f:
            img = Image.open(f)
            ax[row][col].imshow(img)

            if(col < 5):
                col = col + 1
            else: 
                col = 0
                row = row + 1
    fig.show()
render_images(large_family_df)

large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[1]]
render_images(large_family_df)
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')

large_family_df = train_images_df[train_images_df["familyId"]  == family_with_most_members.index[2]]
render_images(large_family_df)
render_bar_chart(large_family_df, 'uniqueId', 'Pic Count by Members', 'members')

# train_df = pd.read_csv("../datasets/train_relationships.csv")
# print(train_df.head())

fam = os.listdir("../datasets/train")
print('We have',len(fam),'families')
ind = []
num = []
pic = []
tot = 0
totpic = 0
for i in fam:
    path = "../datasets/train/"+str(i)
    temp = os.listdir(path)
    ind.append(temp)
    num.append(len(temp))
    tot+=len(temp)
    for j in temp:
        newpath = path+"/"+str(j)
        temp = os.listdir(newpath)
        pic.append(temp)
        totpic+=len(temp)
print('And',tot,'individuals with',totpic,'pictures.')
print('On average, we see',tot/len(fam),'members per family.')
print('With an average of',totpic/tot,'per individual.')

# Create graph from datasets 
g = nx.Graph()
color_map = []
for i in range(0,len(fam)): #len(names)
    g.add_node(fam[i], type = 'fam')
    for j in ind[i]:
        temp = fam[i]+j
        g.add_node(temp, type = 'ind')
        g.add_edge(fam[i], temp, color='green', weight=1)
for n1, attr in g.nodes(data=True):
    if attr['type'] == 'fam':
        color_map.append('lime')
    else: 
        if attr['type'] == 'ind':
            color_map.append('cyan')
        else:
            color_map.append('red')


plt.figure(3,figsize=(90,90))  
edges = g.edges()
colors = [g[u][v]['color'] for u,v in edges]
nx.draw(g,node_color = color_map, edge_color = colors, with_labels = True)
plt.show()

print('Reference Graph')
print('Do we have a fully connected graph? ',nx.is_connected(g))

nx.isolates(g)
h = g.to_directed()
N, K = h.order(), h.size()
avg_deg= float(K) / N
print ("# Nodes: ", N)
print ("# Edges: ", K)
print ("Average Degree: ", avg_deg)
# Extract reference graph facts & metrics 
in_degrees= h.in_degree() # dictionary node:degree
