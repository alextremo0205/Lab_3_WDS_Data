import torch
import itertools
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class Normalizer:
    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        
        self.attributes = ['base_demand', 
                           'elevation', 
                           'base_head', 
                           'length', 
                           'roughness', 
                           'diameters',
                           'pressure']

        self.mins = {}
        self.maxs = {}

        for atr in self.attributes:
            self.mins[atr] = self.get_func_from_tensor(torch.min, training_dataset, atr)
            self.maxs[atr] = self.get_func_from_tensor(torch.max, training_dataset, atr)
        
    def get_normalized_training_dataset(self):
        normalized_training_dataset = self.normalize_dataset(self.training_dataset)
        return normalized_training_dataset
    
    def normalize_example(self, example):
        for atr in self.attributes:
            example['norm_'+atr] = (example[atr] - self.mins[atr])/(self.maxs[atr]-self.mins[atr])
        return example

    def normalize_dataset(self, dataset):
        normalized_dataset = list(map(self.normalize_example, dataset))
        return normalized_dataset

    def get_func_from_tensor(self, func, training_dataset, field):
        return func(torch.Tensor([func(example[field]) for example in training_dataset]))

    def unnormalize_attribute(self, example, attribute):
        norm_attribute = example[attribute]
        unnormalized_attribute = norm_attribute*(self.maxs[attribute] - self.mins[attribute]) + self.mins[attribute]

        return unnormalized_attribute

    def unnormalize_pressure_tensor(self, norm_estimation):
        unnormalized_pressure = norm_estimation*(self.maxs['pressure'] - self.mins['pressure']) + self.mins['pressure']
        return unnormalized_pressure



def plot_distribution_attribute_in_element(dataset, attribute, node=None, link=None):
    
    assert node != None or link != None, 'Choose a node or a link'

    if node != None:
        element  = node
        name_element = 'node'
    elif link != None:
        element  = link
        name_element = 'link'
    
    attribute_in_db = []
    for i in dataset:
        attribute_in_db.append(i[attribute].reshape(-1,1))
        
    block_attribute = torch.cat(attribute_in_db, dim = 1)
    block_np = block_attribute.numpy()
    block_pd = pd.DataFrame(block_np)

    ax = block_pd.iloc[element,:].plot.hist(bins = 30)#(marker='o', linestyle='none')

    ax.set_xlabel(attribute+" at "+name_element+" {} ".format(node))
    ax.set_ylabel("Frequency")



def plot_simulated_attribute_in_element(dataset, attribute, node = None, link = None):
    
    assert node != None or link != None, 'Choose a node or a link'

    if node != None:
        element  = node
        name_element = 'node'
    elif link != None:
        element  = link
        name_element = 'link'
        
    attribute_in_db = []
    for i in dataset:
        attribute_in_db.append(i[attribute].reshape(-1,1))
        
    block_attribute = torch.cat(attribute_in_db, dim = 1)
    block_np = block_attribute.numpy()
    block_pd = pd.DataFrame(block_np)

    ax = block_pd.iloc[element,:].plot(marker='o', linestyle='none')
    ax.set_xlabel("Simulation")
    ax.set_ylabel(attribute + " at "+name_element+" {} ".format(node))



def get_trainable_params(model):
    # this function returns the number of trainable parameters in a model
    return sum(p.numel() for p in model.parameters())
  


def get_accuracy(model, X, Y):
    total = Y.argmax(1) == model(X).argmax(1)
    correct = total.sum()
    accuracy = correct/total.shape[0]
    return accuracy.item()*100
    

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,figsize=(5,5)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    adapted from https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')
    
    print(cm)
    
    f, ax = plt.subplots(1,figsize=figsize)
    ax.imshow(cm, interpolation='nearest', cmap=cmap,)
    ax.set_title(title)
    # ax.setcolorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      ax.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
    
    f.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')




def plot_pressure_comparison(target_across_sims, estimation_across_sims):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(target_across_sims))), 
                            y=target_across_sims, marker = dict(size = 10),
                            mode='markers', 
                            name='Target', 
                            hovertemplate = '%{text}', 
                            text = ['<b></b> {value:.2f} m'.format(value = i) for i in target_across_sims]))

    fig.add_trace(go.Scatter(x=list(range(len(estimation_across_sims))), 
                            y=estimation_across_sims, marker = dict(size = 10),
                            mode ='markers', 
                            name='Estimation', 
                            hovertemplate = '%{text}', 
                            text = ['<b></b> {value:.2f} m'.format(value = i) for i in estimation_across_sims]))

    fig.update_layout(title = 'Pressure comparison', 
                    xaxis_title ="Simulation",
                    yaxis_title ="Pressure (m)",
                    legend_title="Legend",
                    yaxis_range = [0,60],
                    template =  custom_template)
    fig.show()



def plot_error_bar_plot(error, xaxis_title):
    fig = go.Figure()
    fig.add_trace(go.Bar(x = list(range(len(error))), y = error))
    fig.update_layout(title = 'Error in estimated pressure', 
                        xaxis_title =xaxis_title,
                        yaxis_title ="Pressure difference (m)",
                        legend_title="Legend",
                        template =  custom_template)
    fig.show()



custom_template = {
    "layout": go.Layout(
        font={
            "family": "Nunito",
            "size": 16,
            "color": "#707070",
        },
        title={
            "font": {
                "family": "Lato",
                "size": 22,
                "color": "#1f1f1f",
            },
        },
        xaxis={
            "showspikes":   True,
            "spikemode":    'across',
            "spikesnap":    'cursor',
            "showline":     True,
            "showgrid":     True,
        },
        hovermode  = 'x',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
    )
}