import torch
import itertools
import numpy as np
import pandas as pd

import wntr

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


def get_resilience_index(head, pressure, demand, flowrate, wn, Pstar):
    """
    Compute Prasad & Park index.

    The Prasad & Park index is related to the capability of a system to overcome
    failures while still meeting demands and pressures at the nodes. 
    
    The Todini index defines resilience at a specific time as a measure of surplus
    power at each node and measures relative energy redundancy.
    
    The modification that Prasad & Park include is the uniformity coefficient C. 

    Parameters
    ----------
    head : pandas DataFrame
        A pandas Dataframe containing node head 
        (index = times, columns = node names).
        
    pressure : pandas DataFrame
        A pandas Dataframe containing node pressure 
        (index = times, columns = node names).
        
    demand : pandas DataFrame
        A pandas Dataframe containing node demand 
        (index = times, columns = node names).
        
    flowrate : pandas DataFrame
        A pandas Dataframe containing pump flowrates 
        (index = times, columns = pump names).

    wn : wntr WaterNetworkModel
        Water network model.  The water network model is needed to 
        find the start and end node to each pump.

    Pstar : float
        Pressure threshold.

    Returns
    -------
    A pandas Series that contains a time-series of Prasad & Park indexes
    """

    POut = {}
    PExp = {}
    PInRes = {}
    PInPump = {}

    time = head.index
    
    for name in wn.junction_name_list:
        #Begin---------------------- Modification-------------------------
        diams = []
        adj_pipes = wn.get_links_for_node(name)
        for i in adj_pipes:
            try:
                diams.append(wn.get_link(i).diameter)
            except:
                pass
				#print('Pump at link: ', i)
        c = sum(diams)/(len(diams)*max(diams))
        #print(diams, c)
        
        #End---------------------- Modification-------------------------
        
        h = np.array(head.loc[:,name]) # m
        p = np.array(pressure.loc[:,name])
        e = h - p # m
        q = np.array(demand.loc[:,name]) # m3/s
        #print(q, h, c, q*h*c)
        
        #Begin---------------------- Modification-------------------------
        POut[name] = q*h*c
        PExp[name] = q*(Pstar+e)*c
        #End---------------------- Modification-------------------------

    for name, node in wn.nodes(wntr.network.Reservoir):
        H = np.array(head.loc[:,name]) # m
        Q = np.array(demand.loc[:,name]) # m3/s
        PInRes[name] = -Q*H # switch sign on Q.

    for name, link in wn.links(wntr.network.Pump):
        start_node = link.start_node_name
        end_node = link.end_node_name
        h_start = np.array(head.loc[:,start_node]) # (m)
        h_end = np.array(head.loc[:,end_node]) # (m)
        h = h_start - h_end # (m)
        q = np.array(flowrate.loc[:,name]) # (m^3/s)
        PInPump[name] = q*(abs(h)) # assumes that pumps always add energy to the system

    PPindx = (sum(POut.values()) - sum(PExp.values()))/  \
        (sum(PInRes.values()) + sum(PInPump.values()) - sum(PExp.values()))

    PPindx = pd.Series(data = PPindx.tolist(), index = time)

    return PPindx

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
