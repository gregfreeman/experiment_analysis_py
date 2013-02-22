import  sys
import json
import urllib
import urlparse
import numpy as np
import os

def load_json(folder,file):
    parts = urlparse.urlsplit(folder)  
    data=[]
    if parts.scheme and parts.netloc:      
        filename=folder+'/'+file
        data_file=urllib.urlopen(filename)
        data=json.load(data_file)
    else:
        filename=os.path.join(folder,file)
        data_file=open(filename)
        data=json.load(data_file)
    return data



class Experiment:
    """Loads, filters, process experiment data."""

    def __init__(self,folder=''):
        self.results = []
        self.paramset = []
        self.info = []
        self.dim= []
        if(folder!=''):
            self.load(folder)

    def load(self, folder):
        results =load_json(folder,'results.json')
        paramset =load_json(folder,'paramset.json')
        if type(paramset)!=list: # single parameters are serialized as a dictionary
            paramset=[paramset]
        self.paramset=paramset
        try:
            self.info =load_json(folder,'info.json')
        except:
            self.info =[]
        dim=[]
        for param in self.paramset:
            dim.append(len(param['values']))
        dim2=dim[::-1]
        ndim=len(dim)
        results = np.array(results).reshape(dim2)
        results=np.transpose(results,ndim-1-np.arange(ndim))
        self.results = results
        self.dim=dim

    def filter(self, filters):
        d=self.results
        idxs=[]
        ndim=self.results.ndim
        for i,param in  enumerate(self.paramset):
            field=param['field']
            values=param['values']

            if filters.has_key(field):
                test=filters[field]
                if not isinstance(test, list):
                    test=[test]
                idx=[idx for idx, val in enumerate(values) if (val in test) ]
                idxs.append(idx)
                k=np.repeat(np.s_[:],ndim)
                k[i]=idx
                d=d[tuple(k)]
            else:
                idxs.append(range(len(param['values'])))
        return d


    def series(self,filters,xvar,selector):
        # look for keyword means in 
        filters2={ a:b for a,b in filters.items() if b!='mean' }
        means=[ a for a,b in filters.items() if b=='mean' ]
        d=self.filter(filters2)
        y=np.ndarray(d.shape)
        for i, val in np.ndenumerate(d):
            y[i]=selector(val)
        ndim=d.ndim     
        for i,x in enumerate(self.paramset):
            if(x['field'] in means):
                # take sum across dimension
                y=y.mean(i)
                # return singleton dimension
                k=np.repeat(np.s_[:],ndim)
                k[i]=None
                y=y[tuple(k)]
        
        i=next((i for i,x in enumerate(self.paramset) if xvar==x['field']), None)
        p=range(y.ndim)
        del p[i]
        p.insert(0, i)
        y=np.transpose(y, p)
        x=self.paramset[i]['values']
        return x,y
