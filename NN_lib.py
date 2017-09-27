# two nucleon library
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

def read_data(h5file):
    data = dict()
    for s in ['a','b']:
        data[s] = dict()
        data[s]['cfg'] = np.array(h5file['cl3_24_48_b6p1_m0p2450'][s]['cfgs'])
        data[s]['nucleon'] = np.squeeze(np.array(h5file['cl3_24_48_b6p1_m0p2450'][s]['nucleon']['nsq_0']))
        data[s]['nn_I0'] = dict()
        for q in range(4):
            data[s]['nn_I0']['nsq_%s' %q] = np.squeeze(np.array(h5file['cl3_24_48_b6p1_m0p2450'][s]['nn_I0']['T1_plus_3S1']['mu_avg']['nsq_%s' %q]))
        data[s]['nn_I1'] = dict()
        for q in range(4):
            data[s]['nn_I1']['nsq_%s' %q] = np.squeeze(np.array(h5file['cl3_24_48_b6p1_m0p2450'][s]['nn_I1']['A1_plus_1S0']['mu_avg']['nsq_%s' %q]))
    return data

def plot_meff(gvdata):
    fig = plt.figure('effective mass',figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    x = np.arange(len(gvdata)//2)
    meff = np.log(gvdata/np.roll(gvdata,-1))[x]
    ax.errorbar(x = x, y = [d.mean for d in meff], yerr = [d.sdev for d in meff], ls='None', marker='o', capsize=2, fillstyle='none')
    plt.draw()

def plot_scor(gvdata):
    fig = plt.figure('scaled correlator',figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    x = np.arange(len(gvdata)//2)
    meff = np.log(gvdata/np.roll(gvdata,-1))[x]
    scor = gvdata[x]*np.exp(meff*x)
    ax.errorbar(x = x, y = [d.mean for d in scor], yerr = [d.sdev for d in scor], ls='None', marker='o', capsize=2, fillstyle='none')
    plt.draw()

class Fit(object):
    def __init__(self,params):
        self.nstates = params['nstates']
        self.key = params['data_select']
    def priors(self,prior):
        p = dict()
        for d in self.key:
            for q in prior:
                if (set(d.split('_')).issubset(q.split('_')) or q.split('_')[-1][0] is 'E') and int(q[-1]) < self.nstates:
                    p[q] = prior[q]
        return p
    def En(self,p,n,ag):
        E = p['%s_E0' %ag]
        for i in range(1,n+1):
            E += np.exp(p['%s_E%s' %(ag,i)])
        return E
    def An(self,p,n,tag):
        A = p['%s_A%s' %(tag,n)]
        return A
    def two_point_correlator(self,x,p,tag):
        ag = tag.replace('new_','').replace('old_','').replace('_local','').replace('_nonlocal','')
        r = 0
        for n in range(self.nstates):
            E = self.En(p,n,ag)
            A = self.An(p,n,tag) 
            r += A*np.exp(-E*x)
        return r
    def fit_function(self,x,p):
        r = dict()
        for k in self.key:
            r[k] = self.two_point_correlator(x,p,k)
        return r
