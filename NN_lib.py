import numpy as np
import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import h5py as h5

class Fit(object):
    def __init__(self,nstates):
        self.n = nstates
    def E(self,p,i):
        En = p['E0']
        for j in range(1,i):
            En += np.exp(p['E%s' %j])
        return En
    def Z(self,p,i,s):
        return p['Z%s_%s' %(i,s)]
    def twopt(self,x,p,src,snk):
        r = 0
        for i in range(self.n):
            En = self.E(p,i)
            Zn_i = self.Z(p,i,src)
            Zn_f = self.Z(p,i,snk)
            r += Zn_i*Zn_f*np.exp(-En*x)
        return r
    def twopt_ssps(self,x,p):
        r=dict()
        r['SS'] = self.twopt(x,p,'s','s')
        r['PS'] = self.twopt(x,p,'s','p')
        return r

if __name__=='__main__':
    fpath = './data/nn_new_cl3_b6p1_m0p02450.h5'
    dpath = '/cl3_24_48_b6p1_m0p2450/a/nn_I0/T1_plus_3S1/mu_0/nsq_0'
    Aclass = Analysis(fpath,dpath)
    Aclass.read_data()
    print(Aclass.data)
