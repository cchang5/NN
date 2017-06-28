import numpy as np
import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import h5py as h5

# fitting params
show = False
nstates = 5
tmin = [3,10]
tmax = [14,14]
# priors
p = dict()
#LC NN
p['NN_A0'] = gv.gvar(5E-10, 2.5E-10)
p['NN_A1'] = gv.gvar(0.0, 5E-9)
p['NN_A2'] = gv.gvar(0.0, 5E-9)
p['NN_A3'] = gv.gvar(0.0, 5E-9)
p['NN_A4'] = gv.gvar(0.0, 5E-9)
p['NN_A5'] = gv.gvar(0.0, 5E-9)
p['NN_E0'] = gv.gvar(1.467, 0.2)
p['NN_E1'] = gv.gvar(-0.93, 0.7) # 2*(2*pi*n/L) 
p['NN_E2'] = gv.gvar(-1.8, 0.28)
p['NN_E3'] = gv.gvar(-2.0, 0.25)
p['NN_E4'] = gv.gvar(-2.25, 0.13)
p['NN_E5'] = gv.gvar(-2.38, 0.1)
# Single nucleon
p['N_A0'] = gv.gvar(1.8E-5, 9E-6)
p['N_A1'] = gv.gvar(0.0, 1.8E-4)
p['N_A2'] = gv.gvar(0.0, 1.8E-4)
p['N_A3'] = gv.gvar(0.0, 1.8E-4)
p['N_A4'] = gv.gvar(0.0, 1.8E-4)
p['N_A5'] = gv.gvar(0.0, 1.8E-4)
p['N_E0'] = gv.gvar(0.72, 0.02)
p['N_E1'] = gv.gvar(-0.45, 0.7)
p['N_E2'] = gv.gvar(-0.45, 0.7)
p['N_E3'] = gv.gvar(-0.45, 0.7)
p['N_E4'] = gv.gvar(-0.45, 0.7)
p['N_E5'] = gv.gvar(-0.45, 0.7)

# read file
LC = h5.File('./data/NN_LC.mpi700.32c64.x000+100.hdf5','r')
LCsinglet = np.array(LC['boost_px0py0pz0']['ppcorr_SING_0_0_px0py0pz0'])
LCsingletgv = gv.dataset.avg_data(LCsinglet[:,:,0])

LCmeff = np.log(LCsingletgv/np.roll(LCsingletgv,-1))
fig = plt.figure('NN singlet meff', figsize=(7,4.326237))
ax = plt.axes([0.15,0.15,0.8,0.8])
ax.errorbar(x=np.arange(len(LCmeff)), y=[i.mean for i in LCmeff], yerr=[i.sdev for i in LCmeff], ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
plt.draw()
if show:
    plt.show()

rng = [5,15]
LCscaledcorr = LCsingletgv*np.exp(LCmeff*np.arange(len(LCsingletgv)))
fig = plt.figure('NN singlet Aeff', figsize=(7,4.326237))
ax = plt.axes([0.15,0.15,0.8,0.8])
ax.errorbar(x=np.arange(len(LCscaledcorr))[rng[0]:rng[1]], y=np.array([i.mean for i in LCscaledcorr])[rng[0]:rng[1]], yerr=np.array([i.sdev for i in LCscaledcorr])[rng[0]:rng[1]], ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
plt.draw()
if show:
    plt.show()

LC = h5.File('./data/NN_LC.mpi700.32c64.x000+100.hdf5','r')
LCp = np.array(LC['boost_px0py0pz0']['pcorr'])
LCpgv = gv.dataset.avg_data(LCp[:,:,0])

LCpmeff = np.log(LCpgv/np.roll(LCpgv,-1))
fig = plt.figure('N meff', figsize=(7,4.326237))
ax = plt.axes([0.15,0.15,0.8,0.8])
ax.errorbar(x=np.arange(len(LCpmeff)), y=[i.mean for i in LCpmeff], yerr=[i.sdev for i in LCpmeff], ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
plt.draw()
if show:
    plt.show()

LCpscaledcorr = LCpgv*np.exp(LCpmeff*np.arange(len(LCpgv)))
fig = plt.figure('N Aeff', figsize=(7,4.326237))
ax = plt.axes([0.15,0.15,0.8,0.8])
ax.errorbar(x=np.arange(len(LCpscaledcorr))[rng[0]:rng[1]], y=np.array([i.mean for i in LCpscaledcorr])[rng[0]:rng[1]], yerr=np.array([i.sdev for i in LCpscaledcorr])[rng[0]:rng[1]], ls='None',marker='o',fillstyle='none',markersize='5',elinewidth=1,capsize=2)
plt.draw()
if show:
    plt.show()

T = len(LCpgv)

class fitfunction():
    def __init__(self,T,nstates):
        self.T = T
        self.nstates = nstates
        return None
    def N_E(self,p,n):
        E = p['N_E0']
        for i in range(1,n+1):
            E += np.exp(p['N_E%s' %str(i)])
        return E
    def N_A(self,p,n):
        return p['N_A%s' %str(n)]
    def twopt(self,x,p):
        C = 0
        for i in range(self.nstates):
            A = self.N_A(p,i)
            E = self.N_E(p,i)
            C += A*np.exp(-E*x)
        return C
    def NN_E(self,p,n):
        E = p['NN_E0']
        for i in range(1,n+1):
            E += np.exp(p['NN_E%s' %str(i)])
        return E
    def NN_A(self,p,n):
        return p['NN_A%s' %str(n)]
    def NN(self,x,p):
        C = 0
        for i in range(self.nstates):
            A = self.NN_A(p,i)
            E = self.NN_E(p,i)
            C += A*np.exp(-E*x)
        return C
    def proton_NN_LC(self,x,p):
        proton = self.twopt(x,p)
        NN = self.NN(x,p)
        result = np.concatenate((proton,NN))
        return result

def find_priors(p,nstates):
    priors = dict()
    for k in p.keys():
        if int(k[-1]) < int(nstates):
            priors[k] = p[k]
        else: pass
    return priors

# concatenate data
data = np.concatenate((LCp[:,:,0],LCsinglet[:,:,0]),axis=1)
datagv = gv.dataset.avg_data(data)
ff = fitfunction(T,nstates)
prior = find_priors(p,nstates)
print prior
for t1 in range(tmin[0],tmin[1]+1):
    for t2 in range(tmax[0],tmax[1]+1):
        x = np.arange(t1,t2+1)
        y = datagv[np.concatenate((x,x+T))]
        fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=ff.proton_NN_LC)
        #print fit
        print t1, t2, fit.chi2/fit.dof, fit.p['NN_E0']-2.*fit.p['N_E0']
