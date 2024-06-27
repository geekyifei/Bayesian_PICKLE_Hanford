# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:26:11 2022

@author: yifei
"""
#%% inmport dependencies

import os
import copy
import numpy as np
import numpy.random as npr
import scipy.optimize as spo
import scipy.linalg as spl
from time import perf_counter
from matplotlib import pyplot as plt, collections as mc, patches as mpatches, cm
import h5py
import GPy
import seaborn as sns
import pandas as pd

from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.tpfa import TPFA
from sdfs.tpfa_tf import Tpfa_tf
import ckli.mapest as mapest
import ckli.ckliest_l2reg as ckliest

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions

#%% Helper functions

rl2e = lambda yest, yref : spl.norm(yest - yref, 2) / spl.norm(yref + Yfac, 2) 
infe = lambda yest, yref : spl.norm(yest - yref, np.inf) 
lpp = lambda h, href, sigma: np.sum( -(h - href)**2/(2*sigma**2) - 1/2*np.log( 2*np.pi) - 2*np.log(sigma))

def plot_patch(patches, values, fig, ax, points, title, fontsize = 15, cmin=None, cmax=None, cb=False):
    p = mc.PatchCollection(patches, cmap=cm.jet)
    p.set_array(values)
    p.set_clim([cmin, cmax])
    ax.add_collection(p)
    if points is not None:
        ax.plot(*points, 'ko', markersize=0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale(tight=True)
    ax.set_title(title, fontsize = fontsize)
    if cb:
        fig.colorbar(p, ax=ax)
    return p

def plot_all(patches, Yref, Ypred, Y_std, Y_env, fontsize = 15, title = None, savefig = True):
    plt.rc('text', usetex=False)
    plt.rc('image', cmap='plasma')
    points = geom.cells.centroids[:, iYobs]
    diff = np.abs(Ypred - Yref)
    
    fig, ax = plt.subplots(1, 4, figsize = (16,4), dpi = 300)
    plot_patch(patches, Ypred + Yfac, fig, ax[0], points, title = 'Y pred mean', fontsize = fontsize, cmin = Yref.min() + Yfac, cmax = Yref.max() + Yfac, cb = True)
    plot_patch(patches, diff, fig, ax[1], points, title = 'Y diff', fontsize = fontsize, cmin = diff.min(), cmax = diff.max(), cb = True)
    plot_patch(patches, Y_std, fig, ax[2], points, title = 'Y pred std', fontsize = fontsize, cmin = Y_std.min(), cmax = Y_std.max(), cb = True)
    plot_patch(patches, Y_env, fig, ax[3], points, title = 'Y coverage', fontsize = fontsize, cmin = 0, cmax = 1, cb = True)
    fig.tight_layout()
    
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_statistics_all_HMC.png'))

def plot_each(patches, Yref, Ypred, Y_std,  Y_env, fontsize = 15, title = None, savefig = True):
    plt.rc('text', usetex=False)
    plt.rc('image', cmap='plasma')
    points = geom.cells.centroids[:, iYobs]
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    plot_patch(patches, Ypred + Yfac, fig, ax, points, None, fontsize, cmin = Yref.min() + Yfac, cmax = Yref.max() + Yfac, cb = True)
    fig.tight_layout()
    if title:
        ax.set_title('Y pred mean')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_mean_HMC.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    diff = np.abs(Ypred - Yref)
    plot_patch(patches, diff, fig, ax, points, None, fontsize, cmin = diff.min(), cmax = diff.max(), cb = True)
    fig.tight_layout()
    if title:
        ax.set_title('Y diff')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_diff_HMC.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    plot_patch(patches, Y_std, fig, ax, points, None, fontsize, cmin = Y_std.min(), cmax = Y_std.max(), cb = True)
    fig.tight_layout()
    if title:
        ax.set_title('Y pred std ')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_std_HMC.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    plot_patch(patches, Y_env, fig, ax, points, None, fontsize, 0, 1, True)
    fig.tight_layout()
    if title:
        ax.set_title('Y coverage')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_coverage_HMC.png'))


#%% Parameters

seed = 0 #random seed
res_fac = 1
resolution = '1x' 
resolution_fine = '16x' #splitting cell into finer equal-area cells
field = 'Y_HD' 
sigma_r = 0.1
sigma_p = 1
sigma_q = 1
Nchains = 5
Nsamples = 5000
Nburn = 20000
std_dev = 1.0 #std for the covariance kernel
cor_len = 0.1 #correlation length for the covariance kernel
Neumann_sd = 0
lsq_method = 'trf'
data_path = r'./test/data/'
geom_filename = data_path + f'geom/geom_{resolution}.mat'
geom_fine_filename = data_path + f'geom/geom_{resolution_fine}.mat'
bc_filename = data_path + f'bc/bc_{resolution}.mat'
conduct_filename = data_path + f'RF1/conduct_log_RF1_{resolution}.mat'
well_cells_filename = data_path + f'well_cells/well_cells_{resolution}.mat'


#%% Y and u reference field

Yfac = 7.0 # Rescaling factor for log-conductivity. Must be applied to Yref and the BCs
geom = GeomMRST(geom_filename)
bc = BCMRST(geom, bc_filename)
bc.rescale('N', Yfac)
tpfa = TPFA(geom, bc)
rs = npr.RandomState(seed)
Nc = geom.cells.num 
Ninf = geom.faces.num_interior 
patches = [mpatches.Polygon(v, closed=True) for v in geom.nodes.coords.T[geom.cells.nodes.T, :]] #get the coordinate of nodes of polygons (1475, 2, 4)

if field == 'Y_HD':
    
    NYobs = 100
    Nuobs = 323
    NYxi = 1000
    Nuxi = 1000
    Nens = 5000
    yobs_filename = data_path + f'yobs/yobs_{NYobs}_{resolution}.npy'
    yobs_fine_filename = data_path + f'yobs/yobs_{NYobs}_{resolution_fine}.npy'
    
    with h5py.File(conduct_filename, 'r') as f:
        Yref = f.get('conduct_log')[:].ravel() - Yfac #reference log-conductivity (1475,)
    prob = DarcyExp(TPFA(geom, bc), None) 
    uref = prob.randomize_bc('N', Neumann_sd).solve(Yref) # Given Yref, solve for reference uref (1475,)
    
    with h5py.File(well_cells_filename, 'r') as f:
        iuobs = f.get('well_cells')[:].ravel() - 1 # observation well index
    uobs = uref[iuobs] #extract u obs from reference
  
elif field == 'Y_LD':
    
    NYobs = 5
    Nuobs = 5
    NYxi = 20
    Nuxi = 10
    Nens = 5000
    Yref = np.loadtxt(os.path.abspath(r'./test/data/yobs/Y_smooth.out')) - Yfac
    yobs_filename = data_path + f'yobs/yobs_{NYobs}_{resolution}.npy'
    yobs_fine_filename = data_path + f'yobs/yobs_{NYobs}_{resolution_fine}.npy'
    prob = DarcyExp(TPFA(geom, bc), None) 
    uref = prob.randomize_bc('N', Neumann_sd).solve(Yref) #Given Yref, solve for reference uref (1475,)

    # u observations
    with h5py.File(well_cells_filename, 'r') as f:
        iuobs = f.get('well_cells')[:].ravel() - 1 #observation well index
    
    iuobs = rs.choice(iuobs, Nuobs, replace  = False)
    uobs = uref[iuobs] #extract u obs from reference

save_pth = f'{field}_NY_{NYobs}_Nu_{Nuobs}_Nuxi_{Nuxi}_NYxi_{NYxi}_sigmar_{sigma_r}_HMC_Nchains_{Nchains}_Nburn_{Nburn}_Nsamples_{Nsamples}'
results_path =  os.path.join(save_pth, 'results')
if not os.path.exists(results_path):
    os.makedirs(results_path)
fig_path = os.path.join('.', save_pth, 'figures')
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
f_rec = open(os.path.join(results_path,'record.out'), 'a+')

if os.path.exists(yobs_filename):
    print(f"iYobs set read from file {yobs_filename}")
    print(f"iYobs set read from file {yobs_filename}", file = f_rec)
    iYobs = np.load(yobs_filename)
    iYobs = iYobs[0] if iYobs.ndim > 1 else iYobs
    
elif os.path.exists(yobs_fine_filename):
    print(f"iYobs set read from file {yobs_fine_filename} and randomly selected nearby cell")
    print(f"iYobs set read from file {yobs_fine_filename} and randomly selected nearby cell",file = f_rec)
    iYobs_fine = np.load(yobs_fine_filename)
    geom_fine = GeomMRST(geom_fine_filename)
    iYobs = geom.anyCellsWithin(geom_fine.nodes.coords.T[geom_fine.cells.nodes.T[iYobs_fine]])
    np.save(yobs_filename, iYobs)
    
else:
    print("iYobs same as iuobs")
    print("iYobs same as iuobs", file = f_rec)
    iYobs = iuobs
    np.save(yobs_filename, iYobs)

print(f"Yobs index shape: {iYobs.shape}")
print(f'Yobs index: {iYobs}')
print(f"Yobs index shape: {iYobs.shape}", file = f_rec)
print(f'Yobs index: {iYobs}', file = f_rec)

#%% constrcuting conditional mean and covariance for Y and u, where Y is from GPR and u from MC ensemble

Yobs = Yref[iYobs] 
ts = perf_counter()
klearn = GPy.kern.sde_Matern52(input_dim=2, variance=std_dev**2, lengthscale=cor_len) #matern 52 kernel - prior
mYlearn = GPy.models.GPRegression(geom.cells.centroids[:, iYobs].T, Yobs[:, None], klearn, noise_var=np.sqrt(np.finfo(float).eps))
mYlearn.optimize(messages=True, ipython_notebook=False)
mYref = GPy.models.GPRegression(geom.cells.centroids[:, iYobs].T, Yobs[:, None], mYlearn.kern, noise_var=np.sqrt(np.finfo(float).eps)) #optimized model with conditional kernel
Ypred, CYpred = (lambda x, y : (x.ravel(), y))(*mYref.predict_noiseless(geom.cells.centroids.T, full_cov=True)) #GPR prediction of Y
timings = perf_counter() - ts

rel_err_gpr = rl2e(Ypred, Yref) 
abs_err_gpr = infe(Ypred, Yref) 
sigmaYpred = np.sqrt(np.diag(CYpred))
variance = float(mYlearn.kern.variance.values)
lengthscale = float(mYlearn.kern.lengthscale.values)

print("############# GPR Results #################")
print(f"GPR: {timings:.3f} s")
print(f'GPR variance: {variance:.3f}')
print(f'GPR lengthscale: {lengthscale:.3f}')
print(f"GPR\tRelative error of Y: {rel_err_gpr:.3f}")
print(f"GPR\tInfinity error of Y: {abs_err_gpr:.3f}") 

print(f"GPR: {timings:.3f} s", file = f_rec)
print(f'GPR variance: {variance:.3f}', file = f_rec)
print(f'GPR lengthscale: {lengthscale:.3f}', file = f_rec)
print(f"GPR\tRelative error of Y: {rel_err_gpr:.3f}", file = f_rec)
print(f"GPR\tInfinity error of Y: {abs_err_gpr:.3f}", file = f_rec) 
print("###########################################\n")

ts = perf_counter()
umean, Cu = ckliest.smc_gp(Ypred, CYpred, Nens, copy.deepcopy(prob), rs, randomize_bc=True, randomize_scale=Neumann_sd) #N_ens = 5000
upred, Cupred = ckliest.gpr(umean, Cu, uobs, iuobs) 
sigmaupred = np.sqrt(np.diag(Cupred))
timings = perf_counter() - ts

print(f"Monte Carlo: {timings:.3f} s\n")
print(f"Monte Carlo: {timings:.3f} s\n", file = f_rec)

Ym = Ypred 
CYm = CYpred
um = upred 
Cum = Cupred 

ts = perf_counter()
PsiY, LambdaY = ckliest.KL_via_eigh(CYm, NYxi)
Psiu, Lambdau = ckliest.KL_via_eigh(Cum, Nuxi)
timings = perf_counter() - ts

print(f"eigendecomposition: {timings:.3f} s\n")
print(f"eigendecomposition: {timings:.3f} s\n", file = f_rec)

# Deterministic PICKLE (PICKLE-MAP estimates)
ssv = None if Neumann_sd == 0 else np.delete(np.arange(Nc), np.unique(geom.cells.to_hf[2*geom.faces.num_interior:][bc.kind == 'N']))
Lreg = mapest.compute_Lreg(geom) 
beta_ckli = 0
res = ckliest.LeastSqRes(NYxi, Ym, PsiY, Nuxi, um, Psiu, prob, sigma_r, sigma_p, sigma_q, res_fac, iuobs, uobs, iYobs, Yobs, beta_ckli, ssv=ssv)
ts = perf_counter()
sol = spo.least_squares(res.val, np.zeros(Nuxi + NYxi), jac=res.jac, method='trf', verbose=2) 
timings = perf_counter() - ts

uxi = sol.x[:Nuxi] 
Yxi = sol.x[Nuxi:] 
theta_pickle = np.concatenate((np.array(Yxi, dtype = np.float64), np.array(uxi, dtype = np.float64)))
upickle = um + Psiu @ uxi
Ypickle = Ym + PsiY @ Yxi
rel_err_pickle = rl2e(Ypickle, Yref) 
abs_err_pickle = infe(Ypickle, Yref) 

print("############# PICKLE Results #################")
print(f"PICKLE: {timings:.3f} s")
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.3f}")
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.3f}")
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.3f}", file = f_rec)
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.3f}\n", file = f_rec)
print("#############################################\n")
    
    
#%% Bayesian PICKLE - HMC

tf.keras.backend.set_floatx('float64')
dtype = tf.float64
seed_num = 8888
tpfa2 = Tpfa_tf(geom, bc)
sigma_r_tf = tf.constant(sigma_r, dtype = dtype) 
Psiu_tf = tf.constant(Psiu, dtype = dtype)
PsiY_tf = tf.constant(PsiY, dtype = dtype) 
um_tf = tf.constant(um, dtype = dtype) 
Ym_tf = tf.constant(Ym, dtype = dtype) 
loc = tf.constant(0., dtype = dtype)
scale = tf.constant(1., dtype = dtype)

@tf.function
def kl_pred(theta): #correctly return residual vector with batch dimension in the first axis
    def wrapper_fn(inputs):
        u_, Y_= inputs 
        res_ = tpfa2.residual(u_, Y_)
        return (res_, res_)
    u = um_tf + tf.linalg.einsum('ij,nj->ni', Psiu_tf, theta[:,NYxi:])
    Y = Ym_tf + tf.linalg.einsum('ij,nj->ni', PsiY_tf, theta[:,:NYxi])
    res = tf.map_fn(wrapper_fn, (u,Y))
    return res[0]

@tf.function
def target_log_prob_fn(theta):
    prior = tf.reduce_sum(-tf.math.log(scale) - tf.math.log(2*tf.constant(np.pi, dtype = dtype))/2 -(theta - loc)**2/(2*scale**2), axis = 1)
    r_likelihood = tf.reduce_sum(-tf.math.log(sigma_r_tf) - tf.math.log(2*tf.constant(np.pi, dtype = dtype))/2 -(kl_pred(theta))**2/(2*sigma_r_tf**2), axis = 1)
    return prior + r_likelihood

nuts_kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn = target_log_prob_fn, step_size = 1e-8, max_tree_depth=15, max_energy_diff = 1000.0,
    unrolled_leapfrog_steps=1, parallel_iterations=30
    )

adapt_nuts_dual = tfp.mcmc.DualAveragingStepSizeAdaptation(
    nuts_kernel, num_adaptation_steps= int(Nburn * 0.75)
    )


init_state = tf.zeros((1, NYxi + Nuxi), dtype = dtype)
init_state = tf.concat((init_state, tf.random.normal((1, NYxi + Nuxi), 0, 1, dtype = dtype)), axis = 0)
init_state = tf.concat((init_state, tf.random.normal((1, NYxi + Nuxi), 0, 1, dtype = dtype)), axis = 0)
init_state = tf.concat((init_state, tf.random.normal((1, NYxi + Nuxi), 0, 1, dtype = dtype)), axis = 0)
init_state = tf.concat((init_state, tf.constant(theta_pickle, dtype = dtype)[np.newaxis,:]), axis = 0)

@tf.function
def run_chain(init_state):
  samples, trace = tfp.mcmc.sample_chain(
      num_results= Nsamples,
      num_burnin_steps= Nburn,
      current_state= init_state,
      kernel= adapt_nuts_dual,
      trace_fn= lambda _,pkr: [pkr.inner_results.target_log_prob,
                                       pkr.inner_results.log_accept_ratio,
                                       pkr.inner_results.step_size]
                                       #pkr.inner_results.has_divergence]
    )
  return samples, trace

init_op = tf.compat.v1.global_variables_initializer()
ts = perf_counter()
print('\n Start HMC Sampling')
#with tf.device('/device:GPU:0'):
with tf.device('/device:CPU:0'):
    samples_op, trace_op = run_chain(init_state)
with tf.Session() as sess:
    sess.run(init_op)
    samples, trace = sess.run([samples_op, trace_op]) #samples (Nsamples, Nchains, theta)
elps_time = perf_counter() - ts
np.save(os.path.abspath(results_path + '/chains.npy'), samples)
print('\n Finish HMC Sampling')
print(f'HMC Sampling Time {elps_time:.3f}')
print(f'HMC Sampling Time {elps_time:.3f}', file = f_rec)
print(f'Number of chains: {Nchains}')
print(f'Number of chains: {Nchains}', file = f_rec)
print(f'Number of burn-ins: {Nburn}')
print(f'Number of burn_ins: {Nburn}', file = f_rec)
print(f'Number of samples per chain: {Nsamples}')
print(f'Number of samples per chain {Nsamples}', file = f_rec)

#%% BPICKLE-HMC post-analysis

sess_post = tf.Session()
target_log_prob = trace[0] # (Nsamples, Nchains)
accept_ratio = np.exp(trace[1]) 
step_size = trace[2]
print(f'Average accept ratio for each chain: {np.mean(accept_ratio, axis = 0)}')
print(f'Average step size for each chain: {np.mean(step_size, axis = 0)}')
print(f'Average accept ratio for each chain: {np.mean(accept_ratio, axis = 0)}', file = f_rec)
print(f'Average step size for each chain: {np.mean(step_size, axis = 0)}', file = f_rec)

#Plot Negative log prob with chains
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
for i in range(Nchains):
    ax.plot(np.arange(Nsamples)[::25], trace[0][::25, i], linestyle = '--', label = f'chain {i + 1}')
ax.set_xlabel('Sample index', fontsize = 15)
ax.set_ylabel('Negative log prob', fontsize = 15)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.set_xlim(0,Nsamples)
ax.legend(fontsize=8, ncol=2, facecolor='white', loc = 'lower left')
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'log_posterior_density.png'))
plt.show()

get_u_pred = lambda uxi: tf.linalg.einsum('ij, j -> i', Psiu, uxi)
get_Y_pred = lambda Yxi: tf.linalg.einsum('ij, j -> i', PsiY, Yxi)
u_pred_ens = np.array([sess_post.run(tf.map_fn(get_u_pred, samples[:,i,NYxi:])) for i in range(samples.shape[1])]) # (Nchains, Nsamples, 1475)
Y_pred_ens = np.array([sess_post.run(tf.map_fn(get_Y_pred, samples[:,i,:NYxi])) for i in range(samples.shape[1])]) # (Nchains, Nsamples, 1475)
u_pred_ens_mean = np.mean(u_pred_ens, axis = 1) # (Nchains,  1475)
u_pred_ens_std = np.std(u_pred_ens, axis = 1) # (Nchains,  1475)
Y_pred_ens_mean = np.mean(Y_pred_ens, axis = 1) # (Nchains,  1475)
Y_pred_ens_std = np.std(Y_pred_ens, axis = 1) # (Nchains,  1475)
u_env = np.logical_and( (u_pred_ens_mean < uref + 2*u_pred_ens_std), (u_pred_ens_mean > uref - 2*u_pred_ens_std) ) # (Nchains,  1475)
Y_env = np.logical_and( (Y_pred_ens_mean < Yref + 2*Y_pred_ens_std), (Y_pred_ens_mean > Yref - 2*Y_pred_ens_std) ) # (Nchains,  1475)

rel_err_hmc_Y = [rl2e(i, Yref) for i in Y_pred_ens_mean]
abs_err_hmc_Y = [infe(i, Yref) for i in Y_pred_ens_mean]
rel_err_hmc_u = [rl2e(i, uref) for i in u_pred_ens_mean]
abs_err_hmc_u = [infe(i, uref) for i in u_pred_ens_mean] 

print(f'Relative L2 error of Y for each chain: {rel_err_hmc_Y}')
print(f'Relative L2 error of u for each chain: {rel_err_hmc_u}')
print(f'Relative L2 error of Y for each chain: {rel_err_hmc_Y}', file = f_rec)
print(f'Relative L2 error of u for each chain: {rel_err_hmc_u}', file = f_rec)

# R-hat and ESS calculation samples-(Nsamples, Nchains, theta)
rhat_op = tfp.mcmc.diagnostic.potential_scale_reduction(samples, independent_chain_ndims=1)
ess_op =  tfp.mcmc.effective_sample_size(samples[:,:,:250], filter_beyond_positive_pairs=True)
rhat = sess_post.run(rhat_op)
ess = sess_post.run(ess_op)
idx_low = np.argmin(rhat)
idx_high = np.argmax(rhat)
    
fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(rhat, bins = NYxi + Nuxi, kde=True, kde_kws = {'gridsize':5000})
g.tick_params(labelsize=16)
g.set_xlabel("$\hat{r}$", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'rhat.png'))
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex='col', sharey='col', dpi = 300)
g = sns.histplot(np.mean(ess, axis = 0), bins = 40, kde=False)
g.tick_params(labelsize=16)
g.set_xlabel("ESS", fontsize=18)
g.set_ylabel("Count", fontsize=18)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'ess.png'))
plt.show()

#Trace plot for parameter with lowest rhat
samples1 = samples[:,:,idx_low]
df1 = pd.DataFrame({'chains': 'Chain 1', 'indice':np.arange(0, samples1.shape[0], 5), 'trace':samples1[::5, 0]})
df2 = pd.DataFrame({'chains': 'Chain 2', 'indice':np.arange(0, samples1.shape[0], 5), 'trace':samples1[::5, 1]})
df3 = pd.DataFrame({'chains': 'Chain 3', 'indice':np.arange(0, samples1.shape[0], 5), 'trace':samples1[::5, 2]})
df4 = pd.DataFrame({'chains': 'Chain 4', 'indice':np.arange(0, samples1.shape[0], 5), 'trace':samples1[::5, 3]})
df5 = pd.DataFrame({'chains': 'Chain 5', 'indice':np.arange(0, samples1.shape[0], 5), 'trace':samples1[::5, 4]})
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
plt.figure(figsize=(4,4))
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, Nsamples), hue='chains', joint_kws={'alpha': 0.6})
g.ax_joint.tick_params(labelsize=18)
g.ax_joint.set_xlabel("Index", fontsize=24)
g.ax_joint.set_ylabel("Trace", fontsize=24)
g.ax_joint.legend(fontsize=12, ncol=3, loc = 'lower center',  facecolor='white')
g.ax_marg_x.remove()
plt.gcf().set_dpi(300)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'trace_low.png'))
plt.show()

#Trace plot for parameter with highest rhat
samples2 = samples[:,:,idx_high]
df1 = pd.DataFrame({'chains': 'Chain 1', 'indice':np.arange(0, samples2.shape[0], 5), 'trace':samples2[::5, 0]})
df2 = pd.DataFrame({'chains': 'Chain 2', 'indice':np.arange(0, samples2.shape[0], 5), 'trace':samples2[::5, 1]})
df3 = pd.DataFrame({'chains': 'Chain 3', 'indice':np.arange(0, samples2.shape[0], 5), 'trace':samples2[::5, 2]})
df4 = pd.DataFrame({'chains': 'Chain 4', 'indice':np.arange(0, samples2.shape[0], 5), 'trace':samples2[::5, 3]})
df5 = pd.DataFrame({'chains': 'Chain 5', 'indice':np.arange(0, samples2.shape[0], 5), 'trace':samples2[::5, 4]})
df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
plt.figure(figsize=(4,4))
g = sns.jointplot(data=df, x='indice', y='trace', xlim=(0, Nsamples), hue='chains', joint_kws={'alpha': 0.6})
g.ax_joint.tick_params(labelsize=18)
g.ax_joint.set_xlabel("Index", fontsize=24)
g.ax_joint.set_ylabel("Trace", fontsize=24)
g.ax_joint.legend(fontsize=12, ncol=3, loc = 'lower center',  facecolor='white')
g.ax_marg_x.remove()
plt.gcf().set_dpi(300)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'trace_high.png'))
plt.show()

#%% Plots of Posterior Field Statistics
plt.rc('text', usetex=False)
plt.rc('image', cmap='plasma')

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = Yref.min() + Yfac
cmax = Yref.max() + Yfac
plot_patch(patches, Y_pred_ens_mean[-1] + Yfac , fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_mean.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
diff =  np.abs(Y_pred_ens_mean[-1]  - Yref)
cmax = diff.max()
cmin = diff.min()
plot_patch(patches, diff, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_diff.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = Y_pred_ens_std[-1].min()
cmax = Y_pred_ens_std[-1].max()
plot_patch(patches, Y_pred_ens_std[-1], fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_std.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
plot_patch(patches, Y_env[-1], fig, ax, points = None, title = None, fontsize = 12, cmin = 0, cmax= 1, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_coverage.png'))

Yxi_ens = samples[..., :NYxi]
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(np.arange(NYxi), 0*np.arange(NYxi), 'b--', alpha = 0.8)
for i in range(Nchains):
    ax.plot(np.arange(NYxi), np.mean(Yxi_ens, axis = 0)[i] - Yxi, marker = 'x', markersize = 5, linestyle = 'none', label= f'chain {i + 1}')
ax.set_xlabel('Index', fontsize=16)
ax.set_ylabel(r'$\Delta\xi$', fontsize=16)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.legend(fontsize=8, ncol=2, facecolor='white', loc = 'lower left')
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'yxi_diff.png'))
plt.show()

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = uref.min()
cmax = uref.max()
plot_patch(patches, u_pred_ens_mean[-1] + Yfac , fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'u_mean.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
diff =  np.abs(u_pred_ens_mean[-1]  - uref)
cmax = diff.max()
cmin = diff.min()
plot_patch(patches, diff, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'u_diff.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = u_pred_ens_std[-1].min()
cmax = u_pred_ens_std[-1].max()
plot_patch(patches, u_pred_ens_std[-1], fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'u_std.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
plot_patch(patches, u_env[-1], fig, ax, points = None, title = None, fontsize = 12, cmin = 0, cmax= 1, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'u_coverage.png'))

uxi_ens = samples[..., :Nuxi]
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(np.arange(Nuxi), 0*np.arange(Nuxi), 'b--', alpha = 0.8)
for i in range(Nchains):
    ax.plot(np.arange(Nuxi), np.mean(uxi_ens, axis = 0)[i] - uxi, marker = 'x', markersize = 5, linestyle = 'none', label= f'chain {i + 1}')
ax.set_xlabel('Index', fontsize=16)
ax.set_ylabel(r'$\Delta\eta$', fontsize=16)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.legend(fontsize=8, ncol=2, facecolor='white', loc = 'lower left')
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'ueta_diff.png'))
plt.show()

for i in range(Nchains):
    rl2e_u = rl2e(u_pred_ens_mean[i, :], uref)
    infe_u = infe(u_pred_ens_mean[i, :], uref)
    lpp_u = lpp(u_pred_ens_mean[i, :], uref, u_pred_ens_std[i, :])
    rl2e_Y = rl2e(Y_pred_ens_mean[i, :], Yref)
    infe_Y = infe(Y_pred_ens_mean[i, :], Yref)
    lpp_Y = lpp(Y_pred_ens_mean[i, :], Yref, Y_pred_ens_std[i, :])
    
    print(f'chain {i}:\n')
    print('u prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_u))
    print('Absolute inf error: {}'.format(infe_u))
    print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_u))
    print('Percentage of coverage:{}\n'.format(np.sum(u_env[i, :])/1475))
    
    print('Y prediction:\n')
    print('Relative RL2 error: {}'.format(rl2e_Y))
    print('Absolute inf error: {}'.format(infe_Y))
    print('Average standard deviation: {}'.format(np.mean(Y_pred_ens_std[i, :])))
    print('log predictive probability: {}'.format(lpp_Y))
    print('Percentage of coverage:{}\n'.format(np.sum(Y_env[i, :])/1475))
    
    print(f'chain {i}:\n', file = f_rec)
    print('u prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_u), file = f_rec)
    print('Absolute inf error: {}'.format(infe_u), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(u_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_u), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(u_env[i, :])/1475), file = f_rec)
    
    print('Y prediction:\n', file = f_rec)
    print('Relative RL2 error: {}'.format(rl2e_Y), file = f_rec)
    print('Absolute inf error: {}'.format(infe_Y), file = f_rec)
    print('Average standard deviation: {}'.format(np.mean(Y_pred_ens_std[i, :])), file = f_rec)
    print('log predictive probability: {}'.format(lpp_Y), file = f_rec)
    print('Percentage of coverage:{}\n'.format(np.sum(Y_env[i, :])/1475), file = f_rec)
    
f_rec.close()