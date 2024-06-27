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
from scipy.spatial.distance import pdist, squareform
from time import perf_counter
from matplotlib import pyplot as plt, collections as mc, patches as mpatches, cm
import h5py
import GPy

from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.tpfa import TPFA
from sdfs.tpfa_tf import Tpfa_tf
import ckli.mapest as mapest
import ckli.ckliest_l2reg as ckliest

# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import trange

tf.config.list_physical_devices('GPU')

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
field = 'Y_LD' 
sigma_r = 1
sigma_p = 1
sigma_q = 1
Nsamples = 500
nIter = 20000 #For lr=1e-3, 20000 epoches are enough, for lr=1e-4, may use 80000?
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
    
    NYobs = 10
    Nuobs = 10
    NYxi = 10
    Nuxi = 5
    Nens = 5000
    Yref = np.loadtxt(os.path.abspath(r'./test/data/yobs/Y_smooth.out')) - Yfac
    yobs_filename = data_path + f'yobs/yobs_{NYobs}_{resolution}.npy'
    yobs_fine_filename = data_path + f'yobs/yobs_{NYobs}_{resolution_fine}.npy'
    prob = DarcyExp(TPFA(geom, bc), None) 
    uref = prob.randomize_bc('N', Neumann_sd).solve(Yref) #Given Yref, solve for reference uref (1475,)

    # u observations
    with h5py.File(well_cells_filename, 'r') as f:
        iuobs = f.get('well_cells')[:].ravel() - 1 #observation well index
    
    #iuobs = rs.choice(iuobs, Nuobs, replace  = False) #This should be correct
    iuobs = rs.choice(Nc, (Nuobs,), replace  = False) #This is used in the mauscript
    uobs = uref[iuobs] #extract u obs from reference

save_pth = f'{field}_NY_{NYobs}_Nu_{Nuobs}_Nuxi_{Nuxi}_NYxi_{NYxi}_sigmar_{sigma_r}_SVGD_batch_nIter_{nIter}_Nsamples_{Nsamples}_h=-1'
#save_pth = 'test'
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
    if field == 'Y_LD':
        iYobs[0] = iYobs[3]
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
print(f"GPR\tRelative error of Y: {rel_err_gpr:.5f}")
print(f"GPR\tInfinity error of Y: {abs_err_gpr:.5f}") 

print(f"GPR: {timings:.3f} s", file = f_rec)
print(f'GPR variance: {variance:.3f}', file = f_rec)
print(f'GPR lengthscale: {lengthscale:.3f}', file = f_rec)
print(f"GPR\tRelative error of Y: {rel_err_gpr:.5f}", file = f_rec)
print(f"GPR\tInfinity error of Y: {abs_err_gpr:.5f}", file = f_rec) 
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
np.savetxt(os.path.join(results_path , 'theta_pickle.out'), theta_pickle)
upickle = um + Psiu @ uxi
Ypickle = Ym + PsiY @ Yxi
rel_err_pickle = rl2e(Ypickle, Yref) 
abs_err_pickle = infe(Ypickle, Yref) 

print("############# PICKLE Results #################")
print(f"PICKLE: {timings:.3f} s")
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.5f}")
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.5f}")
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.5f}", file = f_rec)
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.5f}\n", file = f_rec)
print("#############################################\n")
    

#%% Bayesian PICKLE - HMC

tf.keras.backend.set_floatx('float64')
dtype = tf.float64
tf.random.set_seed(8888)

tpfa = Tpfa_tf(geom, bc)
sigma_r_tf = tf.constant(sigma_r, dtype = dtype) 
Psiu_tf = tf.constant(Psiu, dtype = dtype)
PsiY_tf = tf.constant(PsiY, dtype = dtype) 
um_tf = tf.constant(um, dtype = dtype) 
Ym_tf = tf.constant(Ym, dtype = dtype) 
loc = tf.constant(0., dtype = dtype)
scale = tf.constant(1., dtype = dtype)
theta = tf.Variable(
    initial_value = tf.random.normal((Nsamples, NYxi + Nuxi), dtype = dtype),
    trainable=True,
    name = 'theta',
    dtype = dtype
    )
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3,
                                    beta_1 = 0.9, beta_2 = 0.999,
                                    epsilon = 1e-07, name = 'Adam'
                                    )

@tf.function
def kl_pred(theta): #correct batch implementation
    def wrapper_fn(inputs):
        u_, Y_= inputs 
        res_ = tpfa.residual(u_, Y_)
        return (res_, res_)
    u = um_tf + tf.linalg.einsum('ij,nj->ni', Psiu_tf, theta[:,NYxi:])
    Y = Ym_tf + tf.linalg.einsum('ij,nj->ni', PsiY_tf, theta[:,:NYxi])
    res = tf.map_fn(wrapper_fn, (u,Y))
    return res[0]

@tf.function
def target_log_prob_fn(theta): #correct batch implementation
    prior = tf.reduce_sum(-tf.math.log(scale) - 
                          tf.math.log(2*tf.constant(np.pi, dtype = dtype))/2 -(theta - loc)**2/(2*scale**2), 
                          axis = -1
                          )
    r_likelihood = tf.reduce_sum(-tf.math.log(sigma_r_tf) - 
                                  tf.math.log(2*tf.constant(np.pi, dtype = dtype))/2 -(kl_pred(theta))**2/(2*sigma_r_tf**2), 
                                  axis = -1
                                  )
    return prior + r_likelihood


@tf.function
def grad_log_prob_fn(theta):  #correct, with negating PICKLE loss grad
    #return [grad_xi, grad_eta]
    return tfp.math.value_and_gradient(target_log_prob_fn, theta,
                            auto_unpack_single_arg=False)[1]

@tf.function
def svgd_kernel(theta, h): 
    # TF implementation of the svgd_kernel, only differences upto floating point err
        
    theta_expanded = tf.expand_dims(theta, 1)
    pairwise_dists = tf.reduce_sum(tf.square(theta - theta_expanded), 2)
    
    # Compute the RBF kernel
    Kxy = tf.exp(-pairwise_dists / (2 * h ** 2))
    
    # Compute the derivative of the kernel
    sum_kxy = tf.reduce_sum(Kxy, axis=1, keepdims=True)
    dxkxy = -tf.matmul(Kxy, theta)
    dxkxy += theta * sum_kxy
    dxkxy /= h ** 2
    
    return Kxy, dxkxy

#Note in SVGD paper, it is theta_{n+1}= theta_n + alpha*grad. but in sgd it's minus
@tf.function
def svgd_update(theta, h):
    lnpgrad = grad_log_prob_fn(theta) #(Nsamples, Ntheta)
    kxy, dxkxy = svgd_kernel(theta, h)  
    phi = -(tf.linalg.einsum('ij, jk -> ik', kxy, lnpgrad) + dxkxy) / theta.shape[0]  #(Nsamples, Ntheta)
    optimizer.apply_gradients(zip([phi], [theta]))
    return 

get_Y_pred_tf = tf.function(lambda Yxi: tf.linalg.einsum('ij, j -> i', PsiY_tf, Yxi) + Ym_tf)

def svgd_train(theta, h, n_iter = 3000, num_print = 50):
    
    pbar = trange(nIter)
    
    for it in pbar:

        svgd_update(theta, h)
        
        if it % num_print == 0:
            log_prob_mean = tf.reduce_mean(target_log_prob_fn(theta))  
            pbar.set_postfix({'Log prob': log_prob_mean.numpy()})

        
    return theta

ts = perf_counter()
print('\n Start SVGD Sampling')
#samples= svgd_train(theta, Nsamples, nIter, bandwidth = -1) #(Nsamples, theta)
samples = svgd_train(theta, h = -1, n_iter = nIter, num_print = 50)
elps_time = perf_counter() - ts
print('\n Finish SVGD Sampling')
print(f'SVGD Sampling Time {elps_time:.3f}')
print(f'SVGD Sampling Time {elps_time:.3f}', file = f_rec)

# init_op = tf.compat.v1.global_variables_initializer()
# ts = perf_counter()
# print('\n Start SVGD Sampling')
# with tf.device('/device:GPU:0'):
#     samples_op = svgd_train(theta,  nIter)
# with tf.Session() as sess:
#     sess.run(init_op)
#     samples = sess.run([samples_op])
# elps_time = perf_counter() - ts
# print('\n Finish SVGD Sampling')
# print(f'SVGD Sampling Time {elps_time:.3f}')
# print(f'SVGD Sampling Time {elps_time:.3f}', file = f_rec)


#%% BPICKLE-SVGD post-analysis

Yxi_ens, uxi_ens = samples[:,:NYxi], samples[:,NYxi:] 
Yxi_m = np.mean(Yxi_ens, axis = 0) 
uxi_m = np.mean(uxi_ens, axis = 0) 

get_u_pred = lambda uxi: np.einsum('ij, j -> i', Psiu, uxi)
get_Y_pred = lambda Yxi: np.einsum('ij, j -> i', PsiY, Yxi)

upred_ens = np.array([um + get_u_pred(x) for x in uxi_ens])
Ypred_ens = np.array([Ym + get_Y_pred(x) for x in Yxi_ens])

Ypred_ens_mean = np.mean(Ypred_ens, axis = 0)
Ypred_ens_std = np.std(Ypred_ens, axis = 0)
Ypred_env = np.logical_and( (Ypred_ens_mean < Yref + 2*Ypred_ens_std), (Ypred_ens_mean > Yref - 2*Ypred_ens_std) )
upred_ens_mean = np.mean(upred_ens, axis = 0)
upred_ens_std = np.std(upred_ens, axis = 0)
upred_env = np.logical_and( (upred_ens_mean < uref + 2*upred_ens_std), (upred_ens_mean > uref - 2*upred_ens_std) )

plot_all(patches, Yref, Ypred_ens_mean, Ypred_ens_std, Ypred_env, fontsize = 15, title = None, savefig = True)
#plot_each(patches, Yref, Ypred_ens_mean, Ypred_ens_std, Ypred_env, fontsize = 15, title = None, savefig = True)
#plot_all(patches, uref, upred_ens_mean, upred_ens_std, upred_env, fontsize = 15, title = None, savefig = True)
#plot_each(patches, uref, upred_ens_mean, upred_ens_std, upred_env, fontsize = 15, title = None, savefig = True)

np.savetxt(os.path.join(results_path , 'Yxi_ens.out'), Yxi_ens)
np.savetxt(os.path.join(results_path , 'Ypred_ens.out'), Ypred_ens)
np.savetxt(os.path.join(results_path , 'uxi_ens.out'), uxi_ens)
np.savetxt(os.path.join(results_path , 'upred_ens.out'), upred_ens)

rel_err_svgd_Y = rl2e(Ypred_ens_mean, Yref)
abs_err_svgd_Y = infe(Ypred_ens_mean, Yref)
rel_err_svgd_u = rl2e(upred_ens_mean, uref)
abs_err_svgd_u = infe(upred_ens_mean, uref)

print("############# SVGD Results #################")
print(f'svgd Sampling Time {elps_time:.3f}')
print(f'svgd Sampling Time {elps_time:.3f}', file = f_rec)

print(f"svgd\trelative L2 error Y: {rel_err_svgd_Y:.5f}")
print(f"svgd\tabsolute infinity error Y: {abs_err_svgd_Y:.5f}")
print(f"svgd\trelative L2 error u: {rel_err_svgd_u:.5f}") 
print(f"svgd\tabsolute infinity error u: {abs_err_svgd_u:.5f}")
print(f"svgd\trelative L2 error Y: {rel_err_svgd_Y:.5f}", file = f_rec)
print(f"svgd\tabsolute infinity error Y: {abs_err_svgd_Y:.5f}", file = f_rec)
print(f"svgd\trelative L2 error u: {rel_err_svgd_u:.5f}", file = f_rec)
print(f"svgd\tabsolute infinity error u: {abs_err_svgd_u:.5f}", file = f_rec)

print(f'Average standard deviation: {np.mean(Ypred_ens_std):.5f}')
print(f'log predictive probability: {lpp(Ypred_ens_mean, Yref, Ypred_ens_std):.5f}')
print(f'Percentage of coverage:{np.sum(Ypred_env)/1475}')
print(f'Average standard deviation: {np.mean(Ypred_ens_std):.3f}', file = f_rec)
print(f'log predictive probability: {lpp(Ypred_ens_mean, Yref, Ypred_ens_std):.3f}', file = f_rec)
print(f'Percentage of coverage:{np.sum(Ypred_env)/1475}', file = f_rec)
print("#############################################\n")

#%% Plots of Posterior Field Statistics
plt.rc('text', usetex=False)
plt.rc('image', cmap='plasma')

# t = np.arange(0, nIter, 50)
# fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
# ax = fig.add_subplot()
# ax.plot(t, np.array(rl2e_log), color='blue', label='rl2e')
# ax.set_yscale('log')
# ax.set_xlabel('Epochs', fontsize = 16)
# ax.legend(loc='upper right', fontsize = 14)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'rl2e_rec.png'))
# plt.show()

# fig = plt.figure(constrained_layout=False, figsize=(4, 4), dpi = 300)
# ax = fig.add_subplot()
# ax.plot(t, np.array(lpp_log), color='blue', label='lpp')
# ax.set_yscale('log')
# ax.set_xlabel('Epochs', fontsize = 16)
# ax.legend(loc='upper right', fontsize = 14)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'lpp_rec.png'))
# plt.show()

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = Yref.min() + Yfac
cmax = Yref.max() + Yfac
plot_patch(patches, Ypred_ens_mean + Yfac , fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_mean.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
diff =  np.abs(Ypred_ens_mean  - Yref)
cmax = diff.max()
cmin = diff.min()
plot_patch(patches, diff, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_diff.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
cmin = Ypred_ens_std.min()
cmax = Ypred_ens_std.max()
plot_patch(patches, Ypred_ens_std, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_std.png'))

fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
plot_patch(patches, Ypred_env, fig, ax, points = None, title = None, fontsize = 12, cmin = 0, cmax= 1, cb = True)
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'y_coverage.png'))

Yxi_ens = samples[..., :NYxi]
fig, ax = plt.subplots(dpi = 300, figsize = (4,4))
ax.plot(np.arange(NYxi), 0*np.arange(NYxi), 'b--', alpha = 0.8)
ax.plot(np.arange(NYxi), np.mean(Yxi_ens, axis = 0) - Yxi, marker = 'x', markersize = 5, linestyle = 'none', label= 'SVGD')
ax.set_xlabel('Index', fontsize=16)
ax.set_ylabel(r'$\Delta\xi$', fontsize=16)
ax.tick_params(axis='both', which = 'major', labelsize=12)
ax.legend(fontsize=8, ncol=2, facecolor='white', loc = 'lower left')
fig.tight_layout()
fig.savefig(os.path.join(fig_path, 'yxi_diff.png'))
plt.show()

# fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
# cmin = uref.min()
# cmax = uref.max()
# plot_patch(patches, upred_ens_mean , fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'u_mean.png'))

# fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
# diff =  np.abs(upred_ens_mean  - uref)
# cmax = diff.max()
# cmin = diff.min()
# plot_patch(patches, diff, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'u_diff.png'))

# fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
# cmin = upred_ens_std.min()
# cmax = upred_ens_std.max()
# plot_patch(patches, upred_ens_std, fig, ax, points = None, title = None, fontsize = 12, cmin = cmin, cmax= cmax, cb = True)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'u_std.png'))

# fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
# plot_patch(patches, upred_env, fig, ax, points = None, title = None, fontsize = 12, cmin = 0, cmax= 1, cb = True)
# fig.tight_layout()
# fig.savefig(os.path.join(fig_path, 'u_coverage.png'))
    
f_rec.close()

# def svgd_kernel2(theta, h = -1):
#     # Note here h is sigma, the std to the rbf kernel
#     # This is the same as the original implementation, but more concise
    
#     sq_dist = pdist(theta)
#     pairwise_dists = squareform(sq_dist)**2
#     if h < 0: # if h < 0, using median trick
#         h = np.median(pairwise_dists)  
#         h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

#     # compute the rbf kernel and derivative
#     Kxy = np.exp(-pairwise_dists / (2*h**2) ) #(Nsamples, Nsamples)
#     dxkxy = -np.matmul(Kxy, theta) 
#     dxkxy = dxkxy + np.einsum('ij, i -> ij', theta, np.sum(Kxy, axis=1))
#     dxkxy = dxkxy / (h**2) #(Nsamples, Ntheta)

#     return (Kxy, dxkxy)


# import jax.numpy as jnp
# from jax import vmap

# def median_trick_h(theta): #same
#     '''
#     The scipy one seems even faster and memory efficient
    
#     '''
#     sq_dist = pdist(theta)
#     pairwise_dists = squareform(sq_dist)**2
#     h = np.median(pairwise_dists)  
#     h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
#     return h

# def rbf_kernel(theta1, theta2, h): #same
#     '''
#     Evaluate the rbf kernel k(x, x') = exp(-|x - x'|^2/(2h^2))
#     input: theta1, theta2 are 1d array of parameters, 
#             h is correlation length
#     output: a scalar value of kernel evaluation 
#     '''
#     # here theta1 and theta2 are 1d-array of parameters
#     return jnp.exp(-((theta1 - theta2)**2).sum(axis=-1) / (2 * h**2))

# def compute_kernel_matrix(theta, h): #same
#     return vmap(vmap(lambda x, y: rbf_kernel(x, y, h), in_axes=(None, 0)), in_axes=(0, None))(theta, theta)

# def kernel_and_grad(theta, h): #same
#     '''
#     input theta: (Nsamples, Nparams)
#             h is correlation length
#     output: K: #(Nsamples, Nsamples)
#             grad_K: #(Nsamples, Nparams)
#     '''
#     K = compute_kernel_matrix(theta, h) #(Nsamples, Nsamples)
#     sum_k = jnp.sum(K, axis=1, keepdims=True)
#     grad_K = -jnp.matmul(K, theta)
#     grad_K += theta * sum_k
#     grad_K /= h ** 2
#     return (K, grad_K)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate=1e-3,
#             decay_steps= 1000,
#             decay_rate= 0.9
#             )

# def svgd_train(theta, h, n_iter = 3000, num_print = 50):
    
#     pbar = trange(nIter)
#     rl2e_log = []
#     lpp_log = []
#     log_prob_log = []
    
#     for it in pbar:

#         svgd_update(theta, h)
        
#         if it % num_print == 0:
#             log_prob_mean = tf.reduce_mean(target_log_prob_fn(theta))
            
#             Ypred_ens = tf.map_fn(get_Y_pred_tf, theta[:, :NYxi])
#             Ypred_ens_mean = np.mean(Ypred_ens, axis = 0)
#             Ypred_ens_std = np.std(Ypred_ens, axis = 0)
#             rl2e_y = rl2e(Ypred_ens_mean, Yref)
#             lpp_y = lpp(Ypred_ens_mean, Yref, Ypred_ens_std)
            
#             pbar.set_postfix({'Log prob': log_prob_mean.numpy(),
#                               'rl2e': rl2e_y,
#                               'lpp_y': lpp_y})
            
#             rl2e_log.append(rl2e_y)
#             lpp_log.append(lpp_y)
#             log_prob_log.append(log_prob_mean)
        
#     return theta, rl2e_log, lpp_log, log_prob_log