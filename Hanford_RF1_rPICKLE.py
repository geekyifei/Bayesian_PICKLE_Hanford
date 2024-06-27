# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 23:26:11 2022

@author: yifei
"""

#%% import dependencies

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

from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.tpfa import TPFA
from sdfs.tpfa_tf import Tpfa_tf
import ckli.mapest as mapest
import ckli.ckliest_l2reg_random as ckliest_l2
import ckli.ckliest_h1reg_random as ckliest_h1

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
        fig.savefig(os.path.join(fig_path , 'Y_statistics_all_rPICKLE.png'))

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
        fig.savefig(os.path.join(fig_path , 'Y_mean_rPICKLE.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    diff = np.abs(Ypred - Yref)
    plot_patch(patches, diff, fig, ax, points, None, fontsize, cmin = diff.min(), cmax = diff.max(), cb = True)
    fig.tight_layout()
    if title:
        ax.set_title('Y diff')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_diff_rPICKLE.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    plot_patch(patches, Y_std, fig, ax, points, None, fontsize, cmin = Y_std.min(), cmax = Y_std.max(), cb = True)
    fig.tight_layout()
    if title:
        ax.set_title('Y pred std ')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_std_rPICKLE.png'))
    
    fig, ax = plt.subplots(dpi = 300, figsize=(4, 4))
    plot_patch(patches, Y_env, fig, ax, points, None, fontsize, 0, 1, True)
    fig.tight_layout()
    if title:
        ax.set_title('Y coverage')
    if savefig:
        fig.savefig(os.path.join(fig_path , 'Y_coverage_rPICKLE.png'))


#%% Control Parameters

seed = 0 #random seed
res_fac = 1
field = 'Y_LD' 
sigma_r = 0.01
sigma_p = 1
sigma_q = 1
metropolis  = False # if to perform Metropolization
reg_type = 'h1' # type of regularization
if reg_type == 'h1':
    import ckli.ckliest_h1reg as ckliest
else:
    import ckli.ckliest_l2reg as ckliest
Nsamples = 10000

std_dev = 1.0 #std for the covariance kernel
cor_len = 0.1 #correlation length for the covariance kernel
Neumann_sd = 0
data_path = r'./test/data/'
resolution = '1x' 
resolution_fine = '16x' #splitting cell into finer equal-area cells
geom_filename = data_path + f'geom/geom_{resolution}.mat'
geom_fine_filename = data_path + f'geom/geom_{resolution_fine}.mat'
bc_filename = data_path + f'bc/bc_{resolution}.mat'
conduct_filename = data_path + f'RF1/conduct_log_RF1_{resolution}.mat'
well_cells_filename = data_path + f'well_cells/well_cells_{resolution}.mat'

#%% Y and u observations

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
    
    iuobs = rs.choice(iuobs, Nuobs, replace  = False)
    uobs = uref[iuobs] #extract u obs from reference

save_pth = f'./{field}_NY_{NYobs}_Nu_{Nuobs}_Nuxi_{Nuxi}_NYxi_{NYxi}_sigmar_{sigma_r}_rPICKLE_{reg_type}_metropolis_{metropolis}_Nsamples_{Nsamples}'
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
    #iYobs = np.concatenate((iYobs, 1055*np.ones((10,1), dtype = np.int64)), axis = 1)
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
Lreg = mapest.compute_Lreg(geom) #2766x1475 sparse matrix
beta_ckli = 0

if reg_type == 'h1':
    res = ckliest.LeastSqRes(NYxi, Ym, PsiY, Nuxi, um, Psiu, prob, sigma_r, sigma_p, sigma_q, res_fac, Lreg, iuobs, uobs, iYobs, Yobs, beta_ckli, ssv=ssv)
else:
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
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.5f}")
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.5f}")
print(f"PICKLE\trelative L2 error: {rel_err_pickle:.5f}", file = f_rec)
print(f"PICKLE\tabsolute infinity error: {abs_err_pickle:.5f}", file = f_rec)
print("#############################################\n")
assert(False)
#%% randomized PICKLE

#tf.keras.backend.set_floatx('float64')
dtype = tf.float64
tpfa = Tpfa_tf(geom, bc)
ssv = None if Neumann_sd == 0 else np.delete(np.arange(Nc), np.unique(geom.cells.to_hf[2*geom.faces.num_interior:][bc.kind == 'N']))
Psiu_tf = tf.constant(Psiu, dtype = dtype)
PsiY_tf = tf.constant(PsiY, dtype = dtype) 
um_tf = tf.constant(um, dtype = dtype) 
Ym_tf = tf.constant(Ym, dtype = dtype) 

@tf.function
def kl_pred(theta):
    u = um_tf + tf.linalg.matvec(Psiu_tf, theta[NYxi:])
    Y = Ym_tf + tf.linalg.matvec(PsiY_tf, theta[:NYxi])
    res = tpfa.residual(u, Y)
    return res

if reg_type == 'l2':
    alpha_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(NYxi), dtype = dtype),
                                     scale= tf.convert_to_tensor(np.ones(NYxi), dtype = dtype)), reinterpreted_batch_ndims = 1)
    beta_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(Nuxi), dtype = dtype),
                                     scale= tf.convert_to_tensor(np.ones(Nuxi), dtype = dtype)), reinterpreted_batch_ndims = 1)
    omega_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(Nc), dtype = dtype),
                                    scale= tf.convert_to_tensor(sigma_r*np.ones(Nc), dtype = dtype)), reinterpreted_batch_ndims = 1)
    alpha = alpha_dist.sample()
    beta = beta_dist.sample()
    omega = omega_dist.sample()
    
elif reg_type =='h1':
    Lreg = mapest.compute_Lreg(geom) #2766x1475 sparse matrix
    Nreg = Lreg.shape[0]
    alpha_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(Nreg), dtype = dtype),
                                     scale= tf.convert_to_tensor(np.ones(Nreg), dtype = dtype)), reinterpreted_batch_ndims = 1)
    beta_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(Nreg), dtype = dtype),
                                     scale= tf.convert_to_tensor(np.ones(Nreg), dtype = dtype)), reinterpreted_batch_ndims = 1)
    omega_dist = tfd.Independent(tfd.Normal(loc= tf.convert_to_tensor(np.zeros(Nc), dtype = dtype),
                                    scale= tf.convert_to_tensor(sigma_r*np.ones(Nc), dtype = dtype)), reinterpreted_batch_ndims = 1)
    alpha = alpha_dist.sample()
    beta = beta_dist.sample()
    omega = omega_dist.sample()

def onestep_rpickle_l2(sess, alpha, beta, omega, init):
    
    a = sess.run(alpha)
    b = sess.run(beta)
    w = sess.run(omega)
    
    res = ckliest_l2.LeastSqRes(NYxi, Ym, PsiY, Nuxi, um, Psiu, prob, sigma_r, sigma_p, sigma_q, res_fac, iuobs, uobs, iYobs, Yobs, beta_ckli, a, b, w, ssv=ssv)
    ts = perf_counter()
    sol = spo.least_squares(res.val, init, jac=res.jac, method='trf', verbose=0)
    timings = perf_counter() - ts

    uxi = sol.x[:Nuxi] 
    Yxi = sol.x[Nuxi:] 
    
    print(f"PICKLE: {timings:3f} s")
    
    return uxi, Yxi, a, b, w

def onestep_rpickle_h1(sess, alpha, beta, omega, init):
    
    a = sess.run(alpha)
    b = sess.run(beta)
    w = sess.run(omega)
    
    ts = perf_counter()
    res = ckliest_h1.LeastSqRes(NYxi, Ym, PsiY, Nuxi, um, Psiu, prob, sigma_r, sigma_p, sigma_q, res_fac, Lreg, iuobs, uobs, iYobs, Yobs, beta_ckli, a, b, w, ssv=ssv)
    sol = spo.least_squares(res.val, init, jac=res.jac, method='trf', verbose=0) 
    timings = perf_counter() - ts

    uxi = sol.x[:Nuxi] 
    Yxi = sol.x[Nuxi:] 
    
    print(f"PICKLE: {timings:3f} s")
    
    return uxi, Yxi, a, b, w

def compute_acceptance_ratio(theta_old, theta_new, w_old, w_new, sess):
    
    def det(jac, hess):
        return np.linalg.det(sigma_r**2*np.eye(NYxi + Nuxi) +  np.einsum('ik,kj->ij', jac.T, jac))

    u_old = um + np.einsum('ij, j -> i', Psiu, theta_old[NYxi:])
    u_new = um + np.einsum('ij, j -> i', Psiu, theta_new[NYxi:])
    Y_old = Ym + np.einsum('ij, j -> i', PsiY, theta_old[:NYxi])
    Y_new = Ym + np.einsum('ij, j -> i', PsiY, theta_new[:NYxi])
    
    drdxi_old  = prob.residual_sens_Y(u_old, Y_old).T.dot(PsiY)
    drdxi_new  = prob.residual_sens_Y(u_new, Y_new).T.dot(PsiY)
    drdeta_old  = prob.residual_sens_u(u_old, Y_old).dot(Psiu)
    drdeta_new  = prob.residual_sens_u(u_new, Y_new).dot(Psiu)
    
    jac_old = np.hstack((drdxi_old, drdeta_old))
    jac_new = np.hstack((drdxi_new, drdeta_new))
    
    hess_new = sess.run(hess_op, feed_dict={theta_op: theta_new, w_op: w_new})
    hess_old = sess.run(hess_op, feed_dict={theta_op: theta_old, w_op: w_old})
    
    det_old = det(jac_old, hess_old)
    det_new = det(jac_new, hess_new)
    ratio = np.sqrt(np.abs(det_new))/np.sqrt(np.abs(det_old))
    
    return ratio

def run_rpickle(alpha, beta, omega, Nsamples, metropolis, reg_type):
    
    sess = tf.Session()
    Yxi_ens = np.zeros((Nsamples, NYxi))
    uxi_ens = np.zeros((Nsamples, Nuxi))
    accept_ratio = np.zeros((Nsamples, 1))
    is_accepted = np.zeros((Nsamples, 1))
    
    if reg_type == 'l2':
        uxi_, Yxi_, a_, b_, w_ = onestep_rpickle_l2(sess, alpha, beta, omega, np.zeros(Nuxi + NYxi))
        init = np.concatenate((uxi_, Yxi_))
        theta_ = np.concatenate((Yxi_, uxi_), axis = 0) 
        Yxi_ens[0, :] = Yxi_ 
        uxi_ens[0, :] = uxi_
        if metropolis == True:
            accept_ratio[0, :] = 1.
            is_accepted[0, :] = 1.
        print(f'{0}-th iteration finished')
        
        for i in range(1, Nsamples):
            uxi, Yxi, a, b, w = onestep_rpickle_l2(sess, alpha, beta, omega, init)
            theta = np.concatenate((Yxi, uxi), axis = 0)
            if metropolis == True:
                ratio = compute_acceptance_ratio(theta_, theta, w_, w, sess)
                ratio = np.min((ratio, 1))
                accept_ratio[i, :] = ratio
                u = np.random.uniform(low=0.0, high=1.0, size=())
                if u <= ratio:
                    Yxi_ens[i, :] = Yxi
                    uxi_ens[i, :] = uxi
                    theta_ = theta
                    Yxi_ = Yxi
                    uxi_ = uxi
                    is_accepted[i, :] = 1.
                    print('sample accepted')
                else:
                    Yxi_ens[i, :] = Yxi_
                    uxi_ens[i, :] = uxi_
                    is_accepted[i, :] = 0.
                    print('sample rejected')
                print(f'{i+1}-th iteration finished')
            else:
                Yxi_ens[i, :] = Yxi
                uxi_ens[i, :] = uxi
                init = np.concatenate((uxi, Yxi))
                print(f'{i+1}-th iteration finished')
        
        if metropolis == True:
            return Yxi_ens, uxi_ens, accept_ratio, is_accepted
        else:
            return Yxi_ens, uxi_ens
        
    elif reg_type == 'h1':
        uxi_, Yxi_, a_, b_, w_ = onestep_rpickle_h1(sess, alpha, beta, omega, np.zeros(Nuxi + NYxi))
        init = np.concatenate((uxi_, Yxi_))
        theta_ = np.concatenate((Yxi_, uxi_), axis = 0) 
        Yxi_ens[0, :] = Yxi_ 
        uxi_ens[0, :] = uxi_
        if metropolis == True:
            accept_ratio[0, :] = 1.
            is_accepted[0, :] = 1.
        print(f'{0}-th iteration finished')
        
        for i in range(1, Nsamples):
            uxi, Yxi, a, b, w = onestep_rpickle_h1(sess, alpha, beta, omega, init)
            theta = np.concatenate((Yxi, uxi), axis = 0)
            if metropolis == True:
                ratio = compute_acceptance_ratio(theta_, theta, w_, w, sess)
                ratio = np.min((ratio, 1))
                accept_ratio[i, :] = ratio
                u = np.random.uniform(low=0.0, high=1.0, size=())
                if u <= ratio:
                    Yxi_ens[i, :] = Yxi
                    uxi_ens[i, :] = uxi
                    theta_ = theta
                    Yxi_ = Yxi
                    uxi_ = uxi
                    is_accepted[i, :] = 1.
                    print('sample accepted')
                else:
                    Yxi_ens[i, :] = Yxi_
                    uxi_ens[i, :] = uxi_
                    is_accepted[i, :] = 0.
                    print('sample rejected')
                print(f'{i+1}-th iteration finished')
            else:
                Yxi_ens[i, :] = Yxi
                uxi_ens[i, :] = uxi
                init = np.concatenate((uxi, Yxi))
                print(f'{i+1}-th iteration finished')
        
        if metropolis == True:
            return Yxi_ens, uxi_ens, accept_ratio, is_accepted
        else:
            return Yxi_ens, uxi_ens

ts = perf_counter()
if metropolis == True:
    theta_op = tf.placeholder(dtype, [NYxi + Nuxi])
    w_op = tf.placeholder(dtype, [Nc])
    res_op = kl_pred(theta_op)
    delta_op = res_op - w_op
    hess_op = tf.hessians(tf.einsum('i,i->', res_op , delta_op), theta_op)[0]   
    Yxi_ens, uxi_ens, accept_ratio, is_accepted =  run_rpickle(alpha, beta, omega, Nsamples, metropolis, reg_type)
else:
    Yxi_ens, uxi_ens =  run_rpickle(alpha, beta, omega, Nsamples, metropolis, reg_type)
elps_time = perf_counter() - ts

#%% Post-analysis

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
plot_each(patches, Yref, Ypred_ens_mean, Ypred_ens_std, Ypred_env, fontsize = 15, title = None, savefig = True)
plot_all(patches, uref, upred_ens_mean, upred_ens_std, upred_env, fontsize = 15, title = None, savefig = True)
plot_each(patches, uref, upred_ens_mean, upred_ens_std, upred_env, fontsize = 15, title = None, savefig = True)

np.savetxt(os.path.join(results_path , 'Yxi_ens.out'), Yxi_ens)
np.savetxt(os.path.join(results_path , 'Ypred_ens.out'), Ypred_ens)
np.savetxt(os.path.join(results_path , 'uxi_ens.out'), uxi_ens)
np.savetxt(os.path.join(results_path , 'upred_ens.out'), upred_ens)

rel_err_rpickle_Y = rl2e(Ypred_ens_mean, Yref)
abs_err_rpickle_Y = infe(Ypred_ens_mean, Yref)
rel_err_rpickle_u = rl2e(upred_ens_mean, uref)
abs_err_rpickle_u = infe(upred_ens_mean, uref)

print("############# rPICKLE Results #################")
print(f'rPICKLE Sampling Time {elps_time:.3f}')
print(f'rPICKLE Sampling Time {elps_time:.3f}', file = f_rec)

print(f"rPICKLE\trelative L2 error Y: {rel_err_rpickle_Y:.3f}")
print(f"rPICKLE\tabsolute infinity error Y: {abs_err_rpickle_Y:.3f}")
print(f"rPICKLE\trelative L2 error u: {rel_err_rpickle_u:.3f}") 
print(f"rPICKLE\tabsolute infinity error u: {abs_err_rpickle_u:.3f}")
print(f"rPICKLE\trelative L2 error Y: {rel_err_rpickle_Y:.3f}", file = f_rec)
print(f"rPICKLE\tabsolute infinity error Y: {abs_err_rpickle_Y:.3f}", file = f_rec)
print(f"rPICKLE\trelative L2 error u: {rel_err_rpickle_u:.3f}", file = f_rec)
print(f"rPICKLE\tabsolute infinity error u: {abs_err_rpickle_u:.3f}", file = f_rec)

print(f'Average standard deviation: {np.mean(Ypred_ens_std):.3f}')
print(f'log predictive probability: {lpp(Ypred_ens_mean, Yref, Ypred_ens_std):.3f}')
print(f'Percentage of coverage:{np.sum(Ypred_env)/1475}')
print(f'Average standard deviation: {np.mean(Ypred_ens_std):.3f}', file = f_rec)
print(f'log predictive probability: {lpp(Ypred_ens_mean, Yref, Ypred_ens_std):.3f}', file = f_rec)
print(f'Percentage of coverage:{np.sum(Ypred_env)/1475}', file = f_rec)
print("#############################################\n")

f_rec.close()