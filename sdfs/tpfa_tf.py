#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:44:19 2023

@author: yifeizong
"""
import paths
import os
import copy
import numpy as np
import numpy.random as npr
from sdfs.geom_mrst import GeomMRST
from sdfs.bc_mrst import BCMRST
from sdfs.darcy import DarcyExp
from sdfs.tpfa import TPFA
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
import tensorflow as tf

# resolution = '1x' 
# resolution_fine = '16x' #splitting cell into finer equal-area cells
# data_path = r'./test/data/'
# geom_filename = data_path + f'geom/geom_{resolution}.mat'
# geom_fine_filename = data_path + f'geom/geom_{resolution_fine}.mat'
# bc_filename = data_path + f'bc/bc_{resolution}.mat'
# conduct_filename = data_path + f'RF1/conduct_log_RF1_{resolution}.mat'
# well_cells_filename = data_path + f'well_cells/well_cells_{resolution}.mat'
# geom = GeomMRST(geom_filename)
# bc = BCMRST(geom, bc_filename)

# tf.keras.backend.set_floatx('float64')
# dtype = tf.float64

class Tpfa_tf(object):
    
    def __init__(self, geom, bc):
        self.geom = geom
        self.bc = bc
        
        self.Nc = int(self.geom.cells.num)
        self.Nc_range = np.arange(self.Nc)
        
        self.Ni = int(self.geom.faces.num_interior)
        self.neighbors = self.geom.cells.to_hf[:2*self.Ni]
        self.rows = np.concatenate((self.neighbors, self.Nc_range))
        self.cols = np.concatenate((np.roll(self.neighbors, self.Ni), self.Nc_range))
        
        self.dtype_np = np.float64
        self.dtype = tf.float64

        c = self.geom.faces.centroids[:, self.geom.faces.to_hf] - self.geom.cells.centroids[:, self.geom.cells.to_hf]
        n = self.geom.faces.normals[:, self.geom.faces.to_hf]
        n[:, self.Ni:2*self.Ni] *= -1
        self.alpha = (np.sum(c * n, axis=0) / np.sum(c ** 2, axis=0)).astype(self.dtype_np)

        self.cell_hfs = np.ascontiguousarray(np.argsort(self.geom.cells.to_hf).reshape(4, -1, order='F'))
        self.cell_ihfs = np.where(self.cell_hfs < 2*self.Ni, self.cell_hfs, -1) #(4, 1475)
        self.cell_neighbors = np.where(self.cell_ihfs >= 0,
                                       self.geom.cells.to_hf[(self.cell_ihfs + self.Ni) % (2*self.Ni)],
                                       -1)
        self.alpha_dirichlet = np.bincount(self.geom.cells.to_hf[2*self.Ni:],
                                           self.alpha[2*self.Ni:] * (self.bc.kind == 'D'),
                                           minlength=self.Nc)
        self.rhs_dirichlet = np.bincount(self.geom.cells.to_hf[2*self.Ni:],
                                         self.alpha[2*self.Ni:] * (self.bc.kind == 'D') * self.bc.val,
                                         minlength=self.Nc)
        self.rhs_neumann = np.bincount(self.geom.cells.to_hf[2*self.Ni:],
                                       (self.bc.kind == 'N') * self.bc.val,
                                       minlength=self.Nc)
    @tf.function
    def ops(self, K):
        idx = tf.cast(tf.slice(self.geom.cells.to_hf, [0], [2 * self.Ni]), dtype = tf.int32) #(5532,)
        self.Thf_interior = tf.cast(tf.slice(self.alpha, [0], [2 * self.Ni]), dtype = self.dtype) * tf.gather(K, idx) #(5532,)
        self.Tgf_interior = (lambda x: tf.reduce_prod(x, axis = 0) / tf.reduce_sum(x, axis=0))(tf.reshape(self.Thf_interior, (2, -1)))
        diag = tf.cast(tf.math.bincount(idx, tf.concat((self.Tgf_interior, self.Tgf_interior), axis = 0 ), minlength=self.Nc), dtype = self.dtype) + tf.constant(self.alpha_dirichlet, dtype = self.dtype) * K
        return         tf.sparse.SparseTensor(
                    tf.transpose(tf.stack([tf.cast(self.rows, dtype = tf.int64), tf.cast(self.cols, dtype = tf.int64)], axis = 0)), tf.concat((-self.Tgf_interior, -self.Tgf_interior, diag), axis = 0), (self.Nc, self.Nc)),\
            tf.constant(self.rhs_dirichlet, dtype = self.dtype) * K +  tf.constant(self.rhs_neumann, dtype = self.dtype)

    @tf.function            
    def residual(self, u, Y):
        A, b = self.ops(tf.math.exp(Y))
        return tf.tensordot(tf.sparse.to_dense(tf.sparse.reorder(A)), u, axes=1) - b 
