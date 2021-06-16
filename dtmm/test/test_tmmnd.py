"""Tests for tmm2d, tmm3d and solver"""

import unittest
import numpy as np
from dtmm.tmm2d import f_iso2d, layer_mat2d, stack_mat2d, system_mat2d, \
     reflection_mat2d, reflect2d, list_modes, unlist_modes
from dtmm.tmm3d import f_iso3d, layer_mat3d, stack_mat3d, system_mat3d, \
     reflection_mat3d, reflect3d, upscale2, mode_masks, upscale1
from dtmm.field import field2modes, field2modes1, modes2field, modes2field1

from dtmm.wave import eigenmask1, eigenmask, k0

import dtmm.conf
from dtmm.conf import MKL_FFT_INSTALLED
from dtmm.solver import MatrixBlockSolver3D

rtol, atol = (1e-05,1e-08) if dtmm.conf.PRECISION == "double" else (1e-3,1e-4)

wavelengths = [500,550]
pixelsize = 100
n = 1.5

ks = k0(wavelengths, pixelsize)

SWAP_AXES = False

if SWAP_AXES:
    field = np.random.randn(2,4,16,1) + 0j
else:    
    field = np.random.randn(2,4,1,16) + 0j
    
field_3d = np.random.randn(2,4,16,16) + 0j

emask3, modes3 = field2modes(field,ks)
if SWAP_AXES:
    emask2, modes2 = field2modes1(field[:,:,:,0],ks)
else:
    emask2, modes2 = field2modes1(field[:,:,0,:],ks)

grouped_modes_in2 = list_modes(modes2)
grouped_modes_in3 = list_modes(modes3)

emask_2d = eigenmask1(16,ks)
emask_3d = eigenmask(field.shape[-2:],ks)

d = (1,1,1,1)

eps_shape = (4,) + field.shape[-2:] + (3,)

epsv_3d = np.random.rand(*eps_shape) + 2
epsv_3d[...,1] = epsv_3d[...,0]
epsa_3d = np.random.rand(*eps_shape)

if SWAP_AXES:
    epsv_2d = epsv_3d[:,:,0,:]
    epsa_2d = epsa_3d[:,:,0,:]    
else:
    epsv_2d = epsv_3d[:,0,:,:]
    epsa_2d = epsa_3d[:,0,:,:]

f2 = f_iso2d(emask_2d,ks,n = n, swap_axes = SWAP_AXES)
f3 = f_iso3d(emask_3d,ks, n = n)

cmat2 = stack_mat2d(ks, d, epsv_2d,epsa_2d, mask = emask_2d,swap_axes = SWAP_AXES)
cmat3 = stack_mat3d(ks, d, epsv_3d,epsa_3d, mask = emask_3d)

smat2 = system_mat2d(cmat2, f2)
smat3 = system_mat3d(cmat3, f3)

rmat2 = reflection_mat2d(smat2)
rmat3 = reflection_mat3d(smat3)

grouped_modes_out2 = reflect2d(grouped_modes_in2, rmat2, f2, f2)
grouped_modes_out3 = reflect3d(grouped_modes_in3, rmat2, f3, f3)

modes_out2 = unlist_modes(grouped_modes_out2)
modes_out3 = unlist_modes(grouped_modes_out3)

modes_in2 = unlist_modes(grouped_modes_in2)
modes_in3 = unlist_modes(grouped_modes_in3)

field_out2 = modes2field1(emask2,modes_out2)
field_out3 = modes2field(emask3,modes_out3)

field_in2 = modes2field1(emask2,modes_in2)
field_in3 = modes2field(emask3,modes_in3)

solver = MatrixBlockSolver3D(field.shape[-2:],wavelengths,pixelsize,mask = emask3)
solver.set_optical_block((d, epsv_3d,epsa_3d))
solver.calculate_stack_matrix()
solver.calculate_field_matrix(n,n)
solver.calculate_reflectance_matrix()
solver.transfer_field(field)

def allclose(a,b):
    return np.allclose(a,b, rtol = rtol, atol = atol)

class TestSolver3D(unittest.TestCase):
    def test_field_matrices(self):
        for i in range(len(ks)):
            for a,b in zip(f3[i], solver.field_matrix_in[i]):
                self.assertTrue(allclose(a,b))
        for i in range(len(ks)):
            for a,b in zip(f3[i], solver.field_matrix_out[i]):
                self.assertTrue(allclose(a,b))

    def test_stack_matrices(self):
        for i in range(len(ks)):
            for a,b in zip(cmat3[i], solver.stack_matrix[i]):
                self.assertTrue(allclose(a,b))
                
    def test_refl_matrices(self):
        for i in range(len(ks)):
            for a,b in zip(rmat3[i], solver.refl_matrix[i]):
                self.assertTrue(allclose(a,b))
                
    def test_modes_out(self):
        for i in range(len(ks)):
            for a,b in zip(modes_out3[i], solver.modes_out[i]):
                self.assertTrue(allclose(a,b))        

    def test_modes_in(self):
        for i in range(len(ks)):
            for a,b in zip(modes_in3[i], solver.modes_in[i]):
                self.assertTrue(allclose(a,b))        

    def test_field_out(self):
        self.assertTrue(allclose(field_out3,solver.field_out))        

    def test_field_out(self):
        self.assertTrue(allclose(field_in3,solver.field_in))        


class TestUpscale(unittest.TestCase):
    def test_upscale2(self):
        mask, modes = field2modes(field_3d,ks)
        #2d stack mat
        cmat = stack_mat3d(ks, d, epsv_3d,epsa_3d, mask = mask)
        
        #2d mode mask
        mask_tuple = mode_masks(mask, shape = epsv_3d.shape[-3:-1])
        #upscale to 3d
        out = upscale2(cmat,mask_tuple)
        #test data
        for i in range(len(ks)):
            masks = mask_tuple[i]
            out_list = out[i]
            mats = cmat[i]
            o = out_list[0] #single element list
            
            for mat, mask in zip(mats,masks):
                self.assertTrue(allclose(mat, o[mask][:,mask] ))
                
                
    def test_upscale1(self):
        mask, modes = field2modes(field_3d,ks)
        #2d mode mask
        mask_tuple = mode_masks(mask, shape = epsv_3d.shape[-3:-1])
        
        #1d stack mat 
        cmat = stack_mat3d(ks, d, epsv_3d[...,0:1,0:1,:],epsa_3d[...,0:1,0:1,:], mask = mask)
        #upscale to 3d
        out = upscale1(cmat,mask_tuple)
        #test data
        for i in range(len(ks)):
            masks = mask_tuple[i]
            out_list = out[i]
            mats = cmat[i]
            
            for o, mask in zip(out_list,masks):
                ms = mats[mask]
                for i in range(len(ms)):
                    self.assertTrue(allclose(ms[i], o[i,i] ))
                


class TestEqual(unittest.TestCase):
    def test_equal_f_iso(self):
        #both are tuples of lists of length 1, should be identical
        for i in range(len(ks)):
            self.assertTrue(allclose(f2[i][0],f3[i][0]))
            
    def test_equal_layer_matrix(self):
        lmat2 = layer_mat2d(ks, d[0], epsv_2d[0],epsa_2d[0], mask = emask_2d, swap_axes = SWAP_AXES)
        lmat3 = layer_mat3d(ks, d[0], epsv_3d[0],epsa_3d[0], mask = emask_3d)
        #both are tuples of lists of length 1, should be identical
        for i in range(len(ks)):
            self.assertTrue(allclose(lmat2[i][0],lmat3[i][0]))    
            
    def test_equal_stack_matrix(self):
        #both are tuples of lists of length 1, should be identical
        for i in range(len(ks)):
            self.assertTrue(allclose(cmat2[i][0],cmat3[i][0]))     
            
    def test_equal_system_matrix(self):
        for i in range(len(ks)):
            self.assertTrue(allclose(smat2[i][0],smat3[i][0]))     
        
    def test_equal_reflection_matrix(self):
        for i in range(len(ks)):
            self.assertTrue(allclose(rmat2[i][0],rmat3[i][0]))  
            
    def test_equal_refl_data(self):
        for i in range(len(ks)):
            self.assertTrue(allclose(grouped_modes_in2[i][0],grouped_modes_in3[i][0]))  
        
        for i in range(len(ks)):
            self.assertTrue(allclose(grouped_modes_out2[i][0],grouped_modes_out3[i][0]))  

    def test_equal_modes_data(self):
        for i in range(len(ks)):
            self.assertTrue(allclose(modes_in2[i], modes_in3[i]))  
        
        for i in range(len(ks)):
            self.assertTrue(allclose(modes_out2[i],modes_out3[i]))  

    def test_equal_field_data(self):
        for i in range(len(ks)):
            if SWAP_AXES:
                self.assertTrue(allclose(field_in2[i], field_in3[i][:,:,0]))
            else:
                self.assertTrue(allclose(field_in2[i], field_in3[i][:,0,:]))  
        
        for i in range(len(ks)):
            if SWAP_AXES:
                self.assertTrue(allclose(field_out2[i],field_out3[i][:,:,0]))  
            else:
                self.assertTrue(allclose(field_out2[i],field_out3[i][:,0,:]))
            
            
if __name__ == "__main__":
    unittest.main()