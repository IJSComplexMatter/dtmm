"""Tests for fft"""

import unittest
import numpy as np
from dtmm.fft import fft2, ifft2
import dtmm.conf
from dtmm.conf import MKL_FFT_INSTALLED

class TestFFT(unittest.TestCase):
    
    def setUp(self):
        a0 = np.random.randn(46,64)+0j
        af0 = np.fft.fft2(a0)
        afi0 = np.fft.ifft2(a0)        
        
        a = np.random.randn(2,2,46,64)+0j
        af = np.fft.fft2(a)
        afi = np.fft.ifft2(a)
        
        self._farrays = [(a0,af0),(a,af)]
        self._fiarrays = [(a0,afi0),(a,afi)]
     
    def _assert_fft_inplace(self, fftfunc, inarray, result):
        a = inarray.copy()
        out = fftfunc(a,a)
        self.assertTrue(np.allclose(out, result))

    def _assert_fft_out(self, fftfunc, inarray, result):
        a = inarray.copy()
        out = np.empty_like(a)
        out = fftfunc(a,out)
        self.assertTrue(np.allclose(out, result)) 

    def _assert_fft_new(self, fftfunc, inarray, result):
        a = inarray.copy()
        out = fftfunc(a)
        self.assertTrue(np.allclose(out, result))  
        
    def _assert_fft(self, fftfunc, inarray, result):
        self._assert_fft_new(fftfunc, inarray,result)
        self._assert_fft_out(fftfunc, inarray,result)
        self._assert_fft_inplace(fftfunc, inarray,result)
        
    def test_mkl_fft2(self):
        if MKL_FFT_INSTALLED:
            dtmm.conf.set_fftlib("mkl_fft")
            for a,out in self._farrays:
                self._assert_fft(fft2, a, out)

    def test_mkl_ifft2(self):
        if MKL_FFT_INSTALLED:
            dtmm.conf.set_fftlib("mkl_fft")
            for a,out in self._fiarrays:
                self._assert_fft(ifft2, a, out)
        
    def test_np_fft2(self):
        dtmm.conf.set_fftlib("numpy")
        for a,out in self._farrays:
            self._assert_fft(fft2, a, out)

    def test_np_ifft2(self):
        dtmm.conf.set_fftlib("numpy")
        for a,out in self._fiarrays:
            self._assert_fft(ifft2, a, out)
    
    def test_sp_fft2(self):
        dtmm.conf.set_fftlib("scipy")
        for a,out in self._farrays:
            self._assert_fft(fft2, a, out)

    def test_sp_ifft2(self):
        dtmm.conf.set_fftlib("scipy")
        for a,out in self._fiarrays:
            self._assert_fft(ifft2, a, out)

                
if __name__ == "__main__":
    unittest.main()