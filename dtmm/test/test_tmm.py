"""Tests for fft"""

import unittest
import numpy as np
from dtmm.tmm import f_iso, fvec, fvec2E, E2fvec, alphaf, normalize_f
import dtmm.conf
from dtmm.conf import MKL_FFT_INSTALLED


rtol, atol = (1e-05,1e-08) if dtmm.conf.PRECISION == "double" else (1e-3,1e-4)
def allclose(a,b):
    return np.allclose(a,b, rtol = rtol, atol = atol)


class TestConversion(unittest.TestCase):
    
    def allclose(self,a,b):
        self.assertTrue(allclose(a,b))
        
    def iscomplex(self,a):
        self.assertTrue(a.dtype == dtmm.conf.CDTYPE)

    def isfloat(self,a):
        self.assertTrue(a.dtype == dtmm.conf.FDTYPE)
    
    def setUp(self):
        self.f0 = f_iso(beta = 0.2, phi = 0.2, n = 1.5)
        self.fvecp = fvec(self.f0, jones = (1,0), mode = +1)
        self.fvecm = fvec(self.f0, jones = (1,0), mode = -1)
        self.fvec = self.fvecp + self.fvecm
     
    def test_fvec2E(self):
        jp = fvec2E(self.fvec, self.f0, mode = +1)
        jm = fvec2E(self.fvec, self.f0, mode = -1)
        
        self.assertTrue(np.allclose(jp, self.fvecp[::2]))
        self.assertTrue(np.allclose(jm, self.fvecm[::2]))
        
        jp = fvec2E(self.fvecp, self.f0, mode = -1)
        jm = fvec2E(self.fvecm, self.f0, mode = +1)
        
        self.assertTrue(np.allclose(jp, np.asarray((0j,0j)),atol = 1e-6))
        self.assertTrue(np.allclose(jm, np.asarray((0j,0j)),atol = 1e-6)) 
        
    def test_alphaf(self):
        for e in (1,2):
            alpha0, f0 = alphaf(0, 0, (e,e,e),(0,0,0))
            
            alpha, f = alphaf(0.00001, 0.0001, (e,e,e),(0,0,0))

            self.allclose(np.round(f,3), np.round(f0,3))
            
            alpha, f = alphaf(0.,0, (e,e,e+0.0001),(0,0,0))
            self.allclose(np.round(normalize_f(f),3), np.round(normalize_f(f0),3))

            alpha, f = alphaf(0.0001,0, (e,e,e+0.0001),(0,0,0))
            self.allclose(np.round(normalize_f(f),3), np.round(normalize_f(f0),3))
            
            alpha0, f0 = alphaf(0, np.pi/4, (e,e,e),(0,0,0))

            alpha, f = alphaf(0.0001,np.pi/4, (e,e,e+0.0001),(0,0,0))
            self.allclose(np.round(normalize_f(f),3), np.round(normalize_f(f0),3))



                
if __name__ == "__main__":
    unittest.main()