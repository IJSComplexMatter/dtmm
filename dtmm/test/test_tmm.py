"""Tests for fft"""

import unittest
import numpy as np
from dtmm.tmm import f_iso, fvec, fvec2E, E2fvec
import dtmm.conf
from dtmm.conf import MKL_FFT_INSTALLED

class TestConversion(unittest.TestCase):
    
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


                
if __name__ == "__main__":
    unittest.main()