import unittest
import numpy as np
import dtmm.wave as wave

class TestWave(unittest.TestCase):
    
    def setUp(self):
        self.shape = (64,64)
        self.ks = [1,2]
        self.beta, self.phi = wave.betaphi(self.shape,self.ks)

    def test_planewave_shapes(self):
        shapes = ((16,16), (15,15),(12,13),(7,6))
        k=1
        for shape in shapes:
            b,p = wave.betaphi(shape,k)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    w1 = wave.planewave(shape,k,b[i,j],p[i,j])
                    w2 = wave.eigenwave(shape,i,j)
                    self.assertTrue(np.allclose(w1,w2))  

    def test_planewave_k(self):
        shape = (13,11)
        for k in (0.1,1.,10):
            b,p = wave.betaphi(shape,k)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    w1 = wave.planewave(shape,k,b[i,j],p[i,j])
                    w2 = wave.eigenwave(shape,i,j)
                    self.assertTrue(np.allclose(w1,w2)) 

        
if __name__ == "__main__":
    unittest.main()