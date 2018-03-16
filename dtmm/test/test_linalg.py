import unittest
import numpy as np
import dtmm.linalg as linalg


def dm(d):
    """transforms to diagonal matrix from a vector"""
    dm = np.zeros(d.shape + (d.shape[-1],), d.dtype)
    for i in range(4):
        dm[...,i,i] = d[...,i]    
    return dm

class TestLinalg2(unittest.TestCase):
    
    def setUp(self):
        self.ni = 15
        self.nj = 16
        self.f = np.random.randn(4,self.ni,self.nj)+0j
        self.a = np.random.randn(self.ni,self.nj,4,4)+0j
        self.d = np.random.randn(self.ni,self.nj,4)+0j
        self.b = np.random.randn(self.ni,self.nj,4,4)+0j

    def dotmf(self,f,i,j,*matrices):
        mij = [m[i,j] for m in matrices] + [f[:,i,j]]
        return np.linalg.multi_dot(mij)
    
    def dot_multi(self,i,j,matrices):
        ms = [m[i,j].copy() for m in matrices[0:-1]] + [matrices[-1][:,i,j]]
        return np.linalg.multi_dot(ms)        
    
    def compare_results(self,out, matrices):
        for i in range(self.ni):
            for j in range(self.nj):   
                out2 = self.dot_multi(i,j,matrices)
                self.assertTrue(np.allclose(out[:,i,j],out2))        
        
    def test_dotmf(self):
        out = linalg.dotmf(self.a,self.f)
        self.compare_results(out,[self.a, self.f])
  
    def test_dotmdmf(self):
        out = linalg.dotmdmf(self.a,self.d,self.b,self.f)
        matrices = [self.a,dm(self.d),self.b, self.f]
        self.compare_results(out,matrices)
        
    def test_dotmdmf2(self):
        kd = 2.3
        e = np.exp(1j*kd*self.d)
        out = linalg.dotmdmf(self.a,e,self.b,self.f)
        matrices = [self.a,dm(e),self.b, self.f]
        self.compare_results(out,matrices)        

    def test_ftransmit(self):
        kd = 2.3
        out = linalg.ftransmit(kd,self.a,self.d.real,self.b,self.f)
        matrices = [self.a,dm(np.exp(1j*kd*self.d.real)),self.b,self.f]
        self.compare_results(out,matrices)

    def test_btransmit(self):
        kd = 2.3
        out = linalg.btransmit(kd,self.a,self.d,self.b,self.f)
        out2 = linalg.ftransmit(-kd,self.a,self.d,self.b,self.f)
        self.assertTrue(np.allclose(out,out2)) 

                
if __name__ == "__main__":
    unittest.main()