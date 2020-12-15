import unittest
import numpy as np
import dtmm.wave as wave
import dtmm.conf

rtol, atol = (1e-05,1e-08) if dtmm.conf.PRECISION == "double" else (1e-3,1e-4)
def allclose(a,b):
    return np.allclose(a,b, rtol = rtol, atol = atol)

class TestWave(unittest.TestCase):
    
    def allclose(self,a,b):
        self.assertTrue(allclose(a,b))
        
    def iscomplex(self,a):
        self.assertTrue(a.dtype == dtmm.conf.CDTYPE)

    def isfloat(self,a):
        self.assertTrue(a.dtype == dtmm.conf.FDTYPE)
    
    def setUp(self):
        self.shape = (4,3)
        self.ks = [1,2]
        self.beta, self.phi = (np.array([[[0.        , 2.0943951 , 2.0943951 ],
         [1.57079633, 2.61799388, 2.61799388],
         [3.14159265, 3.77572447, 3.77572447],
         [1.57079633, 2.61799388, 2.61799388]],
 
        [[0.        , 1.04719755, 1.04719755],
         [0.78539816, 1.30899694, 1.30899694],
         [1.57079633, 1.88786223, 1.88786223],
         [0.78539816, 1.30899694, 1.30899694]]]),
 np.array([[ 0.        ,  0.        ,  3.14159265],
        [ 1.57079633,  0.64350111,  2.49809154],
        [-1.57079633, -0.98279372, -2.15879893],
        [-1.57079633, -0.64350111, -2.49809154]]))
        
        self.betax, self.betay = (np.array([[[ 0.        ,  2.0943951 , -2.0943951 ],
         [ 0.        ,  2.0943951 , -2.0943951 ],
         [ 0.        ,  2.0943951 , -2.0943951 ],
         [ 0.        ,  2.0943951 , -2.0943951 ]],
 
        [[ 0.        ,  1.04719755, -1.04719755],
         [ 0.        ,  1.04719755, -1.04719755],
         [ 0.        ,  1.04719755, -1.04719755],
         [ 0.        ,  1.04719755, -1.04719755]]]),
 np.array([[[ 0.        ,  0.        ,  0.        ],
         [ 1.57079633,  1.57079633,  1.57079633],
         [-3.14159265, -3.14159265, -3.14159265],
         [-1.57079633, -1.57079633, -1.57079633]],
 
        [[ 0.        ,  0.        ,  0.        ],
         [ 0.78539816,  0.78539816,  0.78539816],
         [-1.57079633, -1.57079633, -1.57079633],
         [-0.78539816, -0.78539816, -0.78539816]]]))
        
        self.eigenindices = (np.array([[0, 0],
                                [1, 0],
                                [3, 0]]),
                         np.array([[0, 0],
                                [0, 1],
                                [0, 2],
                                [1, 0],
                                [1, 1],
                                [1, 2],
                                [2, 0],
                                [3, 0],
                                [3, 1],
                                [3, 2]]))
        
    def test_betaphi(self):
        b,p = wave.betaphi(self.shape,self.ks)
        self.allclose(self.beta,b)
        self.allclose(self.phi,p)
        
        out = np.empty_like(b), np.empty_like(p)
        wave.betaphi(self.shape,self.ks, out = out)
        self.allclose(self.beta,out[0])
        self.allclose(self.phi,out[1])
        
        b,p = wave.betaphi(self.shape,self.ks[0])
        self.allclose(self.beta[0],b)
        self.allclose(self.phi,p)   
        self.isfloat(b)
        self.isfloat(p)
        
    def test_betaxy(self):
        bx,by = wave.betaxy(self.shape,self.ks)
        self.allclose(self.betax,bx)
        self.allclose(self.betay,by)
        
        out = np.empty_like(bx), np.empty_like(by)
        wave.betaxy(self.shape,self.ks, out = out)
        self.allclose(self.betax,out[0])
        self.allclose(self.betay,out[1])
        
        bx,by = wave.betaxy(self.shape,self.ks[1])
        self.allclose(self.betax[1],bx)
        self.allclose(self.betay[1],by)  
        self.isfloat(bx)
        self.isfloat(by)
        
    def test_betaxy2betaphi(self):
        b,p = wave.betaxy2betaphi(self.betax, self.betay)
        self.allclose(b,self.beta)
        self.allclose(p,self.phi)
        self.isfloat(b)
        self.isfloat(p)
        
    def test_betaphi2betaxy(self):
        bx,by = wave.betaphi2betaxy(self.beta, self.phi)
        self.allclose(self.beta*np.cos(self.phi), bx)
        self.allclose(self.beta*np.sin(self.phi), by)
        self.isfloat(bx)
        self.isfloat(by)

    def test_eigenwave(self):
        w = wave.eigenwave((4,3), 0, 0, amplitude = 4*3)
        self.allclose(w,np.ones((4,3)))
        w = wave.eigenwave((4,3), 0, 0)
        self.allclose(w,np.ones((4,3)))
        out = np.empty_like(w)
        wave.eigenwave((4,3), 0, 0, out = out)
        self.allclose(out,np.ones((4,3)))
        self.iscomplex(out)

        
    def test_eigenbeta(self):
        
        m0 = self.beta[0] < 1.8
        m1 = self.beta[1] < 1.8
        
        out0 = self.beta[0][m0]
        out1 = self.beta[1][m1]
        
        out = wave.eigenbeta(self.shape, self.ks, betamax = 1.8)
        
        self.allclose(out[0], out0)
        self.allclose(out[1], out1)
        self.isfloat(out[0])
        
    def test_eigenphi(self):
        
        m0 = self.beta[0] < 1.8
        m1 = self.beta[1] < 1.8
        
        out0 = self.phi[m0]
        out1 = self.phi[m1]
        
        out = wave.eigenphi(self.shape, self.ks, betamax = 1.8)
        
        self.allclose(out[0], out0)
        self.allclose(out[1], out1)
        self.isfloat(out[0])
        
    def test_eigenmask(self):       
        m = self.beta < 1.8        
        out = wave.eigenmask(self.shape, self.ks, betamax = 1.8)      
        self.allclose(out, m)
    
    def test_eigenindices(self):            
        out = wave.eigenindices(self.shape, self.ks, betamax = 1.8)      
        self.allclose(out[0], self.eigenindices[0])   
        self.allclose(out[1], self.eigenindices[1])  
        
    def test_k0(self):
        out = wave.k0([550,650], 100)
        self.allclose(out, 2*np.pi/np.array((550,650))*100)
        self.isfloat(out)
        
    def test_mask2beta(self):
        mask = wave.eigenmask((4,3), (1,2), 1.8)
        beta = wave.mask2beta(mask, (1,2))
        self.allclose(beta[0], self.beta[0][mask[0]])
        self.allclose(beta[1], self.beta[1][mask[1]])       
        beta = wave.mask2beta(mask[1], 2)
        self.allclose(beta, self.beta[1][mask[1]])
        self.isfloat(beta)
 
    def test_mask2phi(self):
        mask = wave.eigenmask((4,3), (1,2), 1.8)
        phi = wave.mask2phi(mask, (1,2))
        self.allclose(phi[0], self.phi[mask[0]])
        self.allclose(phi[1], self.phi[mask[1]])       
        phi = wave.mask2phi(mask[1], 2)
        self.allclose(phi, self.phi[mask[1]])
        self.isfloat(phi)    

    def test_mask2indices(self):
        mask = wave.eigenmask((4,3), (1,2), 1.8)
        indices = wave.mask2indices(mask, (1,2))
        self.allclose(indices[0], self.eigenindices[0])
        self.allclose(indices[1], self.eigenindices[1])       
        indices = wave.mask2indices(mask[1], 2)
        self.allclose(indices, self.eigenindices[1])    
        
    def test_wave2eigenwave(self):
        w = wave.planewave((12,14),(1,2),0.5,0)
        e = wave.wave2eigenwave(w)
        e0 = wave.eigenwave((12,14),0,1)
        self.allclose(e[0],e0)  
        e0 = wave.eigenwave((12,14),0,2)
        self.allclose(e[1],e0)   
        out = np.empty_like(e)
        wave.wave2eigenwave(w, out = out)
        self.allclose(out,e) 
        self.iscomplex(e)
        
        

    def test_planewave_shapes(self):
        shapes = ((16,16), (15,15),(12,13),(7,6))
        k=1
        for shape in shapes:
            b,p = wave.betaphi(shape,k)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    w1 = wave.planewave(shape,k,b[i,j],p[i,j])
                    w2 = wave.eigenwave(shape,i,j)
                    self.allclose(w1,w2)

    def test_planewave_k(self):
        shape = (13,11)
        for k in (0.1,1.,10):
            b,p = wave.betaphi(shape,k)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    w1 = wave.planewave(shape,k,b[i,j],p[i,j])
                    w2 = wave.eigenwave(shape,i,j)
                    self.allclose(w1,w2) 
                    
    def test_eigenmask1(self):
        m = wave.eigenmask((1,13), (1,2), betamax = 1.5)
        m1 = wave.eigenmask1(13,(1,2), betamax = 1.5)
        self.allclose(m[:,0,:], m1)
        m = wave.eigenmask((1,13), 2, betamax = 1.5)
        m1 = wave.eigenmask1(13,2, betamax = 1.5)
        self.allclose(m[0,:], m1)  
        
        m = wave.eigenmask((1,13), 2, betamax = 1.5)
        m1 = wave.eigenmask1(13,2, betamax = 1.5)
        self.allclose(m[0,:], m1)
        
    def test_betax1(self):
        bx,by = wave.betaxy((1,19),(1,2))
        bx1 = wave.betax1(19,(1,2))
        self.allclose(bx[0][0],bx1[0]) 
        out = np.empty_like(bx1)
        bx1 = wave.betax1(19,(1,2), out = out)
        self.allclose(bx[0][0],out[0])         

        bx,by = wave.betaxy((1,19),2)
        bx1 = wave.betax1(19,2)
        self.allclose(bx[0],bx1) 
        
    def test_eigenwave1(self):
        w = wave.eigenwave((1,12),0,2)
        w1 = wave.eigenwave1(12,2)
        self.allclose(w[0,:],w1)  
        w1 = wave.eigenwave1(12,2, amplitude = 12)
        self.allclose(w[0,:],w1)          
                    
        
    def test_planewave1(self):
        w = wave.planewave((1,12),(1,2),0.4,0)
        w1 = wave.planewave1(12,(1,2),0.4)
        self.allclose(w[...,0,:],w1)
        w = wave.planewave((12,1),(1,2),0.4,np.pi/2)
        self.allclose(w[...,0],w1)   
        out = np.empty_like(w1)
        wave.planewave1(12,(1,2),0.4, out = out)
        self.allclose(out,w1)
        
        w = wave.planewave((1,12),2,0.4,0)
        w1 = wave.planewave1(12,2,0.4)
        self.allclose(w[0,:],w1)
        

    def test_eigenbetax1(self):
        beta = wave.eigenbeta((1,6),(1,2),2)
        phi = wave.eigenphi((1,6),(1,2),2)       
        beta1 = wave.eigenbeta1(6,(1,2),2)
        self.allclose(beta[0]*np.cos(phi[0]),beta1[0])
 
        beta = wave.eigenbeta((1,6),2,2)
        phi = wave.eigenphi((1,6),2,2)       
        beta1 = wave.eigenbeta1(6,2,2)
        self.allclose(beta*np.cos(phi),beta1)       
 
    def test_eigenindices1(self):            
        out = wave.eigenindices((1,13), (2,3), betamax = 1.8)   
        out1 = wave.eigenindices1(13, (2,3), betamax = 1.8) 
        self.allclose(out[0][:,1], out1[0])   
        self.allclose(out[1][:,1], out1[1]) 
        
        out = wave.eigenindices((1,13), 3, betamax = 1.8)   
        out1 = wave.eigenindices1(13, 3, betamax = 1.8) 
        self.allclose(out[:,1], out1)   
        
    def test_mask2indices1(self):
        mask = wave.eigenmask((1,12), (1,2), betamax = 1.8)
        indices = wave.mask2indices(mask, (1,2))
        mask1 = wave.eigenmask1(12, (1,2), betamax = 1.8)
        indices1 = wave.mask2indices1(mask1, (1,2))
        self.allclose(indices[0][...,1], indices1[0])
        self.allclose(indices[1][...,1], indices1[1])

        mask = wave.eigenmask((1,12), 2, betamax = 1.8)
        indices = wave.mask2indices(mask, 2)
        mask1 = wave.eigenmask1(12, 2, betamax = 1.8)
        indices1 = wave.mask2indices1(mask1, 2)
        self.allclose(indices[...,1], indices1)
        
    def test_mask2betax1(self):
        mask = wave.eigenmask((12,1), (1,2), 1.8)
        beta = wave.mask2beta(mask, (1,2))
        mask1 = wave.eigenmask1(12, (1,2), 1.8)
        betax1 = wave.mask2betax1(mask1, (1,2))        
        
        self.allclose(beta[0], np.abs(betax1[0]))
        self.allclose(beta[1], np.abs(betax1[1]))
        
        mask = wave.eigenmask((12,1), 2, 1.8)
        beta = wave.mask2beta(mask, 2)
        mask1 = wave.eigenmask1(12, 2, 1.8)
        betax1 = wave.mask2betax1(mask1, 2)        
        
        self.allclose(beta, np.abs(betax1))
   
if __name__ == "__main__":
    unittest.main()