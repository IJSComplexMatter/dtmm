import unittest
import numpy as np
import dtmm.data as data
import dtmm.rotation as rot
import dtmm.conf as conf

Rx = rot.rotation_matrix_x
Ry = rot.rotation_matrix_y
Rz = rot.rotation_matrix_z

class TestData(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test_rotations(self):
        
        rot90str = ("+x","-2x","+2y","-y","y","z","3z")
        rotmat = (Rx,Rx,Ry,Ry,Ry,Rz,Rz)
        angles = (np.pi/2,-np.pi,np.pi,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2)

        d = np.random.randn(5,5,5,3) #nlayers,height,width, 3  
        d = np.asarray(d, dtype = conf.FDTYPE) #in case we are in single precision cast here
        for s,r,a in zip(rot90str,rotmat,angles):
            d90 = data.rot90_director(d,s)
            rotm = r(a)
            dR = data.rotate_director(rotm, d)#should be same, no interpolation needed
        
            self.assertTrue(np.allclose(d90,dR)) 


        
if __name__ == "__main__":
    unittest.main()