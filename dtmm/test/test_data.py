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

#    def test_rotate_director(self):
#        """
#        Tests the rotate_director function using rot90_director.
#        This test only verifies 90 degree rotations since other rotations are not exact.
#        Returns
#        -------
#
#        """
#        # Strings representing the rotation
#        rotation_strings = ("+x", "-2x", "+2y", "-y", "y", "z", "3z")
#        # Rotation functions to use
#        rotation_functions = (Rx, Rx, Ry, Ry, Ry, Rz, Rz)
#        # Angles used to create rotation matrices
#        angles = (np.pi/2, -np.pi, np.pi, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2)
#
#        # Number of vertical layers
#        n_layers = 5
#        # height of each layer
#        heigh = 5
#        # width of each layer
#        width = 5
#
#        # Create random test data of size
#        test_data = np.random.randn(n_layers, heigh, width, 3)
#        # Cast data to correct data type. Done in case single precision is being used
#        test_data = np.asarray(test_data, dtype=conf.FDTYPE)
#
#        # Iterate through each rotation test
#        for rotation_str, rotation_function, angle in zip(rotation_strings, rotation_functions, angles):
#            # Calculate exact result using rot90_director
#            rotated_data_goal = data.rot90_director(test_data, rotation_str)
#            # Create rotation matrix
#            rotation_matrix = rotation_function(angle)
#            # Rotate director
#            rotated_data = data.rotate_director(rotation_matrix, test_data)
#            # Compare
#            self.assertTrue(np.allclose(rotated_data_goal, rotated_data))


if __name__ == "__main__":
    unittest.main()
