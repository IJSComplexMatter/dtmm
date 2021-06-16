"""This example uses a standard (non-diffractive) 4x4 and 2x2 methods
to calculate transmitance and reflections of a nematic droplet.
"""

import dtmm
import numpy as np

dtmm.conf.set_verbose(2)

#: pixel size in nm
PIXELSIZE = 200
#: compute box dimensions
NLAYERS, HEIGHT, WIDTH = 60,96,96
#: illumination wavelengths in nm
WAVELENGTHS = np.linspace(380,780,9)
#: create some experimental data (stack)
d, epsv, epsa = dtmm.nematic_droplet_data((NLAYERS, HEIGHT, WIDTH), 
          radius = 30, profile = "x", no = 1.5, ne = 1.6, nhost = 1.5)

#: create non-polarized input light
f,w,p = dtmm.illumination_data((HEIGHT, WIDTH), WAVELENGTHS, 
                pixelsize = PIXELSIZE, beta = 0., phi = 0.) 

field_data_in = f,w,p

#transpose to field vector
ft = dtmm.field.field2fvec(f)

# build kd phase values
kd = [x*(dtmm.k0(WAVELENGTHS, PIXELSIZE))[...,None,None] for x in d]

#build stack matrix and transmit... 
cmat = dtmm.tmm.stack_mat(kd, epsv, epsa, method = "2x2")
#convert field vector to E vector (assuming vacuum)
Ein_2x2 = dtmm.tmm.fvec2E(ft)
Eout_2x2 = dtmm.linalg.dotmv(cmat,Ein_2x2)
#convert E vector to field vector (assuming vacuum)
fout_2x2 = dtmm.tmm.E2fvec(Eout_2x2)
#above is identical to:
fout_2x2_ref = dtmm.tmm.transfer(ft, kd, epsv, epsa, method = "2x2")
assert np.allclose(fout_2x2, fout_2x2_ref)
#also to:
fout_2x2_ref = dtmm.tmm.transmit(ft, cmat)  
assert np.allclose(fout_2x2, fout_2x2_ref)


cmat = dtmm.tmm.stack_mat(kd, epsv, epsa, method = "2x2_1")
#convert field vector to E vector (assuming vacuum)
Ein_2x4 = dtmm.tmm.fvec2E(ft)
#build filed matrices for additional reflections from the first and last interfaces
alpha,fin = dtmm.tmm.alphaf(epsv = epsv[0], epsa = epsa[0])
alpha,fout = dtmm.tmm.alphaf(epsv = epsv[-1], epsa = epsa[-1])
#build transmittance matrices
tmat1 = dtmm.tmm.t_mat(dtmm.tmm.f_iso(),fin)
tmat2 = dtmm.tmm.t_mat(fout, dtmm.tmm.f_iso())
#trasnmit through air-sample interface
Ein_2x4 = dtmm.linalg.dotmv(tmat1,Ein_2x4)
#transmit through sample
Eout_2x4 = dtmm.linalg.dotmv(cmat,Ein_2x4)
#transmit through sample-air interface
Eout_2x4 = dtmm.linalg.dotmv(tmat2,Eout_2x4)
fout_2x4 = dtmm.tmm.E2fvec(Eout_2x4)

# above is identical to:
fout_2x4_ref = dtmm.tmm.transfer(ft, kd, epsv, epsa, method = "2x2_1", reflect_in = True, reflect_out = True)
assert np.allclose(fout_2x4, fout_2x4_ref)
# also to:
fout_2x4_ref = dtmm.tmm.transmit(ft, cmat, tmatin = tmat1, tmatout = tmat2)   
assert np.allclose(fout_2x4, fout_2x4_ref)

cmat = dtmm.tmm.stack_mat(kd, epsv, epsa, method = "4x4_1")
smat = dtmm.tmm.system_mat(cmat)
rmat = dtmm.tmm.reflection_mat(smat)
fout_4x2 = dtmm.tmm.reflect(ft,rmat)
# fout_4x2 = dtmm.tmm.transmit(ft,cmat)
# above is identical to:
fout_4x2_ref = dtmm.tmm.transfer(ft, kd, epsv, epsa, method = "4x4_1")
assert np.allclose(fout_4x2, fout_4x2_ref)


cmat = dtmm.tmm.stack_mat(kd, epsv, epsa, method = "4x4")
smat = dtmm.tmm.system_mat(cmat)
rmat = dtmm.tmm.reflection_mat(smat)
fout_4x4 = dtmm.tmm.reflect(ft,rmat)
fout_4x4_ref = dtmm.tmm.transfer(ft, kd, epsv, epsa, method = "4x4")
assert np.allclose(fout_4x4, fout_4x4_ref)

# inverse transpose to build field data for visualization
field_data_out_2x2 = dtmm.field.fvec2field(fout_2x2),w,p
field_data_out_2x4 = dtmm.field.fvec2field(fout_2x4),w,p
field_data_out_4x2 = dtmm.field.fvec2field(fout_4x2),w,p
field_data_out_4x4 = dtmm.field.fvec2field(fout_4x4),w,p


#: visualize output field
viewer1 = dtmm.field_viewer(field_data_out_2x2, diffraction = False)
viewer1.set_parameters(sample = 45, intensity = 2,
                polarizer = 0, analyzer = 90)

fig,ax = viewer1.plot()
ax.set_title("2x2 method (jones method, no reflections)")
fig.show()

viewer2 = dtmm.field_viewer(field_data_out_2x4, diffraction = False )
viewer2.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer2.plot()
ax.set_title("2x4 method (jones method, with reflections)")
fig.show()

viewer3 = dtmm.field_viewer(field_data_out_4x2, diffraction = False )
viewer3.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer3.plot()
ax.set_title("4x2 method (4x4 with single reflection)")
fig.show()

viewer4 = dtmm.field_viewer(field_data_out_4x4, diffraction = False )
viewer4.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer4.plot()
ax.set_title("4x4 method (4x4 with interference)")
fig.show()

viewer5 = dtmm.field_viewer(field_data_in, mode = "r" )
viewer5.set_parameters(sample = 45, intensity = 2,
                polarizer = 0,  analyzer = 90)

fig,ax = viewer5.plot()
ax.set_title("4x4 method - reflected field")
fig.show()


