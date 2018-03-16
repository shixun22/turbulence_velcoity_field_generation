"""
Generate 3D velocity field v0 on grid of shape (NX, NY, NZ), that has Gaussian pdf, and a given power spectrum Ek with specified normalisation sigmaV and kmax
Commented codes: check E(k) and pdf of the generated velocity field v0
"""

import numpy as np
#import matplotlib.pyplot as plt

# input parameters:
sigmaV = 300.  # velocity dispersion
kmax = 2. * np.pi / 200.  # where E(k) peaks
NX = 100  
NY = 400
NZ = 2 
gridsize = 10.  # Lx = NX * gridsize, etc.

def Ek(k, kmax, sigmaV):
    """
    Turbulence power spectrum E(k) with -5/3 Kolmogorov spectrum and a low-k cutoff 
    k Ek normalisation given by sigmaV^2
    Ek peaks at kmax
    """
    kc = kmax / 0.6  
    E_k = (k/kc)**(- 5./3) * np.exp(-1./(k / kc)) * (sigmaV**2 / 2.71) / kc
    return E_k

def vf_3d_spec(k, kmax, sigmaV):
    #! 4. * pi * k**3 * |vf_3d_spec(k)|**2  =  2. * k * Ek   
    vf2 = 2. * k * Ek(k, kmax, sigmaV) / (4. * np.pi * k**3)
    return vf2**0.5

# generate v
kx = np.fft.fftfreq(NX).reshape(NX,1,1) / gridsize * 2. * np.pi
ky = np.fft.fftfreq(NY).reshape(NY,1) / gridsize * 2. * np.pi
kz = np.fft.fftfreq(NZ) / gridsize * 2. * np.pi
kk = (kx**2 + ky**2 + kz**2)**0.5

v0 = []
for i in range(3):
    vx1 = np.random.randn(NX,NY,NZ)
    vx1 = (vx1 - vx1.mean()) / vx1.std()
    vx_f = np.fft.fftn(vx1)
    vx_f = vx_f * vf_3d_spec(kk, kmax, sigmaV/3.**0.5) * (2. * np.pi / gridsize)**(3./2) 
    vx_f[0,0,0] = 0.
    vx = np.fft.ifftn(vx_f).real
    v0.append(vx)
v0 = np.array(v0)


# save generated velocity field to binary file to be read by Fortran code
v0.reshape((3,NX,NY,NZ),order='F').tofile('v0_xyz_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')


## check properties of generated velocity field
#v0_tilde = np.array([np.fft.fftn(v0[ii]) * (gridsize / (2. * np.pi))**3 for ii in range(3)])
#ps = (np.abs(v0_tilde)**2).sum(axis=0)
#kkk = (kk.flatten())[1:]
#ppp = (ps.flatten())[1:]
#ksort = kkk[kkk.argsort()]
#psort = ppp[kkk.argsort()]
#NK = 50
#kh = np.linspace(ksort[0]*1., ksort[-1], NK+1)
#khc = ((kh[1:]**3 + kh[:-1]**3)/2.)**(1./3) #**0.5
#pph = np.array([(psort[np.where((ksort>kh[i]) & (ksort<kh[i+1]))]).mean() for i in range(kh.size-1)])
#Ek_ = pph * (2. *np.pi)**3 * (2. * np.pi * khc**2) / gridsize**3 / NX / NY / NZ
#
#print 'v2/2:', v0.var() / 2.
#print 'Ek:', Ek_.mean() * np.diff(khc)[0] * NK 
#
#Ek_in = Ek(khc, kmax, sigmaV)
#
#print 
#print 'v0.std:',  v0.std()
#print 'sigma from 1d spectrum', ((Ek_[1:] * np.diff(khc)).sum() * 2.)**0.5
#print '(should be smaller than input sigmaV because not the whole k range is numerically sampled)'
#
#plt.figure()
#plt.ion()
#plt.loglog(khc, Ek_, 'b-', label='Ek, sampled')
#plt.loglog(khc, Ek_in, 'k--', label='Ek, input')
#plt.xlabel('$k [kpc^{-1}]$', fontsize=18)
#plt.ylabel('$E(k)$', fontsize=18)
#plt.legend(loc=0)
#plt.show()
#
#print  Ek_ / Ek_in
#
#plt.figure()
#plt.hist(v0.flatten(), 1000)
#plt.xlabel('$v_0$', fontsize=16)


