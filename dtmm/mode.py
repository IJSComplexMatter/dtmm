#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFT mode selection
"""
import numpy as np
from dtmm.wave import betaxy
from dtmm.conf import cached_result, BETAMAX, FDTYPE, NFDTYPE, NUMBA_CACHE,NUMBA_TARGET
import numba as nb

@nb.guvectorize([(NFDTYPE[:,:],NFDTYPE[:,:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:],NFDTYPE[:,:])], 
                 "(i,j),(i,j),(),(),(),(),(),(),()->(i,j)",
                 target = NUMBA_TARGET, cache = NUMBA_CACHE)
def fft_window(fftbetax, fftbetay,betax, betay, stepx,stepy, xtyp, ytyp, betamax, out):
    ni, nj = fftbetax.shape
    for i in range(ni):
        for j in range(nj):
            fftbetaxij = fftbetax[i,j]
            dx = (fftbetaxij - betax[0])/stepx[0]
            cx = 1. - np.abs(dx)
            if cx < 0.:
                cx = 0.  
            else:
                if xtyp[0] < 0:
                    if dx < 0:
                        cx = 1.
                elif xtyp[0] > 0:
                    if dx > 0:
                        cx = 1.
            fftbetayij = fftbetay[i,j]
            dy = (fftbetayij - betay[0])/stepy[0]
            cy = 1. - np.abs(dy)
            if cy < 0.:
                cy = 0.
            else:
                if ytyp[0] < 0:
                    if dy < 0:
                        cy = 1.
                elif ytyp[0] > 0:
                    if dy > 0:
                        cy = 1.
            beta = (fftbetaxij**2 + fftbetayij**2)**0.5 
            if beta >= betamax[0]:
                coeff = 0.
            else:
                coeff = cx * cy
            if coeff > 0.:
                out[i,j] = coeff
            else:
                out[i,j] = 0.

def fft_windows(betax, betay, n, betax_off = 0., betay_off = 0., betamax = BETAMAX, out = None):

    d = 2*betamax/(n-1)
    xoffset = np.mod(betax_off, d)
    xoffsetm = np.mod(betax_off, -d) 
    mask = np.abs(xoffset) > np.abs(xoffsetm)
    try:
        xoffset[mask] = xoffsetm[mask]
    except TypeError: #scalar
        if mask:
            xoffset = xoffsetm
    
    yoffset = np.mod(betay_off, d)
    yoffsetm = np.mod(betay_off, -d) 
    mask = np.abs(yoffset) > np.abs(yoffsetm)
    try:
        yoffset[mask] = yoffsetm[mask]
    except TypeError:
        if mask:
            yoffset = yoffsetm
    
    ax = np.linspace(-betamax, betamax, n)
    ay = np.linspace(-betamax, betamax, n)
    step = ax[1]-ax[0]
    for i,bx in enumerate(ax):
        if i == 0:
            xtyp = -1
        elif i == n-1:
            xtyp = 1
        else:
            xtyp = 0
        for j,by in enumerate(ay):
            index = i*n+j
            if out is None:
                _out = None
            else:
                _out = out[index]
            
            if j == 0:
                ytyp = -1
            elif j == n-1:
                ytyp = 1
            else:
                ytyp = 0
            
            fmask = fft_window(betax,betay,bx+xoffset,by+yoffset,step,step,xtyp,ytyp,betamax, out = _out) 
            if out is None:
                out = np.empty((n*n,)+fmask.shape, fmask.dtype)
                out[0] = fmask
        
    return out

def fft_betaxy(shape, k0):
    bx,by = betaxy(shape[-2:], np.asarray(k0,FDTYPE)[...,None])
    return bx,by #np.broadcast_to(bx,shape),np.broadcast_to(by,shape)

def fft_betaxy_mean(betax, betay, fft_windows):
    axis = tuple(range(-betax.ndim,0))
    fft_windows = fft_windows**2 #weight should be the intensity, so square
    bx = betax*fft_windows
    by = betay*fft_windows
    norm = fft_windows.sum(axis = axis)
    bx = bx.sum(axis = axis) 
    by = by.sum(axis = axis) 
    mask = (norm>0)
    return np.divide(bx,norm, where = mask, out = bx), np.divide(by,norm, where = mask, out = by)

def betaxy2betaphi(bx,by):
    beta = (bx**2+by**2)**0.5
    phi = np.arctan2(by,bx)
    return beta, phi

@cached_result
def fft_mask_full(shape, k0, n, betax_off = 0., betay_off = 0., betamax = BETAMAX):
    
    betax, betay = fft_betaxy(shape, k0)
    windows = fft_windows(betax, betay, n, betax_off = betax_off, betay_off = betay_off, betamax = betamax)
    bxm, bym = fft_betaxy_mean(betax, betay, windows)

    return windows, betaxy2betaphi(bxm,bym)

@cached_result
def fft_mask(shape, k0, n, betax_off = 0., betay_off = 0., betamax = BETAMAX):
    windows, (bs,ps) = fft_mask_full(shape, k0, n, betax_off, betay_off, betamax)
    zero = np.asarray([np.alltrue(w == 0.) for w in windows])
    nonzero = np.logical_not(zero)
    
    return windows[nonzero], (bs[nonzero], ps[nonzero])
    
    
if __name__ == "__main__":
    S = 64 # array size
    W = 500 #wavelength
    P = 200 # pixel size in real space
    k = 2*np.pi/W * P #k for 500 nm
    N = 8
    p = W/P/S #pixel size in reciprocal spaxe 
    
    XOFF = 2*0
    YOFF = 2*0
    
    BMAX = 0.8
        
    BX = XOFF*p
    BY = YOFF*p
    
    w,bp = fft_mask_full((S,S), (k,), N, betax_off = BX, betay_off = BY, betamax = BMAX)
    import matplotlib.pyplot as plt
    
    beta,phi = bp
    beta = beta/p
    
    centerx = beta*np.cos(phi) + S//2
    centery = beta*np.sin(phi) + S//2
    
    originx = XOFF + S//2
    originy = YOFF + S//2
    
    xticks = np.linspace(-BMAX/p, BMAX/p,N) + S//2 + XOFF
    yticks = np.linspace(-BMAX/p, BMAX/p,N) + S//2 + YOFF
    #xlabel = np.arange(N) - N//2
    
    
    #fig = plt.figure(figsize=(4., 4.))
    #from mpl_toolkits.axes_grid1 import ImageGrid
    #grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #             nrows_ncols=(N, N),  # creates 2x2 grid of axes
    #             axes_pad=0.1,  # pad between axes in inch.
    #             )
    
    fig,axes = plt.subplots(N,N)
    for i in range(N):
        for j in range(N):
            n = i + N*j
            
            ax = axes[i,j]
            
            ax.imshow(np.fft.fftshift(w[n,0,0]))
            #ax.axis('off')
            ax.set_title("{},{}".format(-(N//2)+i,-(N//2)+j))
            # Major ticks
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([])
            #ax.set_yticks([originy])
            ax.set_yticklabels([])
            ax.grid(which='major', color='w', linestyle='-', linewidth=0.5)
            ax.plot(centerx[n],centery[n],"r.")
            #ax.plot(originx,originy,"r.")
            
            
            
# ax.set_yticks(np.arange(0, 10, 1))

# # Labels for major ticks
# ax.set_xticklabels(np.arange(1, 11, 1))
# ax.set_yticklabels(np.arange(1, 11, 1))

# # Minor ticks
# ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
# ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

# # Gridlines based on minor ticks
# ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

# # Remove minor ticks
# ax.tick_params(which='minor', bottom=False, left=False)

    plt.show()
    
    