# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:09:36 2020

@author: Nils Olav Handegard, Alba Ordonez, Rune Øyerhamn
"""
# Regrid
import numpy as np
import matplotlib.pyplot as plt


def resampleWeight(r_t, r_s):
    """
    The regridding is a linear combination of the inputs based
    on the fraction of the source bins to the range bins.
    See the different cases below
    """

    # Create target bins from target range
    bin_r_t = np.append(r_t[0]-(r_t[1] - r_t[0])/2, (r_t[0:-1] + r_t[1:])/2)
    bin_r_t = np.append(bin_r_t, r_t[-1]+(r_t[-1] - r_t[-2])/2)

    # Create source bins from source range
    bin_r_s = np.append(r_s[0]-(r_s[1] - r_s[0])/2, (r_s[0:-1] + r_s[1:])/2)
    bin_r_s = np.append(bin_r_s, r_s[-1]+(r_s[-1] - r_s[-2])/2)

    # Initialize W matrix (sparse)
    W = np.zeros([len(r_t), len(r_s)+1])
    # NB: + 1 length for space to NaNs in edge case

    # Loop over the target bins
    for i, rt in enumerate(r_t):
    
        # Check that this is not an edge case
        if bin_r_t[i] > bin_r_s[0] and bin_r_t[i+1] < bin_r_s[-1]:
            # The size of the target bin
            # example target bin:  --[---[---[---[-
            drt = bin_r_t[i+1] - bin_r_t[i]  # From example: drt = 4
        
            # find the indices in source
            j0 = np.searchsorted(bin_r_s, bin_r_t[i], side='right')-1
            j1 = np.searchsorted(bin_r_s, bin_r_t[i+1], side='right')
        
            # CASE 1: Target higher resolution, overlapping 1 source bin
            # target idx     i    i+1
            # target    -----[-----[-----
            # source    --[-----------[--
            # source idx  j0          j1
        
            if j1-j0 == 1:
                W[i, j0] = 1
                
            # CASE 2: Target higher resolution, overlapping 1 source bin
            # target idx      i   i+1
            # target    --[---[---[---[-
            # source    -[------[------[-
            # source idx j0            j1
        
            elif j1-j0 == 2:
                W[i, j0] = (bin_r_s[j0+1]-bin_r_t[i])/drt
                W[i, j1-1] = (bin_r_t[i+1]-bin_r_s[j1-1])/drt
                
            # CASE 3: Target lower resolution
            # target idx    i       i+1
            # target    ----[-------[----
            # source    --[---[---[---[--
            # source idx  j0          j1
                
            elif j1-j0 > 2:
                for j in range(j0, j1):
                    if j == j0:
                        W[i, j] = (bin_r_s[j+1]-bin_r_t[i])/drt
                    elif j == j1-1:
                        W[i, j] = (bin_r_t[i+1]-bin_r_s[j])/drt
                    else:
                        W[i, j] = (bin_r_s[j+1]-bin_r_s[j])/drt
                    
        #  Edge case 1
        # target idx    i       i+1
        # target    ----[-------[----
        # source        #end# [---[---[
        # source idx          j0  j1
                
        #  Edge case 2
        # target idx    i       i+1
        # target    ----[-------[----
        # source    --[---[ #end#
        # source idx  j0  j1
        else:
            # Edge case (NaN must be in W, not in sv_s.
            # Or else np.dot failed)
            W[i, -1] = np.nan
    return W


def regrid(sv_s, W):
    """
    Use the weights to regrid the sv data
    """
    # Add a row of at the bottom to be used in edge cases
    sv_s_mod = np.vstack((sv_s, np.zeros(N_pings)))
    # Do the dot product
    return np.dot(W, sv_s_mod)


# Test data
N_pings = 43
# Target range
r_t = np.linspace(1, 20, num=50)
# Source range
r_s = np.linspace(4, 16,  num=40)
# Test  source sv
sv_s = np.ones((len(r_s), N_pings))

# Create resampleWeights from source range (r_s) to target range (r_t)
W = resampleWeight(r_t, r_s)
# Regrid from source (sv_s) to target sv (sv_t)
sv_t = regrid(sv_s, W)

# Sanity checks
print('W:')
print(W)
print('Anything other than 0 indicates errors (nan is outside source range):')
print(np.sum(W, axis=1)-1)

# Sanity plot
fig, ax = plt.subplots(2, 1)
ax[0].plot(r_s, sv_s, '-*', label='source')
ax[0].plot(r_t, sv_t, '-*', label='target')
ax[0].legend()

ax[1].plot(1-np.sum(W, axis=1), '-*', label='errors')
ax[1].legend()
plt.show()
