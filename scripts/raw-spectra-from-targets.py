perlmutter = True
cori = False
from astropy.io import fits
from astropy.table import Table, join
import numpy as np
import pylab as plt
import random
from scipy import stats
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from astropy.cosmology import FlatLambdaCDM
from os import listdir
import scipy
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.cmx.cmx_targetmask import cmx_mask
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()

lines=["OII_DOUBLET_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW","HALPHA_EW","NII_6584_EW","SII_6716_EW","SII_6731_EW", "test"]
n = 30*10**3
run = 0
l = args.l

server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']

target_ids = np.load(server_paths[server] + "target_selection/target_ids_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
fiber_ids = np.load(server_paths[server] + "target_selection/fiber_ids_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
petal_locs = np.load(server_paths[server] + "target_selection/petal_locs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
tile_ids = np.load(server_paths[server] + "target_selection/tile_ids_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']  
zs = np.load(server_paths[server] + "target_selection/zs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']

## getting spectra for n points by inverse variance weighting fluxes.
## I am limiting spectra to 10k at a time for memory issues. decades = number of 10k spectra. so decades = 3 is 30,000, stored in separate 10k files
nw = 7781
decades = 1

for i in range(decades):
    i = 2
    n = 5*10**3
    # if i == 2:
    #     n = 5*10**3
    spectra = np.zeros([n,nw])
    tic = time.time()
    for j in range(n):
        k = j + 20*10**3 # this index is 0-10k for i=0 and 10k-20k for i=1 etc...
        coadd_path = "/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative/"+str(tile_ids[k])
        a = listdir(coadd_path)[0]
        coadd_path = "/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative/"+str(tile_ids[k])+"/"+a
        coadd_path = "/global/cfs/cdirs/desi/spectro/redux/fuji/tiles/cumulative/"+str(tile_ids[k])+"/"+a\
                     +"/coadd-"+str(petal_locs[k])+"-"+str(tile_ids[k])+"-thru"+str(a)+".fits"

        test = fits.open(coadd_path)

        if test[1].data.field(0)[fiber_ids[k]%500] != target_ids[k]:
            raise ValueError('Target ids dont match')

        wave_b = test["B_WAVELENGTH"].data
        flux_b = test["B_FLUX"].data[fiber_ids[k]%500,:]
        inv_var_b = test["B_IVAR"].data[fiber_ids[k]%500,:]
        wave_r = test["R_WAVELENGTH"].data
        flux_r = test["R_FLUX"].data[fiber_ids[k]%500,:]
        inv_var_r = test["R_IVAR"].data[fiber_ids[k]%500,:]
        wave_z = test["Z_WAVELENGTH"].data
        flux_z = test["Z_FLUX"].data[fiber_ids[k]%500,:]
        inv_var_z = test["Z_IVAR"].data[fiber_ids[k]%500,:]

        first = np.where(abs(wave_b-wave_r[0])<10**-4)[0][0]
        second = np.where(abs(wave_r-wave_z[0])<10**-4)[0][0]

        nw = first + second + len(wave_z)

        spectrum = np.zeros(nw)

        i2 = np.arange(nw)-first
        i3 = np.arange(nw)-(second+first)

    #     wavelength=np.zeros(nw)

    #     wavelength[:first]=wave_b[:first]
    #     wavelength[first:len(wave_b)]=(wave_b[first:len(wave_b)]+wave_r[i2[first:len(wave_b)]])/2
    #     wavelength[len(wave_b):second+first]=wave_r[i2[len(wave_b):second+first]]
    #     wavelength[second+first:first+len(wave_r)]=(wave_r[i2[second+first:first+len(wave_r)]]+wave_z[i3[second+first:first+len(wave_r)]])/2
    #     wavelength[first+len(wave_r):] = wave_z[i3[first+len(wave_r):]]

        spectrum[:first] = flux_b[:first]  
        spectrum[first:len(wave_b)] = (flux_b[first:len(wave_b)]*inv_var_b[first:len(wave_b)]+flux_r[i2[first:len(wave_b)]]*inv_var_r[i2[first:len(wave_b)]])\
                                    /(inv_var_b[first:len(wave_b)]+inv_var_r[i2[first:len(wave_b)]])
        spectrum[len(wave_b):second+first] = flux_r[i2[len(wave_b):second+first]]
        spectrum[second+first:first+len(wave_r)] = (flux_r[i2[second+first:first+len(wave_r)]]*inv_var_r[i2[second+first:first+len(wave_r)]]+flux_z[i3[second+first:first+len(wave_r)]]*inv_var_z[i3[second+first:first+len(wave_r)]])\
                        /(inv_var_r[i2[second+first:first+len(wave_r)]]+inv_var_z[i3[second+first:first+len(wave_r)]])
        spectrum[first+len(wave_r):] = flux_z[i3[first+len(wave_r):]]

        spectra[j,:] = spectrum[:]
    print(time.time()-tic)

    wavelength = np.arange(3600, 9824+.8, .8)
    
    np.savez_compressed(server_paths[server] + "spectra_from_targets/raw/raw_spectra" +str(i)+ "_selection"+str(run)+"_"+str(lines[l])+".txt", spectra)
    #np.savez_compressed("/pscratch/sd/a/ashodkh/spectra_from_targets/raw/raw_data_wavelengths.txt", wavelength)