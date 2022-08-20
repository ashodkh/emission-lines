perlmutter = True
cori = False
from astropy.io import fits
from astropy.table import Table
import numpy as np
import pylab as plt
import random
from scipy import stats, signal
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from astropy.cosmology import FlatLambdaCDM
from os import listdir
import speclite.filters
import scipy
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()

# parameters
n = 30*10**3    # number of initial data points
nw = 7781       # length of wavelength vector
run = 1         # run is to keep track of which selection

l = args.l
fastspec = True
fastphot = not(fastspec)

lines = ["OII_DOUBLET_EW", "HGAMMA_EW", "HBETA_EW", "OIII_4959_EW", "OIII_5007_EW", "NII_6548_EW", "HALPHA_EW", "NII_6584_EW", "SII_6716_EW", "SII_6731_EW", "test"]

server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']

zs_all = np.load(server_paths[server] + "target_selection/zs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]

## I am limiting spectra to 10k at a time for memory issues. decades = number of 10k spectra. so decades = 3 is 30,000, stored in separate 10k files
decades = 3

for k in range(decades):
    n = 10*10**3
    # if k == 2:
    #     n = 5*10**3
    print(k)

    if fastspec:
        spectra = np.load(server_paths[server] + "spectra_from_targets/fastspec/fastspec_spectra" +str(k)+ "_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]
    elif fastphot:
        spectra = np.load(server_paths[server] + "spectra_from_targets/fastphot/fastphot_spectra" +str(k)+ "_selection"+str(run)+"_"+str(lines[l])+".txt.npz")["arr_0"]


    #wavelengths=np.load("/global/cscratch1/sd/ashodkh/results/raw_data_wavelengths.txt.npz")["arr_0"]
    wavelengths = np.arange(3600, 9824+.8, .8) 
    zs = zs_all[k*n:(k+1)*n]
    
    nw = len(wavelengths)
    d = np.average(.8/(1+zs))

    # setting up wavelength bins to calculate fluxes in them
    w1, w2 = 3400, 7000
    big_bin=np.arange(w1,w2,d)

    Ns = [6, 11, 16, 21, 26, 31, 41, 51]
    #Ns = [31, 41, 51]
    for N in Ns:
        print(N)
        bin_ws = np.linspace(w1,w2,N)
        small_bins = []
        for i in range(N-1):
            small_bins.append(np.arange(bin_ws[i],bin_ws[i+1],d))
            
        # calculating fluxes in bins for all the spectra
        c=3*10**18
        fluxes_bin=np.zeros([n,N-1])

        for i in range(n):
            rest_waves = wavelengths/(1+zs[i])
            for j in range(N-1):  # using nn interpolation to get spectrum on bin grids and then calculating flux
                tree = KDTree(rest_waves.reshape(-1,1))
                dist, ind = tree.query(small_bins[j].reshape(-1,1),k=1)
                spectra_bin = spectra[i,ind].reshape(-1)
                nans = np.where(np.isnan(spectra_bin[:]))[0]
                av = np.average(spectra_bin[spectra_bin==spectra_bin])
                spectra_bin[np.where(np.isnan(spectra_bin))[0]] = av
                fluxes_bin[i,j] = np.trapz(spectra_bin*small_bins[j]*(1+zs[i]),small_bins[j])\
                                /np.trapz(c/small_bins[j],small_bins[j])
        
        if fastspec:
            np.savez_compressed(server_paths[server] + "fluxes_from_spectra/fastspec/fluxes_fastspec" +str(k)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt", fluxes_bin)
        elif fastphot:
            np.savez_compressed(server_paths[server] + "fluxes_from_spectra/fastphot/fluxes_fastphot" +str(k)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt", fluxes_bin)
