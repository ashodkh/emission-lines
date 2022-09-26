perlmutter = True
cori = False
from astropy.io import fits
from astropy.table import Table, join
import numpy as np
import pylab as plt
import random
from scipy import stats,signal
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from astropy.cosmology import FlatLambdaCDM
from os import listdir
import desispec
import speclite.filters
import scipy
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from desitarget.cmx.cmx_targetmask import cmx_mask
from sedpy.observate import load_filters
import prospect
import fitsio
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from desiutil.dust import ext_fitzpatrick
from scipy.ndimage import gaussian_filter1d
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()
cosmo=FlatLambdaCDM(H0=70, Om0=0.3)

lines=["OII_DOUBLET_EW", "HGAMMA_EW", "HBETA_EW", "OIII_4959_EW", "OIII_5007_EW", "NII_6548_EW", "HALPHA_EW", "NII_6584_EW", "SII_6716_EW", "SII_6731_EW", "test"]
run = 2
l = args.l
sv = '1'
fastspec = True
fastphot = not(fastspec)

server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']

zs = np.load(server_paths[server] + "target_selection/sv" + sv + "_zs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
if fastspec:
    coeffs = np.load(server_paths[server] + "target_selection/sv" + sv + "_fastspec_coeffs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
    AV = np.load(server_paths[server] + "target_selection/sv" + sv + "_fastspec_AV_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
elif fastphot:
    coeffs = np.load(server_paths[server] + "target_selection/sv" + sv + "_fastphot_coeffs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
    AV = np.load(server_paths[server] + "target_selection/sv" + sv + "_fastphot_AV_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
        
file_path = "/global/cfs/cdirs/desi/science/gqp/templates/SSP-CKC14z/v1.0/SSP_Padova_CKC14z_Kroupa_Z0.0190.fits"
templates = fits.open(file_path)
sspinfo = Table(fitsio.read(file_path, ext="METADATA"))
myages = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.9, 1.1, 1.4, 2.5, 5, 10.0, 13.3])*1e9
iage = np.array([np.argmin(np.abs(sspinfo['age']-myage)) for myage in myages])
template_waves, wavehdr = fitsio.read(file_path, ext='wave', header=True)

w1_z, w2_z = 3600, 9824
d = .8
wavelength = np.arange(w1_z, w2_z+d, d) # wavelength array of observed frame. Set same as DESI.
template_fluxes = fitsio.read(file_path, ext='flux')[:,iage]

## choosing short wavelength ranges so that code runs faster, because default sizes are huge
w1_needed, w2_needed = 3400, 9824 # wavelength range needed for rest-frame interval
ind1, ind2 = np.argmin(np.abs(template_waves-w1_needed)), np.argmin(np.abs(template_waves-w2_needed))
template_waves2 = template_waves[ind1:ind2]
template_fluxes2 = template_fluxes[ind1:ind2,:]

fluxnorm = 1e17 # normalization factor for the spectra
massnorm = 1e10 # stellar mass normalization factor for the SSPs [Msun]

## I am limiting spectra to 10k at a time for memory issues. decades = number of 10k spectra. so decades = 3 is 30,000, stored in separate 10k files
decades = 3
n_decades = [10*10**3, 10*10**3, 5*10**3]
for j in range(decades):
    n = n_decades[j]
    spectra = np.zeros([n, len(wavelength)])
    tic = time.time()
    for i in range(n):
        k = i + j*n_decades[j-1] # this index is 0-10k for i=0 and 10k-20k for i=1 etc...

        redshift = zs[k]
        rest_wavelength = wavelength/(1+redshift)
        tree = KDTree(template_waves2.reshape(-1,1))
        dist, ind = tree.query(rest_wavelength.reshape(-1,1), k=1)
        template_waves_short = template_waves2[ind].reshape(-1)

        dfactor = (10.0 / cosmo.luminosity_distance(redshift).to(u.pc).value)**2
        atten = (template_waves_short / 5500.0)**(-.7)
        atten = 10**(-0.4*AV[k]*atten)

        spectra[i,:] = np.matmul(template_fluxes2, coeffs[k])[ind].reshape(-1)*fluxnorm*massnorm*dfactor*atten
    print(time.time() - tic)


    if fastspec:
        np.savez_compressed(server_paths[server] + "spectra_from_targets/fastspec/sv" + sv + "_fastspec_spectra" +str(j)+ "_selection"+str(run)+"_"+str(lines[l])+".txt", spectra)
    elif fastphot:
        np.savez_compressed(server_paths[server] + "spectra_from_targets/fastphot/sv" + sv + "_fastphot_spectra" +str(j)+ "_selection"+str(run)+"_"+str(lines[l])+".txt", spectra)
