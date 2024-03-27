import numpy as np
import pylab as plt
from dsps import calc_rest_sed_sfh_table_lognormal_mdf, calc_rest_sed_sfh_table_met_table
from dsps.cosmology import age_at_z, DEFAULT_COSMOLOGY


def generate_spectra_from_params(sfhs=None, mhs=None, cosmic_time=None, mh_scatter=None, zs=None, templates=None, t_obs=None, w1=None, w2=None):
    n = sfhs.shape[0]
    i1,i2 = np.searchsorted(templates.ssp_wave, (w1,w2))
    wavelengths = templates.ssp_wave[i1:i2]
    nw = len(wavelengths)
    spectra = np.zeros([n,nw])
    
    for i in range(n):
        if t_obs is None:
            t_obs = age_at_z(zs[i], *DEFAULT_COSMOLOGY)
            t_obs = t_obs[0]
        if len(mhs.shape)<2:
            sed_info = calc_rest_sed_sfh_table_lognormal_mdf(cosmic_time[i,:], sfhs[i,:], mhs[i], mh_scatter[i],\
                                                             templates.ssp_lgmet, templates.ssp_lg_age_gyr, templates.ssp_flux, t_obs)
        else:
            sed_info = calc_rest_sed_sfh_table_met_table(cosmic_time[i,:], sfhs[i,:], mhs[i,:], mh_scatter[i],\
                                                             templates.ssp_lgmet, templates.ssp_lg_age_gyr, templates.ssp_flux, t_obs)

        spectra[i,:] = sed_info.rest_sed[i1:i2]

    return spectra, wavelengths

def mask_emission_lines(spectra=None, wavelengths=None, line_wavelengths=None, line_windows=None):
    n = spectra.shape[0]
    masked_spectra = np.zeros_like(spectra)
    for i in range(n):
        masked_spectra[i,:] = spectra[i,:]
        for l in range(len(line_wavelengths)):
            i1, i2 = np.searchsorted(wavelengths, (line_wavelengths[l]-line_windows[l], line_wavelengths[l]+line_windows[l]))
            x0, x1 = wavelengths[i1-1], wavelengths[i2+1]
            y0, y1 = spectra[i,i1-1], spectra[i,i2+1]
            masked_spectra[i,i1:i2] = np.interp(wavelengths[i1:i2], [x0,x1], [y0,y1])
            if np.average(masked_spectra[i,i1:i2]) > np.average(spectra[i,i1:i2]):
                masked_spectra[i,i1:i2] = spectra[i,i1:i2]

    return masked_spectra

def calculate_ews(spectra=None, continua=None, wavelengths=None, line_wavelengths=None, line_windows=None):
    n = spectra.shape[0]
    nl = len(line_wavelengths)
    line_ews = np.zeros([n,nl])
    for i in range(n):
        for l in range(nl):
            i1,i2 = np.searchsorted(wavelengths, (line_wavelengths[l]-line_windows[l], line_wavelengths[l]+line_windows[l]))
            line_ews[i,l] = np.trapz((spectra[i,i1:i2]-continua[i,i1:i2])/continua[i,i1:i2], wavelengths[i1:i2])

    return line_ews

def calculate_top_hat_fluxes(spectra=None, wavelength=None, N=None):
    w1, w2 = wavelength[0], wavelength[-1]
    bin_edges = np.linspace(w1, w2, N)
    bin_edge_indices = np.searchsorted(wavelength, bin_edges)

    n = spectra.shape[0]
    bin_fluxes = np.zeros([n,N-1])
    for i in range(n):
        for j in range(N-1):
            i1, i2 = bin_edge_indices[j], bin_edge_indices[j+1]
            bin_fluxes[i,j] = np.average(spectra[i,i1:i2])
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.
    return bin_fluxes, bin_centers

def calculate_colors(bin_fluxes):
    colors = np.zeros([bin_fluxes.shape[0],bin_fluxes.shape[1]-1])
    for i in range(bin_fluxes.shape[0]):
        for j in range(bin_fluxes.shape[1]-1):
            colors[i,j] = -1*np.log10(bin_fluxes[i,j]/bin_fluxes[i,j+1])

    return colors

def calculate_ab_magnitudes(spectrum=None, spectrum_wavelength=None, filter_response=None, filter_wavelength=None):
    c = 2.998e18
    padded_spectrum = np.interp(filter_wavelength, spectrum_wavelength, spectrum)
    num = np.trapz(padded_spectrum*filter_response*filter_wavelength, filter_wavelength)
    denom = np.trapz(3631*10**(-23)*c/filter_wavelength**2*filter_response*filter_wavelength, filter_wavelength)
    return -2.5*np.log10(num/denom)