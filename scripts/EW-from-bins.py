import numpy as np
import pylab as plt
from scipy import stats, signal
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from os import listdir
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow import keras
from tensorflow.keras import layers
from LLR import LLR
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()


## function that transforms fluxes into colors and also selects only data that have positive fluxes in all bins.
def features_and_outcomes(x_in, y_in, ivar, n_out, loga):
    magnitudes = np.zeros([n_out,x_in.shape[1]])
    EWs = np.zeros([n_out,len(lines)])
    
    select_fluxes = x_in[:,0]>0
    for i in range(1, x_in.shape[1]):
        select_fluxes = select_fluxes*(x_in[:,i]>0)
    
    x_in = x_in[select_fluxes,:]
    y_in = y_in[select_fluxes]
    ivar = ivar[select_fluxes]
    
    for i in range(n_out):
        magnitudes[i,:] = -2.5*np.log10(x_in[i,:])
        for j in range(len(lines)):
            EWs[i,j] = y_in[i][j]
    
    ones = np.ones([n_out,1])
    scalar = StandardScaler()
    x_out = np.zeros([n_out,x_in.shape[1]-1])
    for j in range(x_in.shape[1]-1):
        x_out[:,j] = magnitudes[:,j] - magnitudes[:,j+1]
    x_out = scalar.fit_transform(x_out)
    
    if (m == 0 or m == 5) or m == 6:
        x_out = np.concatenate((ones,x_out), axis=1)
        
    if loga:
        y_out = np.log10(EWs[:,l])
    else:
        y_out = EWs[:,l]
        
    return x_out, y_out, ivar

## reading fluxes and equivalent widths
lines = ["OII_DOUBLET_EW", "HGAMMA_EW", "HBETA_EW", "OIII_4959_EW", "OIII_5007_EW", "NII_6548_EW",\
         "HALPHA_EW", "NII_6584_EW", "SII_6716_EW", "SII_6731_EW"]
l = args.l
l_test = 'test' # sv3 testing set is labeled as l=10 which corresponds to "test" in lines
run = 0
m = 5 
#loga = False

data = 3 # 0 is raw_masked, 1 is raw_unmasked, 2 is fastspec, 3 is fastphot
data_file_names = ['raw_masked', 'raw_unmasked', 'fastspec', 'fastphot']
data_flux_names = ['fluxes_raw_masked', 'fluxes_raw_unmasked', 'fluxes_fastspec', 'fluxes_fastphot']

Ns = [6, 11, 16, 21, 26, 31, 41, 51]
decades = 3 ## number of 10k galaxy files I want to load and combine
n_decades = [10*10**3, 10*10**3, 5*10**3]
server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']
for N in Ns:
    fluxes_bin = np.zeros([25*10**3, N-1]) ## fluxes are separated into groups of 10k galaxies
    for i in range(decades):
        n = n_decades[i]
        if i == 2:
            fluxes_bin[10**4*i:25*10**3,:] =  np.load(server_paths[server] + "fluxes_from_spectra/" + data_file_names[data] + "/" + data_flux_names[data]\
                                                +str(i)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt.npz")["arr_0"]
        else:
            fluxes_bin[10**4*i:n*(i+1),:] = np.load(server_paths[server] + "fluxes_from_spectra/" + data_file_names[data] + "/" + data_flux_names[data]\
                                            +str(i)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt.npz")["arr_0"]

    zs = np.load(server_paths[server] + "target_selection/zs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
    target_lines = np.load(server_paths[server] + "target_selection/line_ews_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
    line_ivars = np.load(server_paths[server] + "target_selection/line_ivars_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]

    x, EW, line_ivars = features_and_outcomes(fluxes_bin, target_lines, line_ivars, 23*10**3, loga=True)
    
    N_cv = 10
    print(x.shape)
    x_split = np.split(x,N_cv)
    EW_split = np.split(EW,N_cv)

    EW_fit_all = []
    EW_obs_all = []

    spearman_all = []
    rms_all = []
    nmad_all = []
    nmad2_all = []
    for i in range(N_cv):
        ## assigning the training and validation sets
        x_valid = x_split[i]
        EW_valid = EW_split[i]

        x_to_combine = []
        EW_to_combine = []
        for j in range(N_cv):
            if j != i:
                x_to_combine.append(x_split[j])
                EW_to_combine.append(EW_split[j])
        x_train=np.concatenate(tuple(x_to_combine),axis=0)
        EW_train=np.concatenate(tuple(EW_to_combine),axis=0)

        # predicting EWs using different models
        if m == 0:
            EW_fit,zeros = LLR.LLR(x_valid, x_train, EW_train, 100, 'inverse_distance')
        if m == 1:
            model = RandomForestRegressor(n_estimators=200)
            model.fit(x_train, EW_train)
            EW_fit = model.predict(x_valid)
        if m == 2:
            model = GradientBoostingRegressor(n_estimators=100)
            model.fit(x_train, EW_train)
            EW_fit = model.predict(x_valid)
        if m == 3:
            model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
            model.fit(x_train, EW_train, early_stopping_rounds=5, eval_set=[(x_valid,EW_valid)], verbose=False)
            EW_fit = model.predict(x_valid)
            print(model.best_ntree_limit)
        if m == 4:
            model_input = layers.Input(shape=x.shape[1])
            h1 = layers.Dense(units=100, kernel_initializer="he_normal")(model_input)
            a1 = layers.PReLU()(h1)
            h2 = layers.Dense(units=100, kernel_initializer="he_normal")(a1)
            a2 = layers.PReLU()(h2)
            h3 = layers.Dense(units=100, kernel_initializer="he_normal")(a2)
            a3 = layers.PReLU()(h3)
            output_layer = layers.Dense(1, activation='linear')(a3)
            model = keras.models.Model(inputs=model_input, outputs=output_layer)

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics='mse')

            n_epochs = 100
            batch_size = 100
            history = model.fit(x_train, EW_train, batch_size=batch_size, epochs=n_epochs, verbose=0, validation_data=(x_valid, EW_valid))
            EW_fit = model.predict(x_valid)
            
        if m == 5:
            EW_fit,zeros = LLR.LLR_slow(x_valid, x_train, EW_train, 800, 'inverse_distance')

        if m == 6:
            EW_fit,zeros = LLR.LLR(x_valid, x_train, EW_train, 600, 'inverse_distance')
        # calculating spearman coefficient and nmad for fit. nmad2 has the error in it.
        nmad = np.abs(EW_fit-EW_valid)
        nmad2 = np.abs(EW_fit-EW_valid)

        EW_fit_all.append(EW_fit)
        EW_obs_all.append(EW_valid)

        spearman_all.append(stats.spearmanr(EW_fit,EW_valid)[0])
        rms_all.append(np.sqrt(mean_squared_error(EW_fit,EW_valid)))
        nmad_all.append(1.48*np.median(nmad))
        nmad2_all.append(1.48*np.median(nmad2))

    print(lines[l])
    print(spearman_all)
    print('spearman_average= '+str(np.average(spearman_all)))
    # print(rms_all)
    # print(np.average(rms_all))
    print(nmad_all)
    print('nmad_average= '+str(np.average(nmad_all)))
    print("\n")

    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/logEW_fit_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_fit_all)
    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/logEW_obs_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_obs_all)
    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/line_ivars_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", line_ivars)