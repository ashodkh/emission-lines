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
from sklearn.model_selection import train_test_split
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
def features_and_outcomes(x_in, y_in, n_out, ivar, loga):
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
run_test = 1
m = 3 
#loga = False

data = 0 # 0 is raw_masked, 1 is raw_unmasked, 2 is fastspec, 3 is fastphot
data_file_names = ['raw_masked', 'raw_unmasked', 'fastspec', 'fastphot']
data_flux_names = ['fluxes', 'fluxes', 'fluxes_fastspec', 'fluxes_fastphot']

Ns = [6, 11, 16, 21, 26, 31, 41, 51]
decades = 3 ## number of 10k galaxy files I want to load and combine
n_decades = [10*10**3, 10*10**3, 5*10**3]
server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']
for N in Ns:
    n = 10*10**3
    fluxes_bin = np.zeros([25*10**3, N-1]) ## fluxes are separated into groups of 10k galaxies
    fluxes_bin_test = np.zeros([20*10**3, N-1])
    for i in range(decades):
        if i == 2:
            n = 5*10**3
            fluxes_bin[10**4*i:25*10**3,:] =  np.load(server_paths[server] + "fluxes_from_spectra/" + data_file_names[data] + "/" + data_flux_names[data]\
                                                +str(i)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt.npz")["arr_0"]
        else:
            fluxes_bin[10**4*i:n*(i+1),:] = np.load(server_paths[server] + "fluxes_from_spectra/" + data_file_names[data] + "/" + data_flux_names[data]\
                                            +str(i)+ "_selection"+str(run)+"_"+str(lines[l])+"_bins"+str(N)+".txt.npz")["arr_0"]
            fluxes_bin_test[10**4*i:n*(i+1),:] = np.load(server_paths[server] + "fluxes_from_spectra/" + data_file_names[data] + "/" + data_flux_names[data]\
                                            +str(i)+ "_selection"+str(run_test)+"_"+str(l_test)+"_bins"+str(N)+".txt.npz")["arr_0"]

    zs = np.load(server_paths[server] + "target_selection/zs_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
    target_lines = np.load(server_paths[server] + "target_selection/line_ews_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
    line_ivars = np.load(server_paths[server] + "target_selection/line_ivars_selection" + str(run) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
    zs_test = np.load(server_paths[server] + "target_selection/zs_selection" + str(run_test) + "_" + str(l_test) + ".txt.npz")["arr_0"][0:20*10**3]
    target_lines_test = np.load(server_paths[server] + "target_selection/line_ews_selection" + str(run_test) + "_" + str(l_test) + ".txt.npz")["arr_0"][0:20*10**3]
    line_ivars_test = np.load(server_paths[server] + "target_selection/line_ivars_selection" + str(run_test) + "_" + str(l_test) + ".txt.npz")["arr_0"][0:20*10**3]


    x, EW, line_ivars = features_and_outcomes(fluxes_bin, target_lines, 23*10**3, line_ivars, loga = True)
    
    x_test, EW_test, line_ivars_test = features_and_outcomes(fluxes_bin_test, target_lines_test, 19*10**3, line_ivars_test, loga = False)

    
    # predicting EWs using different models
    if m == 0:
        EW_fit,zeros = LLR.LLR(x_test, x, EW, 100, 'inverse_distance')
    if m == 1:
        model = RandomForestRegressor(n_estimators=200)
        model.fit(x_train, EW_train)
        EW_fit = model.predict(x_valid)
    if m == 2:
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(x_train, EW_train)
        EW_fit = model.predict(x_valid)
    if m == 3:
        x_train, x_valid, EW_train, EW_valid = train_test_split(x, EW, test_size=0.2)
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
        model.fit(x_train, EW_train, early_stopping_rounds=5, eval_set=[(x_valid,EW_valid)], verbose=False)
        EW_fit = model.predict(x_test)
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
        EW_fit,zeros = LLR.LLR_slow(x_test, x, EW, 800, 'inverse_distance')

    # calculating spearman coefficient and nmad for fit. nmad2 has the error in it.
    nmad = 1.48*np.median(np.abs(10**EW_fit-EW_test))
    spearman = stats.spearmanr(10**EW_fit,EW_test)[0]

    print('spearman = ' + str(spearman))
    # print(rms_all)
    # print(np.average(rms_all))
    print('nmad= ' + str(nmad))
    print("\n")

    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_logEW_fit_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_fit)
    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_EW_obs_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_test)
    np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_line_ivars_" + data_file_names[data] + "_selection" + str(run) + \
                                "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", line_ivars_test)
        
    # if loga:
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_logEW_fit_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_fit)
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_logEW_obs_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_test)
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_line_ivars_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", line_ivars_test)
    # else:
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_EW_fit_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_fit)
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_EW_obs_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", EW_test)
    #     np.savez_compressed(server_paths[server] + "ew_results/" + data_file_names[data] + "/m" + str(m) + "/test_line_ivars_" + data_file_names[data] + "_selection" + str(run) + \
    #                         "_line" + str(lines[l]) + "_bins" + str(N) + "_ML" + str(m) + ".txt", line_ivars_test)
            
