from astropy.io import fits
from astropy.table import Table,join
import numpy as np
import pylab as plt
import random
from scipy import stats
from sklearn.neighbors import KDTree
import time
from sklearn.metrics import mean_squared_error
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
from LLR import LLR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from LLR import LLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
import argparse

# which line to be used is set as an argument of the script. Line is important because there is a signal-to-noise cut on each line separately.
parser=argparse.ArgumentParser()
parser.add_argument('l', type=int)
args=parser.parse_args()

## this function is used first to turn magnitudes into features along with classifying the data based on EW[:,l]>class_cut
def features_and_outcomes_classification(x_in, y_in, ivar, n_out, class_cut, l):
    EWs = np.zeros([n_out,len(lines)])
    for i in range(n_out):
        for j in range(len(lines)):
            EWs[i,j] = y_in[i][j]
    
    x_in = x_in[:n_out,:]
    ones = np.ones([n_out,1])
    scalar = StandardScaler()
    x_out = np.zeros([n_out,x_in.shape[1]-1])
    for j in range(x_in.shape[1]-1):
        x_out[:,j] = x_in[:,j] - x_in[:,j+1]
    x_out = scalar.fit_transform(x_out)
    
    if (m == 0 or m == 5):
        x_out = np.concatenate((ones,x_out), axis=1)
        
    select_yes = (EWs[:,l]>=class_cut)
    
    y_out = np.zeros(n_out)
    y_out[select_yes] = 1
        
    ivar = ivar[:n_out]
    return x_out, y_out, ivar, EWs

## this function is used second to separate the data after classifying them. The output from this is used to predict EWs.
def features_and_outcomes_prediction(x_in, y_in, EWs, ivar, loga):
    p = (y_in==1)
    n = (y_in==0)
    n_p = len(np.where(p)[0])
    n_n = len(np.where(n)[0])
    
    x_out_p = x_in[p,:]
    x_out_n = x_in[n,:]
        
    if loga:
        y_out_p = np.log10(EWs[p,l])
        y_out_n = np.log10(EWs[n,l])
    else:
        y_out_p = EWs[p,l]
        y_out_n = EWs[n,l]
        
    ivar_out_p = ivar[p]
    ivar_out_n = ivar[n]
    
    return x_out_p, x_out_n, y_out_p, y_out_n, ivar_out_p, ivar_out_n

## predicting EWs
def predict(x, y, x_test, m):   
    # 0 is LLR with 100 neighbors, 1 is RandomForest, 2 is GradientBoosting from sklearn, 3 is XGboost, 4 is neural network, 5 is LLR with 800 neighbors, 6 is GP
    if m == 0:
        y_fit, zeros = LLR.LLR(x_test, x, y, 100, 'inverse_distance')
    if m == 1:
        model = RandomForestRegressor(n_estimators=200)
        model.fit(x_train, EW_train)
        EW_fit = model.predict(x_valid)
    if m == 2:
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(x_train, EW_train)
        EW_fit = model.predict(x_valid)
    if m == 3:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
        model.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_valid,y_valid)], verbose=False)
        y_fit = model.predict(x_test)
        zeros = []
        print(model.best_ntree_limit)
    if m == 4:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        
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
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0, validation_data=(x_valid, y_valid))
        y_fit = model.predict(x_test).reshape(-1)
        zeros = []

    if m == 5:
        y_fit, zeros = LLR.LLR_slow(x_test, x, y, 800, 'inverse_distance')

    if m ==6:
        kernel = DotProduct()
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(x, EW)
        EW_fit = gpr.predict(x_test)

    return y_fit, zeros

server = 1 # 0 is perlmutter, 1 is cori
server_paths = ['/pscratch/sd/a/ashodkh/', '/global/cscratch1/sd/ashodkh/']

#lines to predict
lines = ["OII_DOUBLET_EW","HGAMMA_EW","HBETA_EW","OIII_4959_EW","OIII_5007_EW","NII_6548_EW","HALPHA_EW"\
         ,"NII_6584_EW","SII_6716_EW","SII_6731_EW"]

# magnitudes to use as features
magnitude_names = ["ABSMAG_SDSS_U", "ABSMAG_SDSS_G", "ABSMAG_SDSS_R", "ABSMAG_SDSS_I", "ABSMAG_SDSS_Z", 'ABSMAG_W1']
 
#parameters
N = 6 ## this is irrelevant now but before it was used to make this data identical with ones from bins. The difference is negligible so I'm not doing it anymore.
run_train = 0
run_test = 1          
n_train = 25*10**3
n_test = 20*10**3
l = args.l
magnitudes_train = np.zeros([n_train, len(magnitude_names)])
magnitudes_test = np.zeros([n_test, len(magnitude_names)])
for i in range(len(magnitude_names)):
    magnitudes_train[:,i] = np.load(server_paths[server]+"target_selection/" + magnitude_names[i] + "_selection" + str(run_train) + "_" + str(lines[l]) + ".txt.npz")['arr_0']
    magnitudes_test[:,i] = np.load(server_paths[server]+"target_selection/" + magnitude_names[i] + "_selection" + str(run_test) + "_test.txt.npz")['arr_0']

zs_train = np.load(server_paths[server] + "target_selection/zs_selection" + str(run_train) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
target_lines_train = np.load(server_paths[server] + "target_selection/line_ews_selection" + str(run_train) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
line_ivars_train = np.load(server_paths[server] + "target_selection/line_ivars_selection" + str(run_train) + "_" + str(lines[l]) + ".txt.npz")["arr_0"]
zs_test = np.load(server_paths[server] + "target_selection/zs_selection" + str(run_test) + "_test.txt.npz")["arr_0"]
target_lines_test = np.load(server_paths[server] + "target_selection/line_ews_selection" + str(run_test) + "_test.txt.npz")["arr_0"]
line_ivars_test = np.load(server_paths[server] + "target_selection/line_ivars_selection" + str(run_test) + "_test.txt.npz")["arr_0"]

n_train = 23*10**3
n_test = 18*10**3

m = 3 # this is the model used for classification. XGBoost performed best
x, y, ivar, EW = features_and_outcomes_classification(magnitudes_train, target_lines_train, line_ivars_train, int(n_train), 5, 6)
x_test, y_test, ivar_test, EW_test = features_and_outcomes_classification(magnitudes_test, target_lines_test, line_ivars_test, int(n_test), 5, 6)

# Predicting classification. The result is a number between (roughly) 0 and 1, with a threshold that selects yes and no. True positive rate is not very sensitive to this threshold, but 
# false positive rate is, so I chose 0.7 threshold, which gives false positive rate ~27%
y_fit, zeros = predict(x, y, x_test, m)


## after classifying, separate the data and predict EWs. The "no" class is given 0 EWs.
y_fit_01 = np.zeros(n_test)
y_fit_01[y_fit>0.7] = 1
x_p, x_n, EW_p, EW_n, ivar_p, ivar_n = features_and_outcomes_prediction(x, y, EW, ivar, loga=True)
x_test_p, x_test_n, EW_test_p, EW_test_n, ivar_test_p, ivar_test_n = features_and_outcomes_prediction(x_test, y_fit_01, EW_test, ivar_test, loga=False)

m_j = [0, 5] # this is the model used for prediction after classification

for m in m_j:
    EW_fit_p, zeros = predict(np.concatenate((x_p, x_n)), np.concatenate((EW_p, EW_n)), x_test_p, m)
    EW_fit_n = np.zeros(len(EW_test_n))


    ## for output data is combined by concatenating predictions with 0 class to predictions with 1 class.
    EW_fit_all = np.concatenate((EW_fit_n, 10**EW_fit_p))
    EW_test_all = np.concatenate((EW_test_n, EW_test_p))
    ivar_test_all = np.concatenate((ivar_test_n, ivar_test_p))

    spearman = stats.spearmanr(EW_fit_all, EW_test_all)[0]
    print(spearman)

    np.savez_compressed(server_paths[server] + "ew_results_classification/ugriz/m" +str(m)+ "/classification_test_logEW_fit_ugriz_selection"+str(run_train)+"_line"+str(lines[l])+"_bins"+str(N)\
                            +"_ML"+str(m)+".txt", EW_fit_all)
    np.savez_compressed(server_paths[server] + "ew_results_classification/ugriz/m" +str(m)+ "/classification_test_EW_obs_ugriz_selection"+str(run_train)+"_line"+str(lines[l])+"_bins"+str(N)\
                            +"_ML"+str(m)+".txt", EW_test_all)
    np.savez_compressed(server_paths[server] + "ew_results_classification/ugriz/m" +str(m)+ "/classification_test_line_ivars_ugriz_selection"+str(run_train)+"_line"+str(lines[l])+"_bins"+str(N)\
                            +"_ML"+str(m)+".txt", ivar_test_all)