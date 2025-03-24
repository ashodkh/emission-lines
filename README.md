# Neural Network to Predict Emission Lines

This repo contains the weights of the JAX-implemented neural network from the paper [Emission Line Predictions for Mock Galaxy Catalogues: a New Differentiable and Empirical Mapping from DESI](https://academic.oup.com/mnras/article/531/1/1454/7665770).

I have trained a JAX-implemented neural network on DESI BGS data (https://www.desi.lbl.gov/2023/06/13/the-desi-early-data-release-in-now-available/) to predict optical emission lines of galaxies from the optical continuum. The model weights are in the model folder, and an example of how to use them is shown in the notebooks/tutorial-predicting-ews-from-fluxes.ipynb notebook.

