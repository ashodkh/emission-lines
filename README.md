# Neural Network to Predict Emission Lines

This repo contains the weights of the JAX-implemented neural network from the paper [Emission Line Predictions for Mock Galaxy Catalogues: a New Differentiable and Empirical Mapping from DESI](https://academic.oup.com/mnras/article/531/1/1454/7665770). It also contains test set data and notebooks to reproduce some of the plots in the paper.

# How to Use the Network
**Requirements:**
```
Python 3.11.3
JAX 0.4.19
```

The model weights are found in the models directory. There are two models: one takes as input only flux ratios, and the other one takes as input flux ratios and luminosity. The latter was used in the paper, but to use it for other data you should be careful about luminosities matching the BGS training sample (see figure 1 in paper). **To predict emission lines for a wider range of luminosities, it is best to use the model without luminosities so that the model does not extrapolate.** 

**Running the model:**  
Simply download the model weights and the script containing the model class (emission_line_model.py in the scripts directory). Then run

```
import emission_line_model as elm

model = elm.EmissionLineModel(path_to_model)
predicted_ews = model.predict(flux_ratios, luminosities) # luminosities=None if using model without luminosities.
```

**Calculating flux ratios:**  
As described in the paper, average fluxes in 12 top-hat bins are calculated from the spectrum. Inputs to the model are flux ratios between consecutive bins. 

![Screenshot 2025-03-24 at 1 16 07 PM](https://github.com/user-attachments/assets/6f1d6c85-7727-4c0d-a646-59240dea83ea)

spectra_utils.py in the scripts directory contains some utility functions which could be useful for calculating the fluxes and flux ratios, masking emission lines, and calculating equivalent widths.

Note that the DESI spectra which were used to train the network are in units of ```1e-17 erg / s / cm^2 / A```.

Citation:
```
@ARTICLE{2024MNRAS.531.1454K,
       author = {{Khederlarian}, Ashod and {Newman}, Jeffrey A. and {Andrews}, Brett H. and {Dey}, Biprateep and {Moustakas}, John and {Hearin}, Andrew and {Juneau}, St{\'e}phanie and {Tortorelli}, Luca and {Gruen}, Daniel and {Hahn}, ChangHoon and {Canning}, Rebecca E.~A. and {Aguilar}, Jessica Nicole and {Ahlen}, Steven and {Brooks}, David and {Claybaugh}, Todd and {de la Macorra}, Axel and {Doel}, Peter and {Fanning}, Kevin and {Ferraro}, Simone and {Forero-Romero}, Jaime and {Gazta{\~n}aga}, Enrique and {Gontcho}, Satya Gontcho A. and {Kehoe}, Robert and {Kisner}, Theodore and {Kremin}, Anthony and {Lambert}, Andrew and {Landriau}, Martin and {Manera}, Marc and {Meisner}, Aaron and {Miquel}, Ramon and {Mueller}, Eva-Maria and {Mu{\~n}oz-Guti{\'e}rrez}, Andrea and {Myers}, Adam and {Nie}, Jundan and {Poppett}, Claire and {Prada}, Francisco and {Rezaie}, Mehdi and {Rossi}, Graziano and {Sanchez}, Eusebio and {Schubnell}, Michael and {Silber}, Joseph Harry and {Sprayberry}, David and {Tarl{\'e}}, Gregory and {Weaver}, Benjamin Alan and {Zhou}, Zhimin and {Zou}, Hu},
        title = "{Emission line predictions for mock galaxy catalogues: a new differentiable and empirical mapping from DESI}",
      journal = {\mnras},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = jun,
       volume = {531},
       number = {1},
        pages = {1454-1470},
          doi = {10.1093/mnras/stae1189},
archivePrefix = {arXiv},
       eprint = {2404.03055},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.1454K},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

