from jax import numpy as jnp
from jax.nn import selu
from jax import vmap as vmap

class EmissionLineModel:
    def __init__(self, path_to_model=None):
        self.model_params = jnp.load(path_to_model, allow_pickle=True).item()
        self.lines = ["OII_DOUBLET", "HGAMMA", "HBETA", "OIII_4959", "OIII_5007",
                      "NII_6548", "HALPHA", "NII_6584", "SII_6716", "SII_6731"]
        
    def _feedforward_prediction(self, params, inputs):
        activations = inputs
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = selu(outputs)

        w_final, b_final = params[-1]
        final_outputs = jnp.dot(w_final, activations) + b_final

        return final_outputs[0]
    
    def predict(self, colors=None, luminosities=None):
        many_predictions = vmap(self._feedforward_prediction, in_axes=(None, 0))

        predictions = {}
        for line in self.lines:
            color_scaler = self.model_params[line]['color_scaler']
            transformed_colors = color_scaler.transform(colors)
            if luminosities is not None:
                luminosity_scaler = self.model_params[line]['luminosity_scaler']
                transformed_luminosities = luminosity_scaler.transform(luminosities.reshape(-1, 1))
                features = jnp.concatenate((transformed_colors, transformed_luminosities), axis=1)
            else:
                features = transformed_colors

            ews = jnp.sinh(many_predictions(self.model_params[line]['params'], features))
            predictions[line] = ews

        return predictions
    
