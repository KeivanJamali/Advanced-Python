import numpy as np
import pandas as pd
from scipy.optimize import minimize

class GravityModel:
    def __init__(self, decay_function, param_names=None):
        """
        decay_function: Callable like f(distances, **params)
        param_names: List of parameter names (str) to calibrate
        """
        self.decay_function = decay_function
        self.param_names = param_names or []
        self.fitted = False

    def _prepare_from_dataframe(self, data):
        zones = sorted(set(data['origin']) | set(data['dest']))
        self.zone_index = {z: i for i, z in enumerate(zones)}
        n = len(zones)

        observed = np.zeros((n, n))
        distances = np.zeros((n, n))
        productions = np.zeros(n)
        attractions = np.zeros(n)

        # Fill OD values and distances
        for row in data.itertuples(index=False):
            i = self.zone_index[row.origin]
            j = self.zone_index[row.dest]
            observed[i, j] = row.demand
            distances[i, j] = row.cost

        # Extract production/attraction
        prod_df = data.drop_duplicates("origin")[["origin", "production"]].set_index("origin")
        attr_df = data.drop_duplicates("dest")[["dest", "attraction"]].set_index("dest")

        for zone, idx in self.zone_index.items():
            productions[idx] = prod_df.at[zone, 'production']
            attractions[idx] = attr_df.at[zone, 'attraction']

        self.productions = productions
        self.attractions = attractions
        self.distance_matrix = distances
        self.observed_OD = observed

    def _compute_OD(self, decay_params):
        F = self.decay_function(self.distance_matrix, **decay_params)
        raw = np.outer(self.productions, self.attractions) * F
        scaling = raw.sum(axis=1, keepdims=True)
        return (raw / scaling) * self.productions[:, None]

    def fit(self, data, decay_params):
        self._prepare_from_dataframe(data)
        self.decay_params = decay_params
        self.OD_matrix = self._compute_OD(decay_params)
        self.fitted = True
        return self

    def calibrate(self, data, initial_params, bounds=None, loss='rmse'):
        self._prepare_from_dataframe(data)

        def objective(param_values):
            params = dict(zip(self.param_names, param_values))
            estimated = self._compute_OD(params)
            if loss == 'rmse':
                return np.sqrt(np.mean((self.observed_OD - estimated) ** 2))
            elif loss == 'mape':
                return np.mean(np.abs((self.observed_OD - estimated) / (self.observed_OD + 1e-6)))
            else:
                raise ValueError(f"Unsupported loss function: {loss}")

        x0 = [initial_params[p] for p in self.param_names]
        result = minimize(objective, x0, bounds=bounds)
        self.decay_params = dict(zip(self.param_names, result.x))
        self.OD_matrix = self._compute_OD(self.decay_params)
        self.fitted = True
        return self.decay_params
