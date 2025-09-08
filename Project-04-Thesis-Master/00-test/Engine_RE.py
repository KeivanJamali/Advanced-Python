import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import Information as info
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import seaborn as sns


class Engine:
    def __init__(self, data:pd.DataFrame, seed):
        self.data = data
        self.seed = seed

    def fit(self, model):
        self._set_setting()
        if model == "statsmodels":
            self._regression_model_stats()
        elif model == "sklearn":
            self._regression_model_sklearn()

    def _set_setting(self):
        self.x = self.data[info.features_dataloader].values
        self.y = self.data[info.target_dataloader].values
        print(f"[INFO] Data splitted to X with {self.x.shape} and Y with {self.y.shape}.")

    def _regression_model_stats(self):
        print("[INFO] Fitting 'statsmodels' LogisticRegression model...")
        # make the data smaller
        x = self.x[:3000]
        y = self.y[:3000]
        print(f"[INFO] We are training on {x.shape} and {y.shape} number os samples.")
        x = sm.add_constant(x)
        y = y.ravel()
        self.model = sm.MNLogit(y, x).fit() # Ordinary Least Square
        y_pred = self.model.predict(x)
        y_pred_label = y_pred.argmax(axis=1)
        print(self.model.summary())
        print("Accuracy:", accuracy_score(y, y_pred_label))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred_label))
        print(classification_report(y, y_pred_label))

    def _regression_model_sklearn(self):
        print("[INFO] Fitting 'scikit-learn' LogisticRegression model...")
        # make the data smaller
        x = self.x[:]
        y = self.y[:]
        print(f"[INFO] We are training on {x.shape} and {y.shape} number os samples.")
        clf = LogisticRegression(penalty="elasticnet",
                                 solver="saga",
                                 l1_ratio=0.5,
                                 C=1.0,
                                 max_iter=200, 
                                 verbose=1,
                                 random_state=self.seed)
        clf.fit(x, y.ravel())
        print("[INFO] Model fitting complete.")
        self.model = clf
        y_pred_label = clf.predict(x)
        y_pred = clf.predict_proba(x)  # Probabilities
        # Evaluation metrics
        acc = accuracy_score(y, y_pred_label)
        cm = confusion_matrix(y, y_pred_label)
        report = classification_report(y, y_pred_label)
        print(f"\n[RESULT] Accuracy: {acc:.4f}")
        print("\n[RESULT] Confusion Matrix:")
        print(cm)
        print("\n[RESULT] Classification Report:")
        print(report)
        # Optional: print coefficients
        print("\n[INFO] Model Coefficients (shape):", clf.coef_.shape)
        feature_names = [f"x{i+1}" for i in range(x.shape[1])]
            
        print(f"\n[INFO] Intercepts per class (shape {clf.intercept_.shape}):")
        print(clf.intercept_)

        # Top 5 most influential features per class
        print("\n[INFO] Top 5 most influential features per class:")
        for i, class_coef in enumerate(clf.coef_):
            top_idx = np.argsort(np.abs(class_coef))[-5:][::-1]
            top_feats = [(feature_names[j], class_coef[j]) for j in top_idx]
            print(f"  Class {clf.classes_[i]}: {top_feats}")

    def weight_plot_statsmodels(self, cons:bool=False, alpha=0.05):
        self.parameters_data = self.model.conf_int(alpha=alpha)
        self.parameters_data = pd.DataFrame(self.parameters_data, columns=["lower", "upper"], index=["cons"]+info.features)        
        self.parameters_data["coef"] = self.model.params
        if not cons:
            self.parameters_data.drop("cons", inplace=True)

        plt.figure(figsize=(8, len(self.parameters_data)*0.6))
        plt.errorbar(x=self.parameters_data['coef'], y=self.parameters_data.index,
                    xerr=[self.parameters_data['coef'] - self.parameters_data['lower'], 
                          self.parameters_data['upper'] - self.parameters_data['coef']],
                    fmt='o', color='navy', ecolor='skyblue', capsize=4)

        plt.axvline(x=0, color='gray', linestyle='--')  # zero line
        plt.title(f'Feature Effects with {int(100-100*alpha)}% Confidence Intervals')
        plt.xlabel('Coefficient Estimate')
        plt.ylabel('Feature')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def weight_plot_sklearn(self, top_k: int = 30, class_idx: int = 0):
        """
        Plot the most influential features for a selected class in the multiclass model.
        
        Parameters:
            top_k (int): Number of top features to plot.
            class_idx (int): The class index to inspect the coefficients for.
        """
        coef = self.model.coef_[class_idx]  # Shape: (n_features,)
        
        if hasattr(self, "x_train") and hasattr(self.x_train, "columns"):
            feature_names = self.x_train.columns
        else:
            feature_names = [f"x{i+1}" for i in range(self.x.shape[1])]

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coef": coef
        }).sort_values(by="coef", key=abs, ascending=False)

        coef_df = coef_df.head(top_k)
        
        plt.figure(figsize=(8, top_k * 0.5))
        plt.barh(coef_df["feature"], coef_df["coef"], color="skyblue")
        plt.axvline(x=0, color="gray", linestyle="--")
        plt.title(f"Top {top_k} Feature Coefficients (Class {class_idx})")
        plt.xlabel("Coefficient")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def effect_plot_statsmodels(self, data_id:int=None):
        self.parameters_data = self.model.params
        self.parameters_data = pd.DataFrame(self.parameters_data, columns=["coef"], index=["cons"]+info.features)
        self.parameters_data.drop("cons", inplace=True)
        self.coefs = self.parameters_data["coef"]
        self.x_effects = self.data[self.parameters_data.index]
        self.effects = self.x_effects.multiply(self.coefs, axis=1)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.effects, orient="h", color="skyblue", showfliers=True)
        if data_id:
            datapoint_effects = self.effects.loc[data_id]
            for i, (feature_name, effect_value) in enumerate(datapoint_effects.items()):
                plt.plot(effect_value, i, 'rx', markersize=10, label='Selected datapoint' if i == 0 else "")  # add label only once

        plt.title("Feature Effect Plot")
        plt.xlabel("Effect on Prediction")
        plt.ylabel("Feature")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def effect_plot_sklearn(self, data_id: int = None, class_idx: int = 0):
        """
        Plot feature effects (x_i * coef_i) using sklearn logistic regression model.

        Parameters:
            data_id (int): Index of sample in self.data to mark on the plot.
            class_idx (int): Class index for multiclass coefficients (default 0).
        """
        # Get model coefficients for the chosen class
        coef = self.model.coef_[class_idx]

        # Get feature names from self.x_train or generate fallback names
        if hasattr(self, "x_train") and hasattr(self.x_train, "columns"):
            feature_names = list(self.x_train.columns)
        else:
            feature_names = [f"x{i+1}" for i in range(self.x.shape[1])]

        # Create DataFrame for coefficients
        self.parameters_data = pd.DataFrame({"coef": coef}, index=feature_names)
        self.coefs = self.parameters_data["coef"]

        # Try to extract features from self.data matching the feature names
        try:
            self.x_effects = self.data[feature_names]
        except KeyError:
            print("[WARNING] Feature names not found in self.data. Falling back to positional indexing.")
            self.x_effects = pd.DataFrame(self.data.values[:, :len(feature_names)], columns=feature_names)

        # Compute elementwise effects
        self.effects = self.x_effects.multiply(self.coefs, axis=1)

        # Plot boxplot of feature effects
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.effects, orient="h", color="skyblue", showfliers=True)

        if data_id is not None:
            datapoint_effects = self.effects.iloc[data_id]
            for i, (feature_name, effect_value) in enumerate(datapoint_effects.items()):
                plt.plot(effect_value, i, 'rx', markersize=10, label='Selected datapoint' if i == 0 else "")
            plt.legend()

        plt.title(f"Feature Effect Plot (Class {class_idx})")
        plt.xlabel("Effect on Prediction (xᵢ·wᵢ)")
        plt.ylabel("Feature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()



