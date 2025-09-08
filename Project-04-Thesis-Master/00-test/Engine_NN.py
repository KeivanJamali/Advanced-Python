from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Information as info

class Engine:
    def __init__(self, model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 random_seed: int=42):
            """
            Initializes the class object with the given model, dataloaders, and device.

            Parameters:
                model (torch.nn.Module): The model to be used for training and evaluation.
                train_dataloader (torch.utils.data.DataLoader): The dataloader for the training data.
                val_dataloader (torch.utils.data.DataLoader): The dataloader for the validation data.
                test_dataloader (torch.utils.data.DataLoader): The dataloader for the test data.
            """
            self.seed = random_seed
            self.model = model
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader
            self.device = None
            self.result = None
            self.train_true_predict_list, self.val_true_predict_list, self.test_true_predict_list = None, None, None


    def _train_step(self, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple:
        """
        Trains the model for one step.

        Args:
            loss_fn (torch.nn.Module): The loss function used for training.
            optimizer (torch.optim.Optimizer): The optimizer used for training.

        Returns:
            tuple: A tuple containing the train loss, train accuracy, and a dictionary
                   containing the true and predicted values.
                   - val_loss (float): The average train loss for one epoch.
                   - val_acc (float): The average train accuracy for one epoch.
                   - true_predict_list (dict): A dictionary containing two lists: "true" and "predict".
                                              The "true" list contains the true labels of the train data,
                                              and the "predict" list contains the predicted labels."""
        self.model.train()
        train_loss, train_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        for batch, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y = y.squeeze()
            y_logit = self.model(x)
            loss = loss_fn(y_logit, y)
            train_loss += loss.item()
            y_pred_labels = y_logit.argmax(dim=1)
            y_pred_true = y
            train_acc += (y_pred_true == y_pred_labels).sum().item() / len(y_pred_true)
            true_predict_list["true"].append(y_pred_true.detach().cpu().numpy())
            true_predict_list["predict"].append(y_pred_labels.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        return train_loss, train_acc, true_predict_list
    
    
    def _val_step(self, loss_fn: torch.nn.Module) -> tuple:
        """
        Calculate the validation loss and accuracy for the current model.

        Parameters:
            loss_fn (torch.nn.Module): The loss function used for calculating the loss.

        Returns:
            tuple: A tuple containing the validation loss, validation accuracy, and a dictionary
                   containing the true and predicted values.
                   - val_loss (float): The average validation loss for one epoch.
                   - val_acc (float): The average validation accuracy for one epoch.
                   - true_predict_list (dict): A dictionary containing two lists: "true" and "predict".
                                              The "true" list contains the true labels of the validation data,
                                              and the "predict" list contains the predicted labels.
        """
        self.model.eval()
        val_loss, val_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.squeeze()
                y_logit = self.model(x)
                val_loss += loss_fn(y_logit, y).item()
                y_pred_labels = y_logit.argmax(dim=1)
                y_pred_true = y
                val_acc += (y_pred_true == y_pred_labels).sum().item() / len(y_pred_true)
                true_predict_list["true"].append(y_pred_true.detach().cpu().numpy())
                true_predict_list["predict"].append(y_pred_labels.detach().cpu().numpy())

            val_loss /= len(self.val_dataloader)
            val_acc /= len(self.val_dataloader)
            return val_loss, val_acc, true_predict_list

    def train(self,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              epochs_num: int,
              device: str = "cuda" if torch.cuda.is_available() else "cpu",
              early_stop_patience: int = None, resolution: int = 1) -> dict:
        """
        Trains the model for a specified number of epochs.

        Args:
            loss_fn (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.
            epochs_num (int): The number of epochs to train.
            device (str, optional): The device to use for training. Defaults to "cuda" if available, otherwise "cpu".
            early_stop_patience (int, optional): The number of epochs to wait for early stopping. Defaults to None,
            means no stop.
            resolution (int, optional): epoch / resolution = every writing. Defaults to 1.

        Returns:
            dict: A dictionary containing training and validation loss and accuracy.

        Raises:
            EnvironmentError: If no writer is found.
        """
        best_loss = float("inf")
        early_stop = 0
        results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        train_true_predict_list = {"true": [], "predict": []}
        val_true_predict_list = {"true": [], "predict": []}
        torch.manual_seed(self.seed)
        self.device = device
        self.model.to(device)
        for epoch in tqdm(range(1, epochs_num + 1)):
            train_loss, train_acc, true_predict_list = self._train_step(loss_fn=loss_fn, optimizer=optimizer)
            train_true_predict_list["true"].append(true_predict_list["true"])
            train_true_predict_list["predict"].append(true_predict_list["predict"])

            val_loss, val_acc, true_predict_list = self._val_step(loss_fn=loss_fn)
            val_true_predict_list["true"].append(true_predict_list["true"])
            val_true_predict_list["predict"].append(true_predict_list["predict"])
            if epoch % resolution == 0:
                print(
                f"Epoch {epoch} | train: Loss {train_loss:.6f} Accuracy {train_acc:.4f} | validation: Loss {val_loss:.6f} Accuracy {val_acc:.4f}")

            results["train_acc"].append(train_acc)
            results["train_loss"].append(train_loss)
            results["val_acc"].append(val_acc)
            results["val_loss"].append(val_loss)

            if early_stop_patience:
                if val_loss <= best_loss:
                    best_loss = val_loss
                    early_stop = 0
                else:
                    early_stop += 1
                if early_stop >= early_stop_patience:
                    print(f"Early_Stop_at {epoch} Epoch")
                    break
        self.train_true_predict_list = train_true_predict_list
        self.val_true_predict_list = val_true_predict_list
        self.result = results
        return self.result

    def test(self, loss_fn: torch.nn.Module) -> tuple:
        """
        Calculate the test loss and accuracy of the model.

        Parameters:
            loss_fn (torch.nn.Module): The loss function used for calculating the test loss.

        Returns:
            tuple: A tuple containing the test loss (float), test accuracy (float), and a dictionary
            containing the true and predicted values for each test sample.
        """
        print("[!!!IMPORTANT NOTE!!!]")
        print("The test_function provided here is intended solely for the final model analysis and reporting purposes.")
        print("Please refrain from using it as a general-purpose function in your own projects. Always refer to")
        print("the appropriate train and validation data for developing and fine-tuning your own models.")
        self.model.eval()
        test_loss, test_acc = 0, 0
        true_predict_list = {"true": [], "predict": []}

        with torch.inference_mode():
            for x, y in self.test_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y = y.squeeze()
                y_logit = self.model(x)
                test_loss += loss_fn(y_logit, y).item()
                y_pred_labels = y_logit.argmax(dim=1)
                y_pred_true = y
                test_acc += (y_pred_labels == y_pred_true).sum().item() / len(y_pred_true)
                true_predict_list["true"].append(y_pred_true.detach().cpu().numpy())
                true_predict_list["predict"].append(y_pred_labels.detach().cpu().numpy())

            test_loss /= len(self.val_dataloader)
            test_acc /= len(self.val_dataloader)
            print(f"Loss : {test_loss} and Accuracy : {test_acc}")
            return test_loss, test_acc, true_predict_list
        
    def predict(self, x: np.ndarray, x_scaler=None) -> pd.DataFrame:
        """
        Predicts the output for the given data using the trained model.

        Parameters:
            x (np.ndarray): The data to be predicted.
            x_scaler: The optional scaler used for scaling the data.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted output.
        """
        x = pd.DataFrame(x, columns=info.features_dataloader)
        if x_scaler:
            x_scaled = x_scaler.transform(x)
        else:
            x_scaled = x.values.copy()
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(dim=0)
        y = self.model(x_scaled.to(self.device)).cpu().detach().numpy()

        x_predict = x.values[-1].reshape(1, -1)

        predict = pd.DataFrame(np.concatenate((x_predict, y), axis=1),
                               columns=info.features_dataloader + info.target_dataloader)
        return predict
    
    def confusion_matrix(self, data_: str, plot: bool=False):
        """
        Calculate the confusion matrix and classification report for the specified data. Must be for
        classification approach.

        Parameters:
            data_ (str): The type of data for which to calculate the confusion matrix and classification report.
            plot (bool): If you want to plot the confusion matrix change this to True.

        Returns:
            tuple: A tuple containing the confusion matrix and the classification report.
        """
        if data_ not in ["train", "val", "test"]:
            raise ValueError("Data must be either 'train', 'val', or 'test'.")
        choices = [self.train_true_predict_list, self.val_true_predict_list, self.test_true_predict_list]
        choice = choices[["train", "val", "test"].index(data_)]
        true_labels = np.concatenate(choice["true"], axis=0)
        predicted_labels = np.concatenate(choice["predict"], axis=0)
        confusion_m = confusion_matrix(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, zero_division=1)
        if plot:
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_m, display_labels=["Survived", "Not_Survived"])
            disp.plot()
            plt.show()
        return confusion_m, report

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.result["train_loss"])), self.result["train_loss"], label="train")
        plt.plot(range(len(self.result["val_loss"])), self.result["val_loss"], label="val")
        plt.legend()
        plt.title("Loss")
        plt.show()

    def plot_acc(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.result["train_acc"])), self.result["train_acc"], label="train")
        plt.plot(range(len(self.result["val_acc"])), self.result["val_acc"], label="val")
        plt.legend()
        plt.title("Accuracy")
        plt.show()

    def plot_predict_real(self):
        y1 = self.train_true_predict_list["predict"][-1][0].squeeze()
        y2 = self.val_true_predict_list["predict"][-1][0].squeeze()

        r1 = self.train_true_predict_list["true"][-1][0].squeeze()
        r2 = self.val_true_predict_list["true"][-1][0].squeeze()

        x1 = [i for i in range(len(r1))]
        x2 = [i for i in range(len(r1), len(r1) + len(r2))]

        plt.figure(figsize=(10, 5))
        plt.plot(x1, y1, label="predict")
        plt.plot(x2, y2, label="predict")
        plt.plot(x1, r1, label="true")
        plt.plot(x2, r2, label="true")
        plt.legend()
        plt.show()

    def partial_dependence_plot(self, pdp_data:dict) -> None:
        result = {}
        for feature, values in pdp_data.items():
            result[feature] = [list(values.keys()), []]
            for value, dataloader in values.items():
                self.model.eval()
                predict_list = []
                with torch.inference_mode():
                    for x, y in dataloader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_logit = self.model(x)
                        y_pred_labels = y_logit.argmax(dim=1)
                        predict_list.append(y_pred_labels.detach().cpu().numpy())
            
                result[feature][1].append(np.array(predict_list).mean())
            plt.plot(result[feature][0], result[feature][1], label=feature)

        plt.legend()
        plt.title("PDP plot")
        plt.ylabel("Target")
        plt.xlabel("Feature Value")
        plt.show()

        self.pdp = result

    def individual_conditional_expectation(self, ice_data:dict) -> None:
        result = {}
        for feature, values in ice_data.items():
            for i, dataloader in values["data"].items():
                self.model.eval()
                with torch.inference_mode():
                    for x, y in dataloader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_logit = self.model(x)
                        y_logit_diying = y_logit[:, 0]
                plt.plot(list(values["values"]), y_logit_diying, c="blue", alpha=0.05)


        plt.title("ICE plot")
        plt.ylabel("Target")
        plt.xlabel("Feature Value")

    def accumulated_local_effects_plot(self, ale_data:list):
        difference_list = []
        devision = ale_data[0]
        data_list = ale_data[1]
        for data in data_list:
            self.model.eval()
            with torch.inference_mode():
                for x, y in data[0]:
                    x, y = x.to(self.device), y.to(self.device)
                    y_logit_min = self.model(x)
                    y_logit_min = y_logit_min[:, 0]
                for x, y in data[1]:
                    x, y = x.to(self.device), y.to(self.device)
                    y_logit_max = self.model(x)
                    y_logit_max = y_logit_max[:, 0]
                differences = y_logit_max - y_logit_min                
                difference_list.append(differences.mean().item())
        y = []
        x = []
        for i in range(1, len(difference_list)):
            x.append((devision[i-1]+devision[i])/2)
            y.append(sum(difference_list[:i]))
        mean_y = sum(y)/len(y)
        for i in range(len(y)):
            y[i] -= mean_y
        plt.plot(x, y, label=ale_data[2][0])
        plt.title("ALE plot")
        plt.legend()
        plt.ylabel("Centered target value")
        plt.xlabel("Feature Value")

    def feature_importance_plot(self, datas:dict, loss_fn, fig_size=(6, 10)):
        all_results = []

        for i in range(2):
            data = datas[i]
            features_loss = {}
            for key, value in data.items():
                self.model.eval()
                loss = 0
                with torch.inference_mode():
                    for x, y in value:
                        x, y = x.to(self.device), y.to(self.device)
                        y = y.squeeze()
                        y_logit = self.model(x)
                        loss += loss_fn(y_logit, y)
                    loss = loss / len(value)

                if key == "True":
                    true_loss = loss.item()
                else:
                    features_loss[key] = (loss.item() / true_loss) - 1

            all_results.append(features_loss)
        
        features = list(all_results[0].keys())
        train_importance = [all_results[0][f] for f in features]
        test_importance = [all_results[1][f] for f in features]
        y = range(len(features))

        plt.figure(figsize=fig_size)
        plt.barh(y, train_importance, height=0.4, color="lightblue", label="Train", align='center')
        plt.barh([i - 0.4 for i in y], test_importance, height=0.4, color="orange", label="Test", align='center')

        plt.yticks(y, features)
        plt.xlabel("(loss_permuted / loss_true) - 1")
        plt.title("Feature Importance (Train vs Test)")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        return all_results

    def feature_intraction_value(self, data:dict):
        result = {}
        features = []
        for feature, values in data.items():
            features.append(feature)
            result[feature] = [list(values.keys()), []]
            for value, dataloader in values.items():
                self.model.eval()
                predict_list = []
                with torch.inference_mode():
                    for x, y in dataloader:
                        x, y = x.to(self.device), y.to(self.device)
                        y_logit = self.model(x)
                        predict_list.append(y_logit.detach().cpu().numpy())
                result[feature][1].append(np.array(predict_list[0]).mean())

        # calculate H-Statistic
        numerator = 0.0
        denominator = 0.0

        for pdp_kj in result[features[2]]:
            pass

    # def save(self):
    #     save_path = f"""model/{Information.model_architecture}/{Information.model_name}_{Information.model_architecture}
    #     _{Information.model_version}.pt"""
    #     torch.save(self.model.state_dict(), save_path)

