from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Information as info

class Engine:
    def __init__(self, data, seed=42):
        self.seed = seed
        self.data = data

    def fit(self):
        self.train_data, self.test_data = self._split_data()
        self.x_train, self.y_train = self.train_data.drop(info.target_dataloader, axis=1), self.train_data[info.target_dataloader]
        self.x_test, self.y_test = self.test_data.drop(info.target_dataloader, axis=1), self.test_data[info.target_dataloader]

    def _split_data(self):
        train, test = train_test_split(self.data, test_size=0.3, random_state=self.seed)
        return train, test
    
    def pre_tree(self, if_plot_tree=False, if_confusion_matrix=False, alpha=0):
        tree = DecisionTreeClassifier(random_state=self.seed, ccp_alpha=alpha)
        tree = tree.fit(self.x_train, self.y_train)
        train_sccore = tree.score(self.x_train, self.y_train)
        test_sccore = tree.score(self.x_test, self.y_test)
        print(f"Accuracy for Train is {train_sccore}")
        print(f"Accuracy for Test is {test_sccore}")

        if if_plot_tree:
            plt.figure(figsize=(60, 30))
            plot_tree(tree,
                      filled=True,
                      rounded=True,
                      class_names=[str(i) for i in sorted(self.y_train["destination_area_code"].unique())],
                      feature_names=self.x_train.columns)
        
        if if_confusion_matrix:
            y_pred = tree.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in sorted(self.y_train["destination_area_code"].unique())])
            disp.plot()
            plt.show()
        self.tree = tree
        
    def find_alpha_for_pruning(self, if_plot_tree=False, if_confusion_matrix=False, plot_loss=False):
        tree = DecisionTreeClassifier(random_state=self.seed)
        tree = tree.fit(self.x_train, self.y_train)
        if if_plot_tree:
            plt.figure(figsize=(30, 15))
            plot_tree(tree,
                      filled=True,
                      rounded=True,
                      class_names=[str(i) for i in sorted(self.y_train["destination_area_code"].unique())],
                      feature_names=self.x_train.columns)
        
        if if_confusion_matrix:
            y_pred = tree.predict(self.x_test)
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in sorted(self.y_train["destination_area_code"].unique())])
            disp.plot()
            plt.show()

        path = tree.cost_complexity_pruning_path(self.x_train, self.y_train)
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[:-1] # No root.

        tree_alters = []
        for ccp_alpha in ccp_alphas:
            tree = DecisionTreeClassifier(random_state=self.seed, ccp_alpha=ccp_alpha)
            tree.fit(self.x_train, self.y_train)
            tree_alters.append(tree)

        train_scores = [tree.score(self.x_train, self.y_train) for tree in tree_alters]
        test_scores = [tree.score(self.x_test, self.y_test) for tree in tree_alters]
        if plot_loss:
            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Axxuracy vs alpha for training and testing sets.")
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            plt.show()

    def cross_validation_to_find_best_alpha(self):
        tree = DecisionTreeClassifier(random_state=self.seed)
        tree = tree.fit(self.x_train, self.y_train)
        path = tree.cost_complexity_pruning_path(self.x_train, self.y_train)
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[:-1] # No root.

        alpha_loop_values = []

        for ccp_alpha in ccp_alphas:
            tree = DecisionTreeClassifier(random_state=self.seed, ccp_alpha=ccp_alpha)
            scores = cross_val_score(tree, self.x_train, self.y_train, cv=5)
            alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

        alpha_result = pd.DataFrame(alpha_loop_values,
                                    columns=['alpha', "mean_accuracy", "std"])

        alpha_result.plot(x="alpha",
                          y="mean_accuracy",
                          yerr="std",
                          marker="o",
                          linestyle="--")
        
    def partial_dependence_plot(self, features:dict):
        result = {}
        for feature, values in features.items():
            result[feature] = [values, []]
            for value in values:
                new_data = self.data.copy()
                new_data[feature] = value
                x_new_data = new_data.drop(info.target, axis=1)
                y_new_data = self.tree.predict(x_new_data)
                y = y_new_data.mean()
                result[feature][1].append(y)

            plt.plot(result[feature][0], result[feature][1], label=feature)

        plt.legend()
        plt.title("PDP plot")
        plt.ylabel("Target")
        plt.xlabel("Feature Value")
        self.pdp = result

    def individual_conditional_expectation(self, features:dict):
        for feature, values in features.items():
            new_data = self.data.copy()
            for i in new_data.index:
                row = new_data.loc[i]
                row_i = row.copy()
                new_data_i = pd.DataFrame(columns=row.index)
                for value in values:
                    row_i[feature] = value
                    new_data_i.loc[len(new_data_i)] = row_i
                new_data_i.drop(info.target, axis=1, inplace=True)
                y_new_data_i = self.tree.predict(new_data_i)
                plt.plot(values, y_new_data_i, c="blue", alpha=0.01, label=feature)

        plt.title("ICE plot")
        plt.ylabel("Target")
        plt.xlabel("Feature Value")

    def accumulated_local_effects_plot(self, features:list, n:int):
        for feature in features:
            difference_list = []
            max_value, min_value = self.data[feature].max(), self.data[feature].min()
            devision = list(np.linspace(min_value, max_value, n+1))
            new_devision = list(np.linspace(min_value, max_value, n+1))
            temp = len(devision)-1
            for i in range(temp):
                mask1 = self.data[feature] >= devision[i]
                mask2 = self.data[feature] <= devision[i+1]
                combined_mask = mask1.astype(int)+mask2.astype(int) == 2
                new_data_min = self.data.copy()[combined_mask]
                new_data_max = self.data.copy()[combined_mask]
                if new_data_min.empty:
                    new_devision.remove(devision[i+1])
                    print(f"[INFO] Empty Interval {devision[i]} to {devision[i+1]}")
                    continue
                new_data_min[feature] = devision[i]
                new_data_max[feature] = devision[i+1]
                new_data_min.drop(info.target, axis=1, inplace=True)
                new_data_max.drop(info.target, axis=1, inplace=True)
                y_new_data_min = self.tree.predict(new_data_min)
                y_new_data_max = self.tree.predict(new_data_max)
                differences = y_new_data_max - y_new_data_min
                difference_list.append(differences.mean())
            y = []
            x = []
            for i in range(1, len(difference_list)):
                x.append((new_devision[i-1]+new_devision[i])/2)
                y.append(sum(difference_list[:i]))
            mean_y = sum(y)/len(y)
            for i in range(len(y)):
                y[i] -= mean_y
            plt.plot(x, y, label=features[0])
            plt.legend()
            plt.title("ALE plot")
            plt.ylabel("Centered target value")
            plt.xlabel("Feature Value")