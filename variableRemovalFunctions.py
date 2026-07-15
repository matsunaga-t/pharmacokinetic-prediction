import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from .protocols import Regressor

from .generalFunctions import get_preprocessor, calculate_feature_importance, load_data

def removeVariablesPFI(X: pd.DataFrame, y: pd.DataFrame | pd.Series,
                    numerical_cols: list[str], categorical_cols: list[str],
                    n_folds: int, random_state: int, iterations: int,
                    model: Regressor,
                    robust: bool = False, mean_feature_importance: bool = True,
                    n_combination: int = 10, tol: float = 0.05, additive: bool = False,
                    verbose: bool = True
                    ) -> pd.DataFrame:
    """
    Iteratively remove variables based on permutation feature importance.
    
    This function:
    1. Evaluates a model's performance and PFI `iterations * n_folds` times using the given feature set, each one with different train and test sets;
    2. Selects the features based on PFI of the worst or average run;
    3. Remove the feature and reevaluate the model with the new set;
        - If the performance decreases by more than `tol`, undo the feature removal and go to the next one;
        - Otherwise, go back to step 1 with the new set.
    
    The function repeats these steps until no more variable can be removed.
    Returns a DataFrame containing the mean and standard deviation of a model's R² and RMSE after each successful variable removal, with the name of the removed variable. For convenience, the first row is the performance of the original variable set, and its "removed feature" is called "NOTHING".

    This function runs a model `iterations * n_folds` times because, internally, it uses `sklearn.model_selection.Kfold` to separate the train and test sets. This is done to make it more likely for all of the runs to have different test sets.
    
    Args:
        X (pd.DataFrame): Feature matrix (should contain 'patient' column)
        y (pd.DataFrame | pd.Series): Target variable
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        n_folds (int): Number of partitions to divide the data
        random_state (int): Randomness seed
        iterations (int): How many times the data will be partitioned
        model (sklearn regressor): The regressor model to use
        robust (bool): Weather or not to use the robust scaler from sklearn
        mean_feature_importance (bool): Weather or not to use the average PFI if no feature is available in the worst run's PFI
        n_combination (int): Number of combinations to use to calculate the PIF
        tol (float): How low the performance of the new variable set _can_ be for it to _still_ be acceptable (permanently remove it)
        additive (bool): If `True`, then the previous performance is subtracted `tol`. Else, it is multiplied by `1 - tol`
        verbose (bool): Weather or not to print the current run and variable removed
        
    Returns:
        pd.DataFrame: DataFrame containing the performance after each variable removal
    """
    
    # Create colors
    colors = {
        'base'   : "\033[m",
        'error'  : "\033[38;2;220;0;0m",
        'yellow' : "\033[38;2;220;220;0m"
    }
    
    # Store patient information before dropping the column
    if 'patient' not in X.columns:
        raise ValueError("Input DataFrame X must contain a 'patient' column")
    
    # Create a copy of X without the patient column for training
    base_X_train_data = X.drop(columns=['patient'])
    
    # Ensure the patient column is not in numerical or categorical columns
    base_numerical_cols = [col for col in numerical_cols if col != 'patient']
    base_categorical_cols = [col for col in categorical_cols if col != 'patient']
    
    # --- Results structure and models to evaluate ---
    results = pd.DataFrame({'feature':[], 'r2_mean':[], 'r2_std':[], 'rmse_mean':[], 'rmse_std':[]}).T
    
    features_to_drop = []
    features_to_keep = []
    
    total_removals = base_X_train_data.shape[1]
    iteration_values = {
        'prev' : {
            'mean' : pd.DataFrame(),
            'worst' : pd.DataFrame(),
            'r2' : -math.inf
        },
        'curr' : {
            'mean' : pd.DataFrame(),
            'worst' : pd.DataFrame(),
            'r2' : -math.inf
        }
    }
    while len(features_to_drop) < total_removals:
        current_idx = len(features_to_drop)
        # Removing the features
        X_train_data = base_X_train_data.drop(columns=features_to_drop)
        categorical_cols = base_categorical_cols.copy()
        numerical_cols = base_numerical_cols.copy()
        for feature in features_to_drop:
            if feature in categorical_cols:
                categorical_cols.remove(feature)
            if feature in numerical_cols:
                numerical_cols.remove(feature)
        
        # Determine model pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', get_preprocessor(numerical_cols, categorical_cols, scale_numeric=True, robust=robust)),
            ('regressor', model)
        ])
        
        worst_fold_r2 = 1
        mean_fi =pd.DataFrame()
        
        intermediate = {
            'r2' : [],
            'rmse': []
        }
        current_iteration = 0
        while current_iteration < iterations:
            
            # --- CV Splitters ---
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=(random_state + current_iteration))
            
            current_iteration += 1
            
            if verbose:
                print(f"{colors['yellow']}Running iteration {current_iteration}/{iterations} with {X_train_data.shape[1]} variables{colors['base']}")
            
            # --- Manual Cross-Validation Loop ---
            for i, (train_idx, test_idx) in enumerate(cv.split(X_train_data, y)):
                if verbose:
                    print(f"{colors['yellow']}   Fold {i+1}/{cv.get_n_splits()}{colors['base']}")
                X_train, X_test = X_train_data.iloc[train_idx], X_train_data.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                fitted = pipeline.fit(X_train, y_train)
                y_pred = fitted.predict(X_test)
                
                fold_fi = calculate_feature_importance(fitted, X_train, y_train, X_test, y_test, n_combination, random_state=random_state)
                
                if mean_feature_importance:
                    mean_fi = mean_fi.add(fold_fi, fill_value=0)
                
                # Calculate and store individual scores
                fold_r2 = r2_score(y_test, y_pred)
                intermediate['r2'].append(fold_r2)
                intermediate['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                
                # Get the lowest score
                if fold_r2 < worst_fold_r2:
                    worst_fold_r2 = fold_r2
                    iteration_values['curr']['worst'] = fold_fi
                if verbose:
                    print(f"\033[1F\033[0J", end='')
            if verbose:
                print(f"\033[1F\033[0J", end='')
        
        if mean_feature_importance:
            mean_fi = mean_fi.div(n_folds * iterations)
            iteration_values['curr']['mean'] = mean_fi
        
        # Adding results
        r2_mean = np.mean(intermediate['r2'])
        iteration_values['curr']['r2'] = r2_mean
        r2_std = np.std(intermediate['r2'], ddof=1)
        rmse_mean = np.mean(intermediate['rmse'])
        rmse_std = np.std(intermediate['rmse'], ddof=1)
        if verbose:
            print(f"          {r2_mean} ± {r2_std}")
            print(f"          {rmse_mean} ± {rmse_std}")
        
        # Check if the new value is better than the previous
        if additive:
            minimum_alowed = iteration_values['prev']['r2'] - tol
        else:
            minimum_alowed = iteration_values['prev']['r2'] - abs(iteration_values['prev']['r2'] * tol)
        
        if (iteration_values['curr']['r2'] < minimum_alowed):
            if verbose:
                print(f"{colors['error']}Undoing {features_to_drop[-1]} removal{colors['base']}")
            # Adding the last removed feature back in and marking it to not be removed
            features_to_keep.append(features_to_drop.pop())
            iteration_values['curr'] = iteration_values['prev'].copy()
        else:
            # Inserting the new result in the DataFrame
            results.insert(loc=current_idx, column=current_idx, value=["NOTHING" if len(features_to_drop) == 0 else features_to_drop[-1], r2_mean, r2_std, rmse_mean, rmse_std])
            iteration_values['prev'] = iteration_values['curr'].copy()
            # Resetting the features to not remove
            features_to_keep = []
            
        # removing the worst variable from the worst fold
        fi = iteration_values['curr']['worst']
        fi = fi[(fi['train_mean'] >= 0) & (fi['test_mean'] <= 0) & (~fi.index.to_series().isin(features_to_keep))]
        
        if fi.shape[0] == 0:
            if mean_feature_importance:
                fi = iteration_values['curr']['mean']
                fi = fi[(fi['train_mean'] >= 0) & (fi['test_mean'] <= 0) & (~fi.index.to_series().isin(features_to_keep))]
                if fi.shape[0] == 0:
                    break
            else:
                break
            
        fi.insert(loc=1, column="diff", value=fi['train_mean'] - fi['test_mean'])
        features_to_drop.append(fi.sort_values(by='diff', axis='index', ascending=False).head(1).index.values[0])
        #results.loc['feature', current_idx] = features_to_drop[-1]
        
        # Printing results
        if verbose:
            print(f"{colors['yellow']}{features_to_drop[-1]} removed{colors['base']}")
        #print(results)
            
    return results.T

def plotPerformancePerRemoval(totalVars : int, results : pd.DataFrame, errorArea: bool = False) -> plt.Figure:
    """
    Plots the performance (r2) of the model after each variable removal using the results from `removeVariablesPFI`.
    
    Args:
        totalVars (int): the total number of variables (starting number of variables, without any removal)
        results (DataFrame): the results obtained from `removeVariablesPFI`
        errorArea (bool): `True` to show error as an area, False to use error bars instead
        
    Returns:
        plt.Figure: Figure object containing the r2 per variable removed
    """
    mean_y = results.loc[:, 'r2_mean']
    std_y = results.loc[:, 'r2_std']
    x_vals = totalVars - results.index.values
    fig, ax = plt.subplots()
    if errorArea: # tipo de erro 0 para pintar uma área e 1 para usar barras
        ax.fill_between(x_vals, mean_y - std_y, mean_y + std_y, color='tab:blue', alpha=.2, label="Standard deviation")
    else:
        ax.errorbar(x_vals, mean_y, std_y, ecolor='k')
    ax.invert_xaxis()
    ax.set_xlabel('number of variables')
    ax.set_ylabel(r'$r^2$')
    ax.plot(x_vals, mean_y, "o-", color='tab:blue', label="Average")
    ax.legend(loc='lower right')
    ax.grid(visible=True, which='both', axis='both')
    return fig

def removalWrapper(file_path, target_col, drug_name, n_folds, random_state, iterations, model, 
                calc_mean, fi_iter, neg_tol, message=""):
    X, y, numerical_cols, categorical_cols, res_dir = load_data(file_path=file_path, target_col=target_col,
                                                    columns_to_drop=['pre_anti_hbc', 'pre_anti_hiv', 'pre_anti_hcv', 'pre_beta_hcg', 'pre_HbsAg'],
                                                    drug_name=drug_name, specific_dir="removing")
    
    startTime = datetime.now()
    
    results = removeVariablesPFI(X=X, y=y, numerical_cols=numerical_cols, categorical_cols=categorical_cols,
                            n_folds=n_folds, random_state=random_state, iterations=iterations, model=model, robust=False,
                            mean_feature_importance=calc_mean, n_combination=fi_iter, tol=neg_tol)
    
    msg = open(os.path.join(res_dir, "message.txt"), "w")
    msg.write(f"{message}\ntotal time: {datetime.now() - startTime}")
    msg.close()
    
    results.to_csv(os.path.join(res_dir, "results.csv"), sep=';', decimal=',')
    
    fig = plotPerformancePerRemoval(X.shape[1] - 1, results=results, errorArea=True)
    fig.savefig(os.path.join(res_dir, 'variable-removal.svg'), bbox_inches='tight')