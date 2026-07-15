import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

from .protocols import Regressor

from generalFunctions import get_preprocessor, plot_feature_importance, calculate_feature_importance

def evaluate_multiple(X: pd.DataFrame, y: pd.Series, numerical_cols: list[str], categorical_cols: list[str],
                      run_dir: str, n_folds: int, random_state: int, iterations: int,
                      model: Regressor, model_name: str,
                      robust: bool, test_range: float = 1, fi_output: list[str] = ['save_mean', 'plot_mean'],
                      save_results: bool = True, n_combination: int = 10, verbose: bool = True
                      ) -> dict[str, list[float]] | tuple[dict[str, list[float]], list[pd.DataFrame]]:
    """
    Evaluate models using CV, multiple times and return the R² and RMSE values for each run and, optionally, a list containing the PFI results.

    Args:
        X (pd.DataFrame): Feature matrix (should contain 'patient' column)
        y (pd.Series): Target variable
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names
        run_dir (str): Directory to save the results
        n_folds (int): Number of folds to divide the data
        random_state (int): Randomness seed
        iterations (int): How many times the model will run
        model (sklearn regressor): The model to use
        model_name (str): name of the model to be used for the PFI plots
        robust (bool): Weather or not to use the robust scaler from sklearn
        test_range (float): Quantile range, centered at 0.5, of `y` that can be used in the test set.
            Any value outside this range is forced into the training set
        fi_output (list): Settings for feature importance output. Has to be a subset of
            `['save_mean', 'return_mean', 'plot_mean', 'save_folds', 'return_folds', 'plot_folds']`.
            - `save_*` saves the result in a csv;
            - `return_*` returns the result in a list of DataFrames;
            - `plot_*` plots the results and saves it.
        save_results (bool): Weather or not to save the R² and RMSE values in csv format
        n_combination (int): Number of combinations to use to calculate the PIF
        verbose (bool): Whether or not to print aditional info and current fold and iteration being run
        
    Returns:
        dict: Dictionary containing results for each model
    """
    
    # Create colors
    colors = {
        'base'   : "\033[m",
        'error'  : "\033[38;2;220;0;0m",
        'yellow' : "\033[38;2;220;220;0m"
    }
    
    fi_output = set(fi_output)
    
    # Create visualization directory for fi plots and data
    if set(['save_mean', 'plot_mean', 'save_folds', 'plot_folds']) & fi_output:
        fi_dir = os.path.join(run_dir, 'feature_importance')
        os.makedirs(fi_dir, exist_ok=True)
    
    # Store patient information before dropping the column
    if 'patient' not in X.columns:
        raise ValueError("Input DataFrame X must contain a 'patient' column")
    
    # Drop the patient column from X for training
    X = X.drop(columns=['patient'])
    
    # Ensure the patient column is not in numerical or categorical columns
    numerical_cols = [col for col in numerical_cols if col != 'patient']
    categorical_cols = [col for col in categorical_cols if col != 'patient']
    
    # --- Results structure and models to evaluate ---
    results = {
            'r2': [],
            'rmse': []
        }
    
    fi_list = []
    
    mean_fi = pd.DataFrame()
    
    if save_results:
        r2_to_save = pd.DataFrame(index=range(iterations), columns=range(n_folds))
        rmse_to_save = pd.DataFrame(index=range(iterations), columns=range(n_folds))
    
    # Determine model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor(numerical_cols, categorical_cols, scale_numeric=True, robust=robust)),
        ('regressor', model)
    ])
    
    # --- Guarantee extremes go to training set ---
    extreme_y_values = None
    if 0 < test_range < 1:
        extreme_train_data_index = (np.quantile(y, (1 - test_range) / 2) > y) | (np.quantile(y, test_range / 2 + .5) < y)
        extreme_X_values = X[extreme_train_data_index]
        extreme_y_values = y[extreme_train_data_index]
        X = X[~extreme_train_data_index]
        y = y[~extreme_train_data_index]
        if verbose:
            print(f"{extreme_train_data_index.sum()} values removed as test candidates.")
        
    current_iteration = 0
    while current_iteration < iterations:
        
        # --- CV Splitters ---
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=(random_state + current_iteration))
        
        current_iteration += 1
        
        if verbose:
            print(f"{colors['yellow']}Running iteration {current_iteration}/{iterations}{colors['base']}")
        
        # --- Manual Cross-Validation Loop ---
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if verbose:
                print(f"{colors['yellow']}   Fold {i+1}/{cv.get_n_splits()}{colors['base']}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if extreme_y_values is not None:
                X_train = pd.concat([X_train, extreme_X_values], axis='index')
                y_train = pd.concat([y_train, extreme_y_values], axis='index')
            
            fitted = pipeline.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            
            if fi_output:
                fold_fi = calculate_feature_importance(fitted, X_train, y_train, X_test, y_test, n_combination, random_state=random_state)
                mean_fi = mean_fi.add(fold_fi, fill_value=0)
                if 'return_folds' in fi_output:
                    fi_list.append(fold_fi)
                if 'save_folds' in fi_output:
                    fold_fi.to_csv(os.path.join(fi_dir, f"fold_{current_iteration * n_folds + i + 1}_feature_importance.csv"), sep=";", decimal=',')
                if 'plot_folds' in fi_output:
                    plot_feature_importance(os.path.join(fi_dir, f'fold_{current_iteration * n_folds + i + 1}_feature_importance.png'), fold_fi, X.shape[1], 
                            f"{model_name} (sample {current_iteration * n_folds + i + 1})" if model_name != "" else
                            f"sample {current_iteration * n_folds + i + 1}")
            
            # Calculate and store scores
            results['r2'].append(r2_score(y_test, y_pred))
            results['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            if save_results:
                r2_to_save[i][current_iteration - 1] = results['r2'][-1]
                rmse_to_save[i][current_iteration - 1] = results['rmse'][-1]
            
            if verbose:
                print(f"\033[1F\033[0J", end='')
        if verbose:
            print(f"\033[1F\033[0J", end='')
        
        if fi_output:
            mean_fi = mean_fi.div(iterations * n_folds)
            if 'return_mean' in fi_output:
                fi_list.append(mean_fi)
            if 'save_mean' in fi_output:
                mean_fi.to_csv(os.path.join(fi_dir, f'mean_feature_importance.csv'), sep=";", decimal=',')
            if 'plot_mean' in fi_output:
                plot_feature_importance(os.path.join(fi_dir, f'mean_feature_importance.png'), mean_fi, X.shape[1], 
                        f"{model_name} (average)" if model_name != "" else
                        f"average")
        
    print(f"r2:\n{results['r2']}\n{np.mean(results['r2'])} ± {np.std(results['r2'], ddof=1)}\n")
    print(f"rmse:\n{results['rmse']}\n{np.mean(results['rmse'])} ± {np.std(results['rmse'], ddof=1)}")
    print('-' * 40 + "\n")
    
    if save_results:
        r2_to_save.rename(columns=lambda x: f"fold {x + 1}", inplace=True)
        rmse_to_save.rename(columns=lambda x: f"fold {x + 1}", inplace=True)
        
        r2_to_save.to_csv(os.path.join(run_dir, f'results_r2.csv'), sep=";", decimal=',', index=False)
        rmse_to_save.to_csv(os.path.join(run_dir, f'results_rmse.csv'), sep=";", decimal=',', index=False)
    
    if set(['return_folds', 'return_mean']) & fi_output:
        return results, fi_list
    return results
