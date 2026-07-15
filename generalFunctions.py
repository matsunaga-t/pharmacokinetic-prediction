import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from datetime import datetime

from usefulValues import featureDict

def get_preprocessor(numerical_cols, categorical_cols, scale_numeric=True, robust=False):
    """Creates a preprocessing pipeline for numerical and categorical features."""
    if scale_numeric:
        if robust:
            numerical_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        else:
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    else:
        numerical_transformer = 'passthrough'

    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor


def get_results_directory(base_dir, drug_name, target_variable, specific_dir = None):
    """
    Generate and optionally create the standard results directory structure.
    
    Args:
        base_dir (str): Base directory for saving results
        drug_name (str): Name of the drug being analyzed. If None, will use 'unspecified_drug'.
        target_variable (str): Name of the target variable. If None, will use 'unspecified_target'.
        create_dirs (bool): Whether to create the directories if they don't exist
        include_timestamp (bool): Whether to include a timestamp in the directory path.
                                If False, the directory will be at base_dir/drug_name/target_variable/
        
    Returns:
        str: Path to the results directory (with or without timestamp)
    """
    
    # Handle None values
    if drug_name is None:
        drug_name = 'unspecified_drug'
    if target_variable is None:
        target_variable = 'unspecified_target'
    
    # Create directory structure
    path_components = [base_dir, str(drug_name), str(target_variable)]
    if specific_dir is not None:
        path_components.append(specific_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path_components.append(timestamp)
    
    run_dir = os.path.join(*path_components)
    
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def load_data(file_path, target_col='ref_AUC_0_t', columns_to_drop=None, columns_to_use=None, cnsp_filter=None, 
             drug_name=None, pk_predictors=None, base_dir='results', specific_dir = None):
    
    # Initialize metadata dictionary
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H%M%S'),
        'file_path': file_path,
        'target_column': target_col,
        'filters_applied': {},
        'excluded_participants': [],
        'columns_used': {},
        'warnings': []
    }
    
    # Create log directory - use the same structure as results but without timestamp initially
    # This ensures the base directory exists for the log file
    # Change dir_drug_name in case more than one drug is being analysed
    dir_drug_name = "Combined" if isinstance(drug_name, list) or drug_name is None else drug_name
    base_log_dir = get_results_directory(base_dir, dir_drug_name, target_col, specific_dir)
    
    # The actual log file will be created in the timestamped directory by save_detailed_results
    # For now, we'll create a temporary log file that will be moved later
    log_file = os.path.join(base_log_dir, f"data_processing_{metadata['timestamp']}.log")
    
    # Ensure the base log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_message(message, warning=False):
        """Helper function to log messages to both console and file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
        if warning:
            metadata['warnings'].append(message)
    
    try:
        # Load the data
        data = pd.read_csv(file_path)
        log_message(f"Loading data from: {file_path}")
        log_message(f"Total rows loaded: {len(data)}")
        log_message("Filtering by initial_columns")
        initial_columns = set(data.columns)
        if columns_to_use is not None:
            initial_columns &= (set(columns_to_use) | set(['cnsp', 'drug', 'patient']) | set([target_col]))
        data = data[list(initial_columns)]
        log_message(f"Initial columns: {', '.join(initial_columns)}")
        
        # Remove columns with only one unique value
        constant_columns = [col for col in data.columns if data[col].nunique() <= 1]
        if constant_columns:
            log_message(f"Removing {len(constant_columns)} constant columns: {', '.join(constant_columns)}")
            data = data.drop(columns=constant_columns)
            metadata['columns_removed'] = {
                'constant_columns': constant_columns,
                'reason': 'Columns with only one unique value',
                'remaining_columns': len(data.columns)
            }
        
        # Check for required columns
        required_cols = ['cnsp', 'drug', 'patient']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found in dataset: {missing_cols}")
        
        # Apply filters if specified
        initial_count = len(data)
        
        if cnsp_filter is not None:
            if not isinstance(cnsp_filter, list):
                cnsp_filter = [cnsp_filter]
            mask = data['cnsp'].isin(cnsp_filter)
            data = data[mask].copy()
            metadata['filters_applied']['cnsp'] = cnsp_filter
            log_message(f"Filtered by cnsp: {cnsp_filter}. Rows remaining: {len(data)}/{initial_count}")
        
        if drug_name is not None:
            if not isinstance(drug_name, list):
                drug_filter = [drug_name]
            else:
                drug_filter = drug_name
            mask = data['drug'].isin(drug_filter)
            data = data[mask].copy()
            metadata['filters_applied']['drug'] = drug_filter
            log_message(f"Filtered by drug: {drug_filter}. Rows remaining: {len(data)}/{initial_count}")
        
        # Check for missing values in required columns
        required_values = ['cnsp', 'drug', 'patient', target_col]
        missing_mask = data[required_values].isna().any(axis=1)
        if missing_mask.any():
            missing_data = data[missing_mask][['cnsp', 'patient', 'drug']].drop_duplicates()
            for _, row in missing_data.iterrows():
                entry = f"Missing required values for cnsp={row['cnsp']}, patient={row['patient']}, drug={row['drug']}"
                metadata['excluded_participants'].append(entry)
            data = data[~missing_mask].copy()
            log_message(f"Excluded {missing_mask.sum()} rows with missing required values")
        
        # Normalize target value if more than one drug is being analysed #######################################
        if data["cnsp"].nunique() > 1:
            log_message(f"Z-Score normalizing target column by CNSP")
            norm_target_col = data[target_col]
            for norm_cnsp in data["cnsp"]:
                norm_cnsp_rows = data["cnsp"] == norm_cnsp
                
                norm_cnsp_mean = norm_target_col[norm_cnsp_rows].mean()
                norm_cnsp_std = norm_target_col[norm_cnsp_rows].std()
                data.loc[norm_cnsp_rows, target_col] = (data.loc[norm_cnsp_rows, target_col] - norm_cnsp_mean) / norm_cnsp_std

        # Identify PK predictor columns
        all_columns = set(data.columns)
        pk_columns = set()
        
        # Find all potential PK parameter columns (those starting with 'ref_' or 'test_')
        all_pk_columns = {col for col in all_columns if col.startswith(('ref_', 'test_'))}
        
        if pk_predictors:
            # Use exact column names as provided in pk_predictors
            for col in pk_predictors:
                if col in all_columns:
                    pk_columns.add(col)
                else:
                    log_message(f"Warning: PK predictor column not found: {col}", warning=True)
            
            # Check for missing PK columns
            missing_pk = [col for col in pk_predictors if col not in all_columns]
            if missing_pk:
                log_message(f"Warning: The following PK predictors were not found: {missing_pk}", warning=True)
        
        # If exclude_pk_params is True, add all PK columns to columns_to_drop unless they're in pk_predictors
        if all_pk_columns:
            if columns_to_drop is None:
                columns_to_drop = []
            # Only exclude PK parameters that are not explicitly included in pk_predictors
            pk_to_exclude = all_pk_columns - pk_columns
            if pk_to_exclude:
                log_message(f"Excluding {len(pk_to_exclude)} PK parameters (not in pk_predictors): {', '.join(sorted(pk_to_exclude))}")
                columns_to_drop.extend(pk_to_exclude)
        
        # Identify columns to drop
        default_drop = ['cnsp', 'drug']  # Keep 'patient' by default
        if columns_to_drop is None:
            columns_to_drop = default_drop.copy()
        else:
            columns_to_drop = list(set(columns_to_drop + default_drop))
        
        # Ensure we don't drop target column or PK predictors
        columns_to_drop = [col for col in columns_to_drop 
                          if col != target_col and col not in pk_columns]
        
        # Drop columns
        columns_before_drop = set(data.columns)
        data = data.drop(columns=columns_to_drop, errors='ignore')
        dropped_columns = columns_before_drop - set(data.columns)
        
        if dropped_columns:
            log_message(f"Complete list of dropped columns: {sorted(dropped_columns)}")
        
        # Check for remaining missing values
        missing_values = data.isna().sum()
        missing_columns = missing_values[missing_values > 0]
        
        if not missing_columns.empty:
            log_message("Warning: The following columns contain missing values:", warning=True)
            for col, count in missing_columns.items():
                log_message(f"  - {col}: {count} missing values", warning=True)
        
        # Separate features and target
        X = data.drop(columns=[target_col], errors='ignore')
        y = data[target_col]
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Update metadata
        metadata['columns_used'] = {
            'predictors': {
                'numerical': numerical_cols,
                'categorical': categorical_cols,
                'pk_predictors': list(pk_columns)
            },
            'target': target_col,
            'dropped_columns': list(dropped_columns)
        }
        
        log_message(f"Data loaded successfully. Shape: {X.shape}")
        log_message(f"Numerical columns ({len(numerical_cols)}): {', '.join(numerical_cols) if numerical_cols else 'None'}")
        log_message(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'None'}")
        
        return X, y, numerical_cols, categorical_cols, base_log_dir
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        log_message(error_msg, warning=True)
        raise RuntimeError(error_msg) from e


def plot_feature_importance(plot_path, results:pd.DataFrame, top_n=20, additional_text=None):
    """
    Generate and save feature importance visualizations.
    
    Args:
        run_dir: Directory to save the plots
        results: Results dictionary containing feature importance data
        top_n: Number of top features to display in the plot
    """
    column_width = 0.8
    subset_color = {
        'total': 'silver',
        'train': 'tab:blue',
        'test' : 'tab:orange'
    }
    x_pos = np.linspace(0, top_n * 4 * column_width, top_n)
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get top N features for this fold
        top_features = results.nlargest(top_n, 'test_mean')
        
        offset = 0
        for subset in ('total', 'train', 'test'):
            ax.bar(
                x_pos + offset,
                top_features[f'{subset}_mean'],
                width=column_width,
                yerr=top_features[f'{subset}_std'],
                capsize=4,
                color=subset_color[subset],
                label=f"{subset} data"
            )
            offset += column_width
        ax.set_xticks(x_pos + column_width)
        ax.set_xticklabels(featureDict['en'][top_features.index], rotation=45, ha="right")
        
        ax.set_ylabel('Permutation Importance ($r^2$)')
        ax.set_xlabel('Feature')
        
        ax.legend()
        
        plot_title = '{0}{1}'.format(
            f"Importance of the Top {top_n} Features" if top_n != results.shape[0] else "Feature Importance",
            f" - {additional_text}" if isinstance(additional_text, str) else ""
        )
        plt.title(plot_title, y=1.02)
        plt.tight_layout()
        
        # Save the per-fold plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting feature importance: {str(e)}")
        import traceback
        traceback.print_exc()


def calculate_feature_importance(model, X_train:pd.DataFrame, y_train, X_test, y_test, n_repeats=10, n_jobs=-1, random_state=42):
    """
    Calculate permutation importance for a given model.
    
    Args:
        model: Fitted scikit-learn model or pipeline
        X: Feature matrix
        y: Target variable
        n_repeats: Number of permutations
        n_jobs: Number of jobs to run in parallel
        
    Returns:
        dict: Dictionary containing importance metrics
    """
    try:
        X_total = pd.concat([X_train, X_test], axis='index')
        y_total = pd.concat([y_train, y_test], axis='index')
        Xy_pairs = {
            'train' : (X_train, y_train),
            'test'  : (X_test,  y_test),
            'total' : (X_total, y_total)
        }
        # Calculate permutation importance
        importance_df = pd.DataFrame({'feature': X_train.columns})
        for subset_name, (X, y) in Xy_pairs.items():
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            )
            # Prepare results
            importance_df.insert(loc=len(importance_df.columns), column=f'{subset_name}_mean', value=result.importances_mean)
            importance_df.insert(loc=len(importance_df.columns), column=f'{subset_name}_std', value=result.importances_std)
        
        importance_df = importance_df.rename(index=importance_df.pop('feature'))
        
        return importance_df
        
    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        return None
