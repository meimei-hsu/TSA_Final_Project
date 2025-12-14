import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union
import logging
import warnings
import shutil
import sys
from itertools import product
from tqdm import tqdm

# Statistical tests
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf

# Darts imports
from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mse, mae, rmse, mape, smape
from darts.dataprocessing.transformers import Scaler

# PyTorch Lightning callbacks for early stopping
from pytorch_lightning.callbacks import EarlyStopping

# Reuse classes from varima_forecasting
from varima_forecasting import DataIngestion, DataPreparation, Evaluator, Visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Constants
DATA_DIR = Path(".")
OUTPUT_DIR = Path("output/lstm_run")
TARGETS = ["Target_Naphtha_Price_Close", "Target_Ethylene_NEAsia_Price_Close"]
DATE_RANGE = ["2021-02-05", "2025-06-30"]
TEST_SPLIT_DATE = "2024-07-01"
FORECAST_HORIZON = 1
VALIDATION_WEEKS = 12

# Create output directory (only affects lstm_run)
if OUTPUT_DIR.exists():
    logger.warning(f"Output directory {OUTPUT_DIR} already exists. Cleaning it...")
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PACFAnalyzer:
    """
    Handles PACF analysis for determining optimal lag parameters.
    """
    
    @staticmethod
    def identify_pacf_lag(series: pd.Series, max_lags: int = 20, 
                         series_name: str = "series") -> int:
        """
        Identify optimal lag parameter from PACF analysis.
        """
        # Calculate PACF
        pacf_values = pacf(series.dropna(), nlags=max_lags, method='ywm')
        
        # Calculate confidence interval
        n = len(series.dropna())
        confidence_interval = 1.96 / np.sqrt(n)
        
        # Find significant lags
        significant_lags = []
        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i]) > confidence_interval:
                significant_lags.append(i)
        
        # Determine p
        if significant_lags:
            p = max(significant_lags)
        else:
            p = 1  # Default to 1 if no significant lags found
        
        # Cap at reasonable value
        p = min(p, max_lags // 2)
        
        logger.info(f"PACF analysis: identified p={p} (significant lags: {significant_lags[:5]}...)")
        
        # Save PACF plot
        fig = plt.figure(figsize=(10, 4))
        plot_pacf(series.dropna(), lags=max_lags, method='ywm')
        
        short_name = "Unknown"
        if "Ethylene" in series_name:
            short_name = "Ethylene"
        elif "Naphtha" in series_name:
            short_name = "Naphtha"
        
        title = f'PACF - {short_name} (Original)'
        plt.title(title)
        plt.tight_layout()
        
        safe_name = series_name.replace('(', '').replace(')', '').replace('.csv', '').replace(' ', '_')
        plt.savefig(OUTPUT_DIR / f'pacf_{safe_name}.png')
        plt.close()
        
        return p


class LSTMForecaster:
    """
    LSTM model with grid search for hyperparameter tuning.
    """
    
    def __init__(self, output_chunk_length: int = 1):
        self.output_chunk_length = output_chunk_length
        self.best_params = None
        self.best_model = None
        self.scaler = None
        
    def grid_search(
        self,
        train_series: TimeSeries,
        base_input_chunk: int,
        target_name: str = "series",
        hidden_dims: List[int] = [32, 64, 128, 256],
        n_layers_list: List[int] = [1, 2, 3, 4],
        input_chunk_multipliers: List[int] = [2, 4, 8, 12, 16, 32, 64],
        batch_sizes: List[int] = [16, 32],
        val_weeks: int = 12
    ) -> Dict[str, Any]:
        """
        Grid search for LSTM hyperparameters.
        """
        logger.info("Starting LSTM hyperparameter grid search...")
        
        # Split train into train_sub and validation
        train_sub = train_series[:-val_weeks]
        val_sub = train_series[-val_weeks:]
        
        best_smape = float('inf')
        results = []
        
        # Generate all combinations
        param_combinations = list(product(
            hidden_dims,
            n_layers_list,
            input_chunk_multipliers,
            batch_sizes
        ))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations...")
        
        # Use tqdm progress bar
        pbar = tqdm(param_combinations, desc="Grid Search Progress")
        for hidden_dim, n_layers, multiplier, batch_size in pbar:
            input_chunk = base_input_chunk * multiplier
            
            # Enforce minimum input chunk size
            MIN_INPUT_CHUNK = 8
            input_chunk = max(input_chunk, MIN_INPUT_CHUNK)
            
            # Skip if input chunk is too large
            if input_chunk >= len(train_sub):
                pbar.set_postfix_str(f"Skipped (chunk too large)")
                continue
            
            try:
                pbar.set_description(f"h={hidden_dim}, l={n_layers}, c={input_chunk}, b={batch_size}")
                
                model = RNNModel(
                    model='LSTM',
                    hidden_dim=hidden_dim,
                    n_rnn_layers=n_layers,
                    input_chunk_length=input_chunk,
                    output_chunk_length=self.output_chunk_length,
                    dropout=0.1,
                    n_epochs=100,
                    batch_size=batch_size,
                    random_state=42,
                    optimizer_kwargs={'lr': 0.001},
                    pl_trainer_kwargs={
                        "accelerator": "gpu",
                        "devices": [1],
                        "enable_progress_bar": False,
                        "enable_model_summary": False
                    }
                )
                
                model.fit(train_sub, verbose=False)
                pred = model.predict(n=len(val_sub))
                val_smape = smape(val_sub, pred)
                
                results.append({
                    'hidden_dim': hidden_dim,
                    'n_layers': n_layers,
                    'input_chunk': input_chunk,
                    'batch_size': batch_size,
                    'smape': val_smape
                })
                
                pbar.set_postfix_str(f"SMAPE={val_smape:.4f}")
                
                if val_smape < best_smape:
                    best_smape = val_smape
                    self.best_params = {
                        'hidden_dim': hidden_dim,
                        'n_rnn_layers': n_layers,
                        'input_chunk_length': input_chunk,
                        'output_chunk_length': self.output_chunk_length,
                        'batch_size': batch_size
                    }
                    logger.info(f"  -> New best! SMAPE: {best_smape:.4f}")
                    pbar.set_postfix_str(f"New best! SMAPE={best_smape:.4f}")
                    
            except Exception as e:
                pbar.set_postfix_str(f"Failed")
                continue
        
        pbar.close()
        
        if self.best_params is None:
            logger.warning("Grid search failed to find valid parameters, using defaults")
            self.best_params = {
                'hidden_dim': 64,
                'n_rnn_layers': 2,
                'input_chunk_length': max(base_input_chunk * 4, 16),
                'output_chunk_length': self.output_chunk_length,
                'batch_size': 16
            }
        
        logger.info(f"Best parameters: {self.best_params} with SMAPE: {best_smape:.4f}")
        
        # Save grid search results
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values('smape')
            safe_target_name = target_name.replace('(', '').replace(')', '').replace('.csv', '').replace(' ', '_')
            grid_search_file = OUTPUT_DIR / f'grid_search_{safe_target_name}.csv'
            results_df.to_csv(grid_search_file, index=False)
        
        return self.best_params
    
    def train(self, train_series: TimeSeries, params: Dict[str, Any]) -> RNNModel:
        """
        Train LSTM model with given parameters.
        """
        logger.info(f"Training LSTM model with params: {params}")
        
        early_stopping = EarlyStopping(
            monitor="train_loss",
            patience=20,
            min_delta=0.0001,
            mode="min"
        )
        
        self.best_model = RNNModel(
            model='LSTM',
            hidden_dim=params['hidden_dim'],
            n_rnn_layers=params['n_rnn_layers'],
            input_chunk_length=params['input_chunk_length'],
            output_chunk_length=params['output_chunk_length'],
            dropout=0.1,
            n_epochs=300,
            batch_size=params.get('batch_size', 16),
            random_state=42,
            optimizer_kwargs={'lr': 0.001},
            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [1],
                "enable_progress_bar": True,
                "enable_model_summary": False,
                "callbacks": [early_stopping]
            }
        )
        
        self.best_model.fit(train_series, verbose=True)
        logger.info("Model training completed")
        
        return self.best_model
    
    def forecast(
        self,
        model: RNNModel,
        train_series: TimeSeries,
        test_series: TimeSeries,
        forecast_horizon: int = 1
    ) -> TimeSeries:
        """
        Generate historical forecasts on test set.
        """
        logger.info("Generating historical forecasts...")
        
        full_series = train_series.append(test_series)
        
        predictions = model.historical_forecasts(
            series=full_series,
            start=len(train_series),
            forecast_horizon=forecast_horizon,
            stride=1,
            retrain=False,
            verbose=True
        )
        
        logger.info(f"Generated {len(predictions)} forecast points")
        return predictions


def plot_comparison(results_multi: Dict, results_uni: Dict, targets: List[str], output_dir: Path):
    """
    Create comparison plots between multivariate and univariate models.
    """
    for target in targets:
        short_name = "Naphtha" if "Naphtha" in target else "Ethylene"
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Left plot: Predictions comparison
        ax1 = axes[0]
        actuals = results_multi[target]['actuals']
        pred_multi = results_multi[target]['predictions']
        pred_uni = results_uni[target]['predictions']
        
        ax1.plot(actuals.index, actuals.values, 'b-', label='Actual', linewidth=2)
        ax1.plot(pred_multi.index, pred_multi.values, 'r--', label='Multivariate LSTM', linewidth=1.5, alpha=0.8)
        ax1.plot(pred_uni.index, pred_uni.values, 'g-.', label='Univariate LSTM', linewidth=1.5, alpha=0.8)
        ax1.set_title(f'{short_name} Price Forecast Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Error comparison
        ax2 = axes[1]
        error_multi = np.abs(actuals.values - pred_multi.values)
        error_uni = np.abs(actuals.values - pred_uni.values)
        
        x = np.arange(len(actuals))
        width = 0.35
        ax2.bar(x - width/2, error_multi, width, label='Multivariate', alpha=0.8)
        ax2.bar(x + width/2, error_uni, width, label='Univariate', alpha=0.8)
        ax2.set_title(f'{short_name} Absolute Error Comparison')
        ax2.set_xlabel('Week')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{short_name}_comparison.png', dpi=150)
        plt.close()
    
    # Overall comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['SMAPE (%)', 'MDA (%)', 'MAE', 'RMSE']
    x = np.arange(len(targets) * 4)
    width = 0.35
    
    multi_values = []
    uni_values = []
    labels = []
    
    for target in targets:
        short_name = "Naphtha" if "Naphtha" in target else "Ethylene"
        for metric_name, key in [('SMAPE (%)', 'smape'), ('MDA (%)', 'mda'), ('MAE', 'mae'), ('RMSE', 'rmse')]:
            multi_values.append(results_multi[target]['metrics'][key])
            uni_values.append(results_uni[target]['metrics'][key])
            labels.append(f'{short_name}\n{metric_name}')
    
    x = np.arange(len(multi_values))
    ax.bar(x - width/2, multi_values, width, label='Multivariate', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, uni_values, width, label='Univariate', color='coral', alpha=0.8)
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Multivariate vs Univariate LSTM Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150)
    plt.close()
    
    logger.info("Comparison plots saved")


def create_comparison_table(results_multi: Dict, results_uni: Dict, targets: List[str], output_dir: Path) -> pd.DataFrame:
    """
    Create a comparison table between multivariate and univariate models.
    """
    comparison_data = []
    
    for target in targets:
        short_name = "Naphtha" if "Naphtha" in target else "Ethylene"
        
        # Multivariate results
        comparison_data.append({
            'Target': short_name,
            'Model': 'Multivariate',
            'SMAPE (%)': f"{results_multi[target]['metrics']['smape']:.2f}",
            'MDA (%)': f"{results_multi[target]['metrics']['mda']:.2f}",
            'MAE': f"{results_multi[target]['metrics']['mae']:.4f}",
            'RMSE': f"{results_multi[target]['metrics']['rmse']:.4f}",
            'Hidden_Dim': results_multi[target]['params']['hidden_dim'],
            'N_Layers': results_multi[target]['params']['n_rnn_layers'],
            'Input_Chunk': results_multi[target]['params']['input_chunk_length']
        })
        
        # Univariate results
        comparison_data.append({
            'Target': short_name,
            'Model': 'Univariate',
            'SMAPE (%)': f"{results_uni[target]['metrics']['smape']:.2f}",
            'MDA (%)': f"{results_uni[target]['metrics']['mda']:.2f}",
            'MAE': f"{results_uni[target]['metrics']['mae']:.4f}",
            'RMSE': f"{results_uni[target]['metrics']['rmse']:.4f}",
            'Hidden_Dim': results_uni[target]['params']['hidden_dim'],
            'N_Layers': results_uni[target]['params']['n_rnn_layers'],
            'Input_Chunk': results_uni[target]['params']['input_chunk_length']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'multivariate_vs_univariate_comparison.csv', index=False)
    
    return comparison_df


def run_lstm_pipeline():
    """
    Main execution pipeline for LSTM forecasting.
    Runs both multivariate and univariate models and compares them.
    """
    logger.info("="*80)
    logger.info("Starting LSTM Forecasting Pipeline (Multivariate + Univariate Comparison)")
    logger.info("="*80)
    
    # 1. Data Ingestion
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Data Ingestion")
    logger.info("="*80)
    
    ingestion = DataIngestion(DATA_DIR, DATE_RANGE)
    df = ingestion.load_and_resample()
    df = ingestion.clean(df)
    
    # Match targets to actual columns
    targets = []
    for target in TARGETS:
        matched = [col for col in df.columns if target in col]
        if matched:
            targets.append(matched[0])
            logger.info(f"Matched target {target} to column {matched[0]}")
        else:
            logger.warning(f"Target {target} not found in data columns")
    
    if not targets:
        logger.error("No targets matched")
        return
    
    # Split data
    train_df, test_df = ingestion.split_data(df, TEST_SPLIT_DATE)
    logger.info(f"Data split: Train={train_df.shape}, Test={test_df.shape}")
    
    # 2. Data Preparation
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Data Preparation")
    logger.info("="*80)
    
    prep = DataPreparation(targets)
    train_df, test_df = prep.impute_missing(train_df, test_df)
    logger.info(f"After imputation: Train={train_df.shape}, Test={test_df.shape}")
    
    # 3. PACF Analysis (per target)
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PACF Analysis")
    logger.info("="*80)
    
    pacf_analyzer = PACFAnalyzer()
    pacf_lags = {}
    for target in targets:
        p_lag = pacf_analyzer.identify_pacf_lag(
            train_df[target], 
            series_name=target
        )
        pacf_lags[target] = p_lag
        logger.info(f"Target {target}: p_lag={p_lag}")
    
    avg_pacf_lag = int(np.mean(list(pacf_lags.values())))
    logger.info(f"\nAverage PACF lag: {avg_pacf_lag}")
    
    # ========================================================================
    # PART A: MULTIVARIATE LSTM MODEL
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART A: MULTIVARIATE LSTM MODEL")
    logger.info("="*80)
    
    # Create MULTIVARIATE TimeSeries
    train_ts_multi = TimeSeries.from_dataframe(train_df[targets])
    test_ts_multi = TimeSeries.from_dataframe(test_df[targets])
    
    # Apply scaling
    scaler_multi = Scaler()
    train_ts_multi_scaled = scaler_multi.fit_transform(train_ts_multi)
    test_ts_multi_scaled = scaler_multi.transform(test_ts_multi)
    
    logger.info(f"Multivariate data shape: {train_ts_multi_scaled.values().shape}")
    
    # Grid search and train
    forecaster_multi = LSTMForecaster(output_chunk_length=FORECAST_HORIZON)
    best_params_multi = forecaster_multi.grid_search(
        train_series=train_ts_multi_scaled,
        base_input_chunk=avg_pacf_lag,
        target_name="Multivariate",
        val_weeks=VALIDATION_WEEKS
    )
    
    model_multi = forecaster_multi.train(train_ts_multi_scaled, best_params_multi)
    
    predictions_multi_scaled = forecaster_multi.forecast(
        model=model_multi,
        train_series=train_ts_multi_scaled,
        test_series=test_ts_multi_scaled,
        forecast_horizon=FORECAST_HORIZON
    )
    
    predictions_multi = scaler_multi.inverse_transform(predictions_multi_scaled)
    
    # Evaluate multivariate results
    results_multi = {}
    for target in targets:
        pred_ts = predictions_multi.univariate_component(target)
        pred_values = pred_ts.to_series()
        actual_values = test_df[target].loc[pred_values.index]
        
        common_idx = actual_values.index.intersection(pred_values.index)
        actual_values = actual_values.loc[common_idx]
        pred_values = pred_values.loc[common_idx]
        
        metrics = Evaluator.calculate_metrics(actual_values.values, pred_values.values)
        logger.info(f"[Multivariate] {target}: SMAPE={metrics['smape']:.2f}%, MDA={metrics['mda']:.2f}%")
        
        results_multi[target] = {
            'metrics': metrics,
            'params': best_params_multi,
            'predictions': pred_values,
            'actuals': actual_values
        }
    
    # ========================================================================
    # PART B: UNIVARIATE LSTM MODELS (One per target)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART B: UNIVARIATE LSTM MODELS")
    logger.info("="*80)
    
    results_uni = {}
    
    for target in targets:
        short_name = "Naphtha" if "Naphtha" in target else "Ethylene"
        logger.info(f"\n--- Training Univariate Model for {short_name} ---")
        
        # Create UNIVARIATE TimeSeries
        train_ts_uni = TimeSeries.from_series(train_df[target])
        test_ts_uni = TimeSeries.from_series(test_df[target])
        
        # Apply scaling
        scaler_uni = Scaler()
        train_ts_uni_scaled = scaler_uni.fit_transform(train_ts_uni)
        test_ts_uni_scaled = scaler_uni.transform(test_ts_uni)
        
        # Grid search and train
        forecaster_uni = LSTMForecaster(output_chunk_length=FORECAST_HORIZON)
        best_params_uni = forecaster_uni.grid_search(
            train_series=train_ts_uni_scaled,
            base_input_chunk=pacf_lags[target],
            target_name=f"Univariate_{short_name}",
            val_weeks=VALIDATION_WEEKS
        )
        
        model_uni = forecaster_uni.train(train_ts_uni_scaled, best_params_uni)
        
        predictions_uni_scaled = forecaster_uni.forecast(
            model=model_uni,
            train_series=train_ts_uni_scaled,
            test_series=test_ts_uni_scaled,
            forecast_horizon=FORECAST_HORIZON
        )
        
        predictions_uni = scaler_uni.inverse_transform(predictions_uni_scaled)
        
        # Evaluate
        pred_values = predictions_uni.to_series()
        actual_values = test_df[target].loc[pred_values.index]
        
        common_idx = actual_values.index.intersection(pred_values.index)
        actual_values = actual_values.loc[common_idx]
        pred_values = pred_values.loc[common_idx]
        
        metrics = Evaluator.calculate_metrics(actual_values.values, pred_values.values)
        logger.info(f"[Univariate] {target}: SMAPE={metrics['smape']:.2f}%, MDA={metrics['mda']:.2f}%")
        
        results_uni[target] = {
            'metrics': metrics,
            'params': best_params_uni,
            'predictions': pred_values,
            'actuals': actual_values
        }
    
    # ========================================================================
    # PART C: COMPARISON
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PART C: COMPARISON (Multivariate vs Univariate)")
    logger.info("="*80)
    
    # Create comparison table
    comparison_df = create_comparison_table(results_multi, results_uni, targets, OUTPUT_DIR)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Create comparison plots
    plot_comparison(results_multi, results_uni, targets, OUTPUT_DIR)
    
    # Save individual forecast plots
    for target in targets:
        Visualization.plot_forecast(
            results_multi[target]['actuals'], 
            results_multi[target]['predictions'], 
            f"{target}_multivariate", 
            OUTPUT_DIR
        )
        Visualization.plot_forecast(
            results_uni[target]['actuals'], 
            results_uni[target]['predictions'], 
            f"{target}_univariate", 
            OUTPUT_DIR
        )
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info("\n" + comparison_df.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("LSTM Forecasting Pipeline completed successfully!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    run_lstm_pipeline()
