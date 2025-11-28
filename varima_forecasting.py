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

# Darts imports
from darts import TimeSeries
from darts.models import VARIMA, LinearRegressionModel, NaiveSeasonal
from darts.metrics import smape, mase, mae, mse, rmse
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.utils.statistics import check_seasonality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Constants
DATA_DIR = Path("Price_Daily.csv")
OUTPUT_DIR = Path("output/varima_run")
TARGETS = ["Target_Naphtha_Price_Close", "Target_Ethylene_NEAsia_Price_Close"]
DATE_RANGE = ["2021-02-05", "2025-06-30"]
TEST_SPLIT_DATE = "2024-07-01"
SEASONAL_PERIOD = 26

# Create output directory
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DataIngestion:
    """
    Loads and prepares raw data for the forecasting pipeline.
    Replicates logic from src/modules/data_ingestion.py
    """
    
    def __init__(self, source_dir: Path, date_range: Optional[List[str]] = None):
        self.source_dir = source_dir
        self.date_range = date_range
        
    def load_and_resample(self) -> pd.DataFrame:
        """Load all data sources and resample to weekly frequency (W-FRI)."""
        logger.info("Loading and resampling data...")
        dfs = {}
        csv_files = list(self.source_dir.glob('*.csv'))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.source_dir}")
            
        for file_path in csv_files:
            try:
                # Load CSV
                df = pd.read_csv(file_path, parse_dates=True, index_col=0)
                
                # Resample based on filename pattern
                if 'Daily' in file_path.name:
                    df_resampled = df.resample('W-FRI').agg(['mean', 'max', 'min', 'last'])
                    df_resampled.columns = ['_'.join(col) for col in df_resampled.columns]
                elif 'Weekly' in file_path.name:
                    df_resampled = df.resample('W-FRI').last()
                elif 'Monthly' in file_path.name or 'Quarterly' in file_path.name:
                    offset = pd.offsets.MonthEnd(0) if 'Monthly' in file_path.name else pd.offsets.QuarterEnd(0)
                    df.index = df.index + offset
                    limit = 5 if 'Monthly' in file_path.name else 14
                    df_resampled = df.resample('W-FRI').mean().interpolate(method='linear', limit=limit)
                elif 'Yearly' in file_path.name:
                    df.index = df.index + pd.offsets.YearEnd(0)
                    df_resampled = df.resample('W-FRI').ffill(limit=53)
                else:
                    df_resampled = df.resample('W-FRI').mean()

                # Rename columns
                df_resampled.columns = [f"({file_path.name}){col}" for col in df_resampled.columns]
                dfs[file_path.name] = df_resampled
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                raise

        # Merge dataframes
        merged_df = pd.concat(dfs.values(), axis=1)
        
        # Filter by date range
        if self.date_range:
            merged_df = merged_df.loc[self.date_range[0]:self.date_range[1]]
            
        return merged_df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe by dropping constant columns and filling specific NaNs."""
        # Drop constant columns
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            df = df.drop(columns=constant_cols)
            
        # Fill missing values with 0 for SD data
        for col_name in df.columns:
            if '(SD_' in col_name:
                df[col_name] = df[col_name].fillna(0)
                
        return df

    def split_data(self, df: pd.DataFrame, test_split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train and test sets."""
        test_split = pd.to_datetime(test_split_date)
        train_df = df.loc[df.index < test_split].copy()
        test_df = df.loc[df.index >= test_split].copy()
        return train_df, test_df


class DataPreparation:
    """
    Handles data preprocessing including imputation and decomposition.
    """
    
    def __init__(self, target_cols: List[str]):
        self.target_cols = target_cols
        
    def impute_missing(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Impute missing values using linear interpolation."""
        # Remove columns with too many missing values (>40%)
        missing_pct = train_df.isnull().sum() / len(train_df)
        cols_to_keep = missing_pct[missing_pct <= 0.4].index.tolist()
        
        train_df = train_df[cols_to_keep].copy()
        test_df = test_df[cols_to_keep].copy()
        
        # Linear interpolation
        test_split = test_df.index[0]
        combined_df = pd.concat([train_df, test_df]).interpolate(method='linear')
        
        train_df = combined_df.loc[combined_df.index < test_split].copy()
        test_df = combined_df.loc[combined_df.index >= test_split].copy()
        
        # Drop remaining NaNs
        train_df = train_df.dropna(axis=1)
        test_df = test_df[train_df.columns] # Keep same columns
        
        return train_df, test_df

    def decompose(self, df: pd.DataFrame, period: int = 26) -> Dict[str, pd.DataFrame]:
        """
        Decompose target series into trend, seasonal, and residual components.
        Returns a dictionary with components for each target.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        components = {}
        for target in self.target_cols:
            if target not in df.columns:
                continue
                
            result = seasonal_decompose(df[target], model='additive', period=period, two_sided=False, extrapolate_trend=0)
            
            comp_df = pd.DataFrame(index=df.index)
            comp_df['trend'] = result.trend
            comp_df['seasonal'] = result.seasonal
            comp_df['residual'] = result.resid
            
            # Drop NaNs created by decomposition
            comp_df = comp_df.dropna()
            components[target] = comp_df
            
        return components


class Evaluator:
    """
    Evaluates model predictions with multiple metrics.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate SMAPE, MDA, MAE, RMSE."""
        # Ensure 1D arrays for simple metrics
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # SMAPE
        smape_val = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
        
        # MDA
        true_diffs = y_true[1:] - y_true[:-1]
        pred_diffs = y_pred[1:] - y_true[:-1]
        true_trends = np.sign(true_diffs)
        pred_trends = np.sign(pred_diffs)
        mda_val = np.mean(true_trends == pred_trends) * 100
        
        # MAE
        mae_val = np.mean(np.abs(y_true - y_pred))
        
        # RMSE
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        return {
            "smape": smape_val,
            "mda": mda_val,
            "mae": mae_val,
            "rmse": rmse_val
        }


class Visualization:
    """
    Generates plots for the pipeline.
    """
    
    @staticmethod
    def plot_decomposition(components: Dict[str, pd.DataFrame], output_dir: Path):
        """Plot decomposition components."""
        for target, df in components.items():
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            df['trend'].plot(ax=axes[0], title=f'{target} - Trend')
            df['seasonal'].plot(ax=axes[1], title=f'{target} - Seasonal')
            df['residual'].plot(ax=axes[2], title=f'{target} - Residual')
            plt.tight_layout()
            plt.savefig(output_dir / f"{target}_decomposition.png")
            plt.close()

    @staticmethod
    def plot_forecast(y_true: pd.Series, y_pred: pd.Series, target_name: str, output_dir: Path):
        """Plot actual vs forecast."""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual')
        plt.plot(y_pred.index, y_pred.values, label='Forecast', color='red')
        plt.title(f'Forecast vs Actual: {target_name}')
        plt.legend()
        plt.savefig(output_dir / f"{target_name}_forecast.png")
        plt.close()


def run_varima_pipeline():
    """
    Main execution pipeline.
    """
    logger.info("Starting VARIMA Forecasting Pipeline")
    
    # 1. Data Ingestion
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
    prep = DataPreparation(targets)
    train_df, test_df = prep.impute_missing(train_df, test_df)
    
    # Decompose (on combined data to ensure continuity, then split)
    full_df = pd.concat([train_df, test_df])
    components = prep.decompose(full_df, period=SEASONAL_PERIOD)
    
    # Visualization: Decomposition
    Visualization.plot_decomposition(components, OUTPUT_DIR)
    
    # 3. Modeling & Forecasting
    
    # Prepare Residuals for VARIMA (Multivariate)
    train_residuals = []
    test_residuals = []
    
    for target in targets:
        if target not in components:
            continue
        
        comp = components[target]
        # Split components back to train/test (align indices)
        train_idx = comp.index.intersection(train_df.index)
        test_idx = comp.index.intersection(test_df.index)
        
        train_comp = comp.loc[train_idx]
        test_comp = comp.loc[test_idx]
        
        train_residuals.append(TimeSeries.from_series(train_comp['residual']))
        test_residuals.append(TimeSeries.from_series(test_comp['residual']))
    
    if not train_residuals:
        logger.error("No targets found for modeling")
        return

    # Create multivariate series
    train_res_multi = train_residuals[0]
    test_res_multi = test_residuals[0]
    
    for i in range(1, len(train_residuals)):
        train_res_multi = train_res_multi.stack(train_residuals[i])
        test_res_multi = test_res_multi.stack(test_residuals[i])
        
    # Hyperparameter Tuning for VARIMA
    logger.info("Starting Hyperparameter Tuning for VARIMA...")
    best_model = None
    best_aic = float('inf')
    
    # Simple grid search for p, d, q
    ps = [1, 2]
    ds = [0, 1]
    qs = [0, 1]
    
    for p in ps:
        for d in ds:
            for q in qs:
                try:
                    logger.info(f"Trying VARIMA(p={p},d={d},q={q})...")
                    model = VARIMA(p=p, d=d, q=q, trend='c')
                    
                    # Validate on last 12 weeks of train
                    val_len = 12
                    train_sub, val_sub = train_res_multi[:-val_len], train_res_multi[-val_len:]
                    model.fit(train_sub)
                    pred = model.predict(n=val_len)
                    err = mse(val_sub, pred)
                    
                    if err < best_aic:
                        best_aic = err
                        best_model = VARIMA(p=p, d=d, q=q, trend='c')
                        logger.info(f"New best VARIMA(p={p},d={d},q={q}) with MSE={err:.4f}")
                        
                except Exception as e:
                    logger.warning(f"VARIMA(p={p},d={d},q={q}) failed: {str(e)}")
                    continue
    
    if best_model is None:
        logger.warning("Hyperparameter tuning failed, using default VARIMA(1,0,0)")
        best_model = VARIMA(p=1, d=0, q=0)
        
    # Fit best model on full train residuals
    logger.info("Fitting best VARIMA model on full train residuals...")
    best_model.fit(train_res_multi)
    
    # Historical Forecasts (on Test set)
    logger.info("Generating historical forecasts...")
    
    # Append test data for historical forecasts
    full_res_multi = train_res_multi.append(test_res_multi)
    
    hist_pred_res = best_model.historical_forecasts(
        series=full_res_multi,
        start=len(train_res_multi),
        forecast_horizon=1,
        stride=1,
        retrain=False,
        verbose=True
    )
    
    # Reconstruct predictions
    for i, target in enumerate(targets):
        if target not in components:
            continue
            
        comp = components[target]
        train_idx = comp.index.intersection(train_df.index)
        test_idx = comp.index.intersection(test_df.index)
        
        train_comp = comp.loc[train_idx]
        test_comp = comp.loc[test_idx]
        
        # 1. Forecast Trend (Linear Regression)
        trend_train = TimeSeries.from_series(train_comp['trend'])
        trend_model = LinearRegressionModel(lags=4)
        trend_model.fit(trend_train)
        
        full_trend = TimeSeries.from_series(comp['trend'])
        trend_pred = trend_model.historical_forecasts(
            series=full_trend,
            start=pd.Timestamp("2022-02-04"),
            forecast_horizon=1,
            stride=1,
            retrain=True
        )
        
        # 2. Forecast Seasonal (NaiveSeasonal)
        seasonal_train = TimeSeries.from_series(train_comp['seasonal'])
        seasonal_model = NaiveSeasonal(K=SEASONAL_PERIOD)
        seasonal_model.fit(seasonal_train)
        
        full_seasonal = TimeSeries.from_series(comp['seasonal'])
        seasonal_pred = seasonal_model.historical_forecasts(
            series=full_seasonal,
            start=pd.Timestamp("2022-02-04"),
            forecast_horizon=1,
            stride=1,
            retrain=True
        )
        
        # 3. Get Residual Prediction
        res_pred = hist_pred_res.univariate_component(i)
        
        # Align indices
        common_index = trend_pred.time_index.intersection(seasonal_pred.time_index).intersection(res_pred.time_index)
        
        trend_vals = trend_pred.slice(common_index[0], common_index[-1]).values().flatten()
        seasonal_vals = seasonal_pred.slice(common_index[0], common_index[-1]).values().flatten()
        res_vals = res_pred.slice(common_index[0], common_index[-1]).values().flatten()
        
        final_pred_vals = trend_vals + seasonal_vals + res_vals
        final_pred_series = pd.Series(final_pred_vals, index=common_index)
        
        # Get Actuals
        actuals = test_df.loc[common_index, target]
        
        # Evaluate
        metrics = Evaluator.calculate_metrics(actuals.values, final_pred_vals)
        logger.info(f"Results for {target}: {metrics}")
        
        # Save metrics
        with open(OUTPUT_DIR / f"{target}_metrics.txt", "w") as f:
            f.write(str(metrics))
            
        # Plot
        Visualization.plot_forecast(actuals, final_pred_series, target, OUTPUT_DIR)
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_varima_pipeline()
