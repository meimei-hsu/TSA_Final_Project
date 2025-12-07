from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List, Union
import logging
import warnings
import shutil
import json

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Darts imports
from darts import TimeSeries
from darts.models import VARIMA, Theta, NaiveSeasonal
from darts.metrics import smape, mase, mae, mse, rmse
from darts.dataprocessing.transformers import Scaler, Diff, MissingValuesFiller
from darts.utils.statistics import check_seasonality
from darts.utils.utils import SeasonalityMode

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
        
        # Load CSV
        try:
            df = pd.read_csv(self.source_dir, parse_dates=True, index_col=0)
        except Exception as e:
            logger.error(f"Error processing {self.source_dir}: {str(e)}")
            raise

        # Resample to weekly frequency (W-FRI)
        df_resampled = df.resample('W-FRI').agg(['mean', 'max', 'min', 'last'])
        df_resampled.columns = ['_'.join(col) for col in df_resampled.columns]

        # Filter by date range
        if self.date_range:
            df_resampled = df_resampled.loc[self.date_range[0]:self.date_range[1]]
            
        return df_resampled

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

    def prepare_multivariate_series(self, data: Dict[str, pd.DataFrame], train_index: pd.Index, 
                                    test_index: pd.Index, col_name: str) -> Tuple[TimeSeries, TimeSeries]:
        """
        Prepare multivariate TimeSeries from a dictionary of DataFrames.
        """
        # Collect relevant series
        series_list = []
        for target in self.target_cols:
            if target in data and col_name in data[target].columns:
                series_list.append(data[target][col_name].rename(target))
        
        if not series_list:
            raise ValueError(f"No valid data found for column '{col_name}'")
            
        # Create unified DataFrame (inner join for alignment)
        multi_df = pd.concat(series_list, axis=1, join='inner')
        
        # Split based on provided indices
        train_df = multi_df.loc[multi_df.index.isin(train_index)]
        test_df = multi_df.loc[multi_df.index.isin(test_index)]
        
        if train_df.empty:
            raise ValueError("Training set is empty after alignment")
            
        # Convert to Darts TimeSeries
        return TimeSeries.from_dataframe(train_df), TimeSeries.from_dataframe(test_df)


class ModelTraining:
    """
    Handles model training and hyperparameter tuning.
    """
    @staticmethod
    def perform_grid_search(
        model_entity, 
        parameters: Dict[str, Any], 
        series: TimeSeries, 
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        metric=smape,
        val_len: int = 12
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform grid search with validation on the last `val_len` points of `series`.
        """
        train = series[:-val_len]
        val = series[-val_len:]
        
        return model_entity.gridsearch(
            parameters=parameters,
            series=train,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            val_series=val,
            metric=metric
        )

    @staticmethod
    def forecast_trend(
        series: pd.Series, 
        train_split_date: Any
    ) -> TimeSeries:
        """
        Fits a Theta model to extrapolate the trend. 
        """
        full_trend_ts = TimeSeries.from_series(series)
        train_trend_ts = full_trend_ts.drop_after(train_split_date)
        
        # Theta(2) is equivalent to linear extrapolation
        model = Theta(theta=2, season_mode=SeasonalityMode.NONE) 
        
        return model.historical_forecasts(
            series=full_trend_ts,
            start=len(train_trend_ts),
            forecast_horizon=1,
            stride=1,
            retrain=True,
            verbose=False
        )

    @staticmethod
    def forecast_seasonal(
        series: pd.Series, 
        train_split_date: Any, 
        period: int
    ) -> TimeSeries:
        """
        Forecast seasonal component using NaiveSeasonal.
        """
        full_seasonal_ts = TimeSeries.from_series(series)
        train_seasonal_ts = full_seasonal_ts.drop_after(train_split_date)
        
        seasonal_model = NaiveSeasonal(K=period)
        seasonal_model.fit(train_seasonal_ts)
        
        return seasonal_model.historical_forecasts(
            series=full_seasonal_ts,
            start=len(train_seasonal_ts),
            forecast_horizon=1,
            stride=1,
            retrain=True,
            verbose=False
        )

    @staticmethod
    def combine_predictions(predictions: List[TimeSeries]) -> TimeSeries:
        """
        Align multiple TimeSeries on their common time index and sum them up.
        """
        if not predictions:
            raise ValueError("No predictions provided")
            
        # Find common intersection
        common_index = predictions[0].time_index
        for ts in predictions[1:]:
            common_index = common_index.intersection(ts.time_index)
            
        if common_index.empty:
            raise ValueError("No overlapping time index found")
            
        # Sum values on common index
        final_values = sum(
            ts.slice(common_index[0], common_index[-1]).values().flatten() 
            for ts in predictions
        )
            
        return TimeSeries.from_times_and_values(common_index, final_values)


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
    def plot_forecast_components(
        target_name: str,
        y_true: pd.Series,
        y_pred: pd.Series,
        actual_components: pd.DataFrame,
        pred_components: Dict[str, pd.Series],
        output_dir: Path
    ):
        """Plot Actual vs Forecast for Total and Components."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Total
        axes[0].plot(y_true.index, y_true.values, label='Actual', color='black')
        axes[0].plot(y_pred.index, y_pred.values, label='Forecast', color='red', linestyle='--')
        axes[0].set_title(f'{target_name} - Total')
        axes[0].legend()
        
        # Trend
        axes[1].plot(actual_components.index, actual_components['trend'].values, label='Actual Trend')
        axes[1].plot(pred_components['trend'].index, pred_components['trend'].values, label='Forecast Trend', color='red', linestyle='--')
        axes[1].set_title(f'{target_name} - Trend')
        axes[1].legend()

        # Seasonal
        axes[2].plot(actual_components.index, actual_components['seasonal'].values, label='Actual Seasonal')
        axes[2].plot(pred_components['seasonal'].index, pred_components['seasonal'].values, label='Forecast Seasonal', color='red', linestyle='--')
        axes[2].set_title(f'{target_name} - Seasonal')
        axes[2].legend()

        # Residual
        axes[3].plot(actual_components.index, actual_components['residual'].values, label='Actual Residual')
        axes[3].plot(pred_components['residual'].index, pred_components['residual'].values, label='Forecast Residual', color='red', linestyle='--')
        axes[3].set_title(f'{target_name} - Residual')
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{target_name}_forecast_components.png")
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
    try:
        train_res_multi, test_res_multi = prep.prepare_multivariate_series(
            components, 
            train_df.index, 
            test_df.index,
            col_name='residual'
        )
        full_res_multi = train_res_multi.append(test_res_multi)
    except ValueError as e:
        logger.error(str(e))
        return

    # Differencing
    diff_trasformer = Diff()
    train_res_cov = diff_trasformer.fit_transform(train_res_multi)
    full_res_cov = diff_trasformer.transform(full_res_multi)

    missing_filler = MissingValuesFiller(fill=0.0)
    train_res_future_cov = missing_filler.transform(train_res_cov.shift(1))
    full_res_future_cov = missing_filler.transform(full_res_cov.shift(1))

    # Align target and covariates to intersection (fixes start date mismatch due to shift/diff)
    train_res_multi = train_res_multi.slice_intersect(train_res_future_cov)
    train_res_future_cov = train_res_future_cov.slice_intersect(train_res_multi)
    
    full_res_multi = full_res_multi.slice_intersect(full_res_future_cov)
    full_res_future_cov = full_res_future_cov.slice_intersect(full_res_multi)
        
    # Hyperparameter Tuning for VARIMA
    parameters = {
        'p': [1, 2, 3],
        'd': [0],
        'q': [0, 1, 2],
        'trend': ['n', 'c']
    }
    
    try:
        best_model, best_params, best_score = ModelTraining.perform_grid_search(
            VARIMA, parameters, train_res_multi, future_covariates=train_res_future_cov
        )
        logger.info(f"Best VARIMA parameters: {best_params} with MSE={best_score}")
        
    except Exception as e:
        logger.warning(f"Grid search failed: {str(e)}. Using default VARIMA(1,0,0)")
        best_model = VARIMA(p=1, d=0, q=0, trend='n')
        
    # Fit best model on full train residuals
    best_model.fit(train_res_multi, future_covariates=train_res_future_cov)
    
    # Historical Forecasts (on Test set)
    hist_pred_res = best_model.historical_forecasts(
        series=full_res_multi,
        future_covariates=full_res_future_cov,
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
            
        # 1. Component Predictions
        res_pred = hist_pred_res.univariate_component(i)

        trend_pred = ModelTraining.forecast_trend(
            series=components[target]['trend'],
            train_split_date=train_df.index[-1]
        )
        
        seasonal_pred = ModelTraining.forecast_seasonal(
            series=components[target]['seasonal'],
            train_split_date=train_df.index[-1],
            period=SEASONAL_PERIOD
        )
        
        # 2. Reconstruct Prediction
        final_pred_series = ModelTraining.combine_predictions([trend_pred, seasonal_pred, res_pred])
        common_index = final_pred_series.time_index

        # Get Actuals
        actuals = full_df.loc[common_index, target]
        
        # Evaluate
        metrics = Evaluator.calculate_metrics(actuals.values, final_pred_series.values())
        logger.info(f"Results for {target}: {metrics}")
        
        # Save metrics
        with open(OUTPUT_DIR / f"{target}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
            
        # Plot
        preds = pd.Series(final_pred_series.values().flatten(), index=common_index)
        
        act_comps = components[target].loc[common_index]
        pred_comps = {
            'trend': pd.Series(trend_pred.values().flatten(), index=trend_pred.time_index),
            'seasonal': pd.Series(seasonal_pred.values().flatten(), index=seasonal_pred.time_index),
            'residual': pd.Series(res_pred.values().flatten(), index=res_pred.time_index)
        }
        
        Visualization.plot_forecast_components(target, actuals, preds, act_comps, pred_comps, OUTPUT_DIR)
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_varima_pipeline()
