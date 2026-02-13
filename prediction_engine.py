"""
Ultimate ML Prediction Engine for NBA Prop Finder v2.0
======================================================
State-of-the-art prediction system using:
- XGBoost + LightGBM + Gradient Boosting ensemble
- Optimized feature engineering (reduced overfitting)
- Distribution fit testing for probability models
- Monte Carlo simulation with optional seeding
- Adaptive ensemble weight optimization
- Transparent confidence scoring with breakdown
- Back-to-back & rest day analysis
- Blowout game filtering
- Minutes restriction detection
- HOME/AWAY performance analysis

v2.0 Improvements:
- Fixed look-ahead bias in training data
- Reduced rolling windows (3, 7, 15) to prevent overfitting
- Added distribution fit testing
- Parametric confidence scoring with transparency
- Optimized rolling calculations using pandas
- Proper error logging
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm, nbinom, beta, shapiro
from scipy.special import gammaln
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from functools import lru_cache
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ML Library imports with graceful fallbacks
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import (
        GradientBoostingRegressor, 
        RandomForestRegressor,
        HistGradientBoostingRegressor,
        StackingRegressor
    )
    from sklearn.linear_model import Ridge, BayesianRidge, QuantileRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

class PredictionConfig:
    """Central configuration for prediction parameters."""
    
    # Feature engineering windows (reduced from 5 to 3 to prevent overfitting)
    # Rationale: 3=recent, 7=weekly, 15=bi-weekly - distinct time horizons
    ROLLING_WINDOWS = [3, 7, 15]
    
    # Ensemble weights by model type (can be optimized via cross-validation)
    MODEL_WEIGHTS = {
        'xgb': 0.35,
        'lgbm': 0.25,
        'gb': 0.20,
        'ridge': 0.10,
        'bayesian': 0.10
    }
    
    # Stat-specific characteristics for adaptive modeling
    STAT_CHARACTERISTICS = {
        'PTS': {'distribution': 'normal', 'variance': 'medium', 'floor': 0, 'typical_max': 60, 'significance_threshold': 0.08},
        'REB': {'distribution': 'poisson', 'variance': 'medium', 'floor': 0, 'typical_max': 25, 'significance_threshold': 0.10},
        'AST': {'distribution': 'poisson', 'variance': 'medium', 'floor': 0, 'typical_max': 20, 'significance_threshold': 0.10},
        'STL': {'distribution': 'poisson', 'variance': 'high', 'floor': 0, 'typical_max': 6, 'significance_threshold': 0.15},
        'BLK': {'distribution': 'poisson', 'variance': 'high', 'floor': 0, 'typical_max': 8, 'significance_threshold': 0.15},
        'PRA': {'distribution': 'normal', 'variance': 'medium', 'floor': 0, 'typical_max': 80, 'significance_threshold': 0.08},
        '3PM': {'distribution': 'poisson', 'variance': 'high', 'floor': 0, 'typical_max': 12, 'significance_threshold': 0.15},
        'FG3M': {'distribution': 'poisson', 'variance': 'high', 'floor': 0, 'typical_max': 12, 'significance_threshold': 0.15}
    }
    
    # Monte Carlo simulation parameters
    MC_SIMULATIONS = 5000
    
    # Blowout detection threshold (point differential)
    BLOWOUT_THRESHOLD = 25
    
    # Minutes restriction detection
    MIN_NORMAL_MINUTES = 20
    
    # Back-to-back penalty factor
    B2B_PENALTY = 0.97
    
    # Home/Away default adjustments (used when insufficient data)
    HOME_BOOST_DEFAULT = 1.02
    AWAY_PENALTY_DEFAULT = 0.98
    
    # Minimum games for various analyses
    MIN_GAMES_FOR_ML = 6
    MIN_GAMES_FOR_VENUE_SPLIT = 3
    MIN_GAMES_FOR_TREND = 5


# =============================================================================
# HOME/AWAY PERFORMANCE ANALYZER
# =============================================================================

class HomeAwayAnalyzer:
    """
    Analyzes player's home vs away performance to calculate venue adjustments.
    Key feature for predicting performance in upcoming games.
    """
    
    @staticmethod
    def calculate_venue_splits(df: pd.DataFrame, stat_col: str) -> Dict[str, Any]:
        """
        Calculate comprehensive home/away performance splits.
        
        Returns dict with:
        - home_avg, away_avg: Average performance at each venue
        - home_std, away_std: Standard deviation at each venue  
        - differential: Points difference (home - away)
        - home_factor, away_factor: Multipliers vs overall average
        - significant: Whether the split is statistically meaningful
        - home_hit_rates, away_hit_rates: Hit rates by venue
        """
        result = {
            'home_avg': 0, 'away_avg': 0, 'overall_avg': 0,
            'home_std': 0, 'away_std': 0,
            'home_games': 0, 'away_games': 0,
            'differential': 0,
            'home_factor': 1.0, 'away_factor': 1.0,
            'significant': False,
            'home_better': None,
            'venue_impact': 'neutral'
        }
        
        if 'IS_HOME' not in df.columns or stat_col not in df.columns:
            return result
        
        home_df = df[df['IS_HOME'] == True]
        away_df = df[df['IS_HOME'] == False]
        
        home_games = len(home_df)
        away_games = len(away_df)
        
        if home_games == 0 and away_games == 0:
            return result
        
        # Calculate averages
        home_avg = home_df[stat_col].mean() if home_games > 0 else 0
        away_avg = away_df[stat_col].mean() if away_games > 0 else 0
        overall_avg = df[stat_col].mean()
        
        # Calculate standard deviations
        home_std = home_df[stat_col].std(ddof=1) if home_games > 1 else 0
        away_std = away_df[stat_col].std(ddof=1) if away_games > 1 else 0
        
        # Calculate differential
        differential = home_avg - away_avg
        
        # Calculate venue factors (multiplier vs overall)
        if overall_avg > 0:
            home_factor = home_avg / overall_avg if home_games >= PredictionConfig.MIN_GAMES_FOR_VENUE_SPLIT else PredictionConfig.HOME_BOOST_DEFAULT
            away_factor = away_avg / overall_avg if away_games >= PredictionConfig.MIN_GAMES_FOR_VENUE_SPLIT else PredictionConfig.AWAY_PENALTY_DEFAULT
        else:
            home_factor = 1.0
            away_factor = 1.0
        
        # Determine if split is statistically significant
        # Use stat-specific significance threshold
        char = PredictionConfig.STAT_CHARACTERISTICS.get(stat_col.replace('FG3M', '3PM'), {})
        significance_threshold = char.get('significance_threshold', 0.10)
        min_games_each = PredictionConfig.MIN_GAMES_FOR_VENUE_SPLIT
        
        significant = (
            home_games >= min_games_each and 
            away_games >= min_games_each and 
            overall_avg > 0 and
            abs(differential) / overall_avg > significance_threshold
        )
        
        # Determine which venue is better
        if significant:
            home_better = differential > 0
            if abs(differential) / overall_avg > 0.20:
                venue_impact = 'strong'
            elif abs(differential) / overall_avg > 0.10:
                venue_impact = 'moderate'
            else:
                venue_impact = 'slight'
        else:
            home_better = None
            venue_impact = 'neutral'
        
        result.update({
            'home_avg': round(home_avg, 2),
            'away_avg': round(away_avg, 2),
            'overall_avg': round(overall_avg, 2),
            'home_std': round(home_std, 2),
            'away_std': round(away_std, 2),
            'home_games': home_games,
            'away_games': away_games,
            'differential': round(differential, 2),
            'home_factor': round(home_factor, 4),
            'away_factor': round(away_factor, 4),
            'significant': significant,
            'home_better': home_better,
            'venue_impact': venue_impact
        })
        
        return result
    
    @staticmethod
    def get_venue_adjustment(splits: Dict, is_home_game: bool, use_default: bool = True) -> Tuple[float, str]:
        """
        Get the venue adjustment factor for the upcoming game.
        
        Returns:
        - adjustment_factor: Multiplier to apply to prediction
        - reason: Explanation for the adjustment
        """
        if not splits.get('significant', False):
            # Not enough data for confident split, use conservative defaults
            if use_default:
                if is_home_game:
                    return PredictionConfig.HOME_BOOST_DEFAULT, "Default home boost (limited data)"
                else:
                    return PredictionConfig.AWAY_PENALTY_DEFAULT, "Default away adjustment (limited data)"
            else:
                return 1.0, "Neutral (insufficient venue data)"
        
        # Use calculated factors
        if is_home_game:
            factor = splits['home_factor']
            diff = splits['differential']
            impact = splits['venue_impact']
            if diff > 0:
                reason = f"Home boost: +{diff:.1f} avg at home ({impact})"
            else:
                reason = f"Worse at home: {diff:.1f} vs road ({impact})"
        else:
            factor = splits['away_factor']
            diff = -splits['differential']  # Flip for away perspective
            impact = splits['venue_impact']
            if diff > 0:
                reason = f"Road boost: +{abs(splits['differential']):.1f} better away ({impact})"
            else:
                reason = f"Road penalty: {splits['differential']:.1f} worse away ({impact})"
        
        return factor, reason
    
    @staticmethod
    def calculate_venue_hit_rates(df: pd.DataFrame, stat_col: str, thresholds: List[int]) -> Dict[str, Dict[str, float]]:
        """Calculate hit rates for thresholds split by venue."""
        result = {'home': {}, 'away': {}}
        
        if 'IS_HOME' not in df.columns or stat_col not in df.columns:
            return result
        
        home_df = df[df['IS_HOME'] == True]
        away_df = df[df['IS_HOME'] == False]
        
        for t in thresholds:
            # Home hit rate
            if len(home_df) >= 2:
                result['home'][str(t)] = round((home_df[stat_col] >= t).mean() * 100, 1)
            else:
                result['home'][str(t)] = None
            
            # Away hit rate  
            if len(away_df) >= 2:
                result['away'][str(t)] = round((away_df[stat_col] >= t).mean() * 100, 1)
            else:
                result['away'][str(t)] = None
        
        return result


# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

class UltimateFeatureEngineer:
    """
    Extracts 80+ predictive features from player game logs.
    Includes advanced temporal, contextual, and statistical features.
    """
    
    @staticmethod
    def parse_minutes(m) -> float:
        """Parse minutes from various formats."""
        if pd.isna(m):
            return 0.0
        if isinstance(m, (int, float)):
            return float(m)
        if isinstance(m, str):
            if ':' in m:
                parts = m.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            try:
                return float(m)
            except:
                return 0.0
        return 0.0
    
    @classmethod
    def compute_rolling_features_fast(cls, series: pd.Series) -> Dict[str, float]:
        """Compute comprehensive rolling statistics using pandas (optimized in C)."""
        features = {}
        
        # Convert to pandas Series if it isn't already
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        for w in PredictionConfig.ROLLING_WINDOWS:
            suffix = f'_{w}'
            rolling = series.rolling(window=w, min_periods=1)
            
            features[f'roll_mean{suffix}'] = rolling.mean().iloc[-1]
            features[f'roll_std{suffix}'] = rolling.std().iloc[-1] if len(series) > 1 else 0
            features[f'roll_min{suffix}'] = rolling.min().iloc[-1]
            features[f'roll_max{suffix}'] = rolling.max().iloc[-1]
            features[f'roll_median{suffix}'] = rolling.median().iloc[-1]
            features[f'roll_range{suffix}'] = (rolling.max() - rolling.min()).iloc[-1]
            
            # Handle quantiles carefully for small windows
            if w >= 4:  # Need at least 4 points for meaningful quartiles
                features[f'roll_p25{suffix}'] = rolling.quantile(0.25).iloc[-1]
                features[f'roll_p75{suffix}'] = rolling.quantile(0.75).iloc[-1]
            else:
                # Use min/max as proxies for small windows
                features[f'roll_p25{suffix}'] = rolling.min().iloc[-1]
                features[f'roll_p75{suffix}'] = rolling.max().iloc[-1]
                
            features[f'roll_p90{suffix}'] = rolling.quantile(0.90).iloc[-1] if w >= 2 else rolling.max().iloc[-1]
        
        return features

    @classmethod
    def compute_rolling_features(cls, values: List[float]) -> Dict[str, float]:
        """Compute comprehensive rolling statistics."""
        # Use the optimized pandas version
        series = pd.Series(values)
        return cls.compute_rolling_features_fast(series)
    
    @classmethod
    def compute_trend_features(cls, values: List[float]) -> Dict[str, float]:
        """Compute advanced trend and momentum indicators with significance testing."""
        features = {}
        arr = np.array(values)
        n = len(arr)
        
        if n < 2:
            return {
                'trend_slope': 0, 'trend_r2': 0, 'trend_significant': 0, 'trend_p_value': 1,
                'trend_acceleration': 0, 'momentum_3': 0, 'momentum_5': 0, 'momentum_10': 0,
                'ema_ratio': 1, 'breakout_score': 0, 'mean_reversion_signal': 0,
                'streak_length': 0, 'streak_direction': 0
            }
        
        # Linear trend
        x = np.arange(n)
        if n >= PredictionConfig.MIN_GAMES_FOR_TREND:  # At least 5 games for trend analysis
            coeffs = np.polyfit(x, arr, 1)
            slope, intercept = coeffs
            
            # Calculate p-value for slope significance
            # Calculate residuals and standard error
            y_pred = slope * x + intercept
            residuals = arr - y_pred
            mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
            if mse > 0 and n > 2:
                se_slope = np.sqrt(mse / np.sum((x - x.mean())**2))
                t_stat = abs(slope) / se_slope if se_slope > 0 else 0
                # Approximate p-value (for 2-tailed test)
                # Use normal approximation for large samples, t-distribution for small
                from scipy.stats import t
                p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2)) if n > 2 else 1
            else:
                p_value = 1
            
            features['trend_slope'] = slope
            features['trend_p_value'] = p_value
            features['trend_significant'] = 1 if p_value < 0.05 else 0  # Significant at 5% level
            
            # R-squared
            ss_res = np.sum((arr - y_pred) ** 2)
            ss_tot = np.sum((arr - np.mean(arr)) ** 2)
            features['trend_r2'] = max(0, 1 - (ss_res / (ss_tot + 1e-10)))
            
            # Quadratic trend for acceleration (only if we have enough data)
            if n >= 6:
                quad_coeffs = np.polyfit(x, arr, 2)
                features['trend_acceleration'] = quad_coeffs[0]
            else:
                features['trend_acceleration'] = 0
        else:
            # Not enough data for proper trend analysis
            features['trend_slope'] = 0
            features['trend_r2'] = 0
            features['trend_significant'] = 0
            features['trend_p_value'] = 1
            features['trend_acceleration'] = 0
        
        # Momentum features
        if n >= 3:
            recent_3 = np.mean(arr[-3:])
            earlier_3 = np.mean(arr[-6:-3]) if n >= 6 else np.mean(arr[:min(3, n)])
            
            # Use absolute difference if baseline is too small
            if earlier_3 > 0.1:
                features['momentum_3'] = (recent_3 - earlier_3) / earlier_3
            else:
                features['momentum_3'] = recent_3 - earlier_3  # Absolute change
        else:
            features['momentum_3'] = 0
        
        if n >= 5:
            recent_5 = np.mean(arr[-5:])
            earlier_5 = np.mean(arr[-10:-5]) if n >= 10 else np.mean(arr[:min(5, n)])
            
            # Use absolute difference if baseline is too small
            if earlier_5 > 0.1:
                features['momentum_5'] = (recent_5 - earlier_5) / earlier_5
            else:
                features['momentum_5'] = recent_5 - earlier_5  # Absolute change
        else:
            features['momentum_5'] = 0
        
        if n >= 10:
            recent_10 = np.mean(arr[-10:])
            earlier_10 = np.mean(arr[-20:-10]) if n >= 20 else np.mean(arr[:min(10, n)])
            
            # Use absolute difference if baseline is too small
            if earlier_10 > 0.1:
                features['momentum_10'] = (recent_10 - earlier_10) / earlier_10
            else:
                features['momentum_10'] = recent_10 - earlier_10  # Absolute change
        else:
            features['momentum_10'] = 0
        
        # Exponential moving average ratio (momentum indicator)
        if n >= 5:
            ema_short = pd.Series(arr).ewm(span=3).mean().iloc[-1]
            ema_long = pd.Series(arr).ewm(span=7).mean().iloc[-1]
            features['ema_ratio'] = ema_short / (ema_long + 1e-6)
        else:
            features['ema_ratio'] = 1
        
        # Breakout score (deviation from expected based on trend)
        expected_next = intercept + slope * (n - 1) if n >= PredictionConfig.MIN_GAMES_FOR_TREND else np.mean(arr)
        current = arr[-1] if n > 0 else 0
        std_dev = np.std(arr) if n > 1 else 1
        features['breakout_score'] = (current - expected_next) / (std_dev + 1e-6) if std_dev > 0 else 0
        
        # Mean reversion signal (how far current is from mean)
        features['mean_reversion_signal'] = (current - np.mean(arr)) / (std_dev + 1e-6) if std_dev > 0 else 0
        
        # Streak analysis
        if n >= 2:
            streak = 1
            direction = 1 if arr[-1] > arr[-2] else -1
            for i in range(len(arr) - 2, 0, -1):
                if (arr[i] > arr[i-1]) == (direction == 1):
                    streak += 1
                else:
                    break
            features['streak_length'] = streak
            features['streak_direction'] = direction
        else:
            features['streak_length'] = 1
            features['streak_direction'] = 0
        
        return features
    
    @classmethod
    def compute_consistency_features(cls, values: List[float]) -> Dict[str, float]:
        """Compute comprehensive consistency metrics."""
        features = {}
        arr = np.array(values)
        n = len(arr)
        
        if n < 2:
            return {
                'cv': 0, 'consistency_score': 50, 'reliability_index': 50,
                'floor_pct': 0, 'ceiling_pct': 0, 'iqr_ratio': 0,
                'skewness': 0, 'kurtosis': 0, 'entropy': 0,
                'above_mean_streak': 0, 'volatility_trend': 0
            }
        
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        
        # Coefficient of variation
        cv = std / (mean + 1e-6)
        features['cv'] = cv
        
        # Consistency score (0-100)
        features['consistency_score'] = max(0, min(100, 100 - (cv * 80)))
        
        # Reliability index (% of games within 1 std of mean)
        within_1std = np.mean(np.abs(arr - mean) <= std) * 100
        features['reliability_index'] = within_1std
        
        # Floor and ceiling percentages
        floor_threshold = mean * 0.6
        ceiling_threshold = mean * 1.4
        features['floor_pct'] = np.mean(arr >= floor_threshold) * 100
        features['ceiling_pct'] = np.mean(arr >= ceiling_threshold) * 100
        
        # IQR ratio
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        features['iqr_ratio'] = iqr / (mean + 1e-6)
        
        # Distribution shape
        if n >= 4:
            features['skewness'] = float(pd.Series(arr).skew())
            features['kurtosis'] = float(pd.Series(arr).kurtosis())
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0
        
        # Entropy (measure of unpredictability)
        if n >= 5:
            hist, _ = np.histogram(arr, bins=min(n // 2, 10), density=True)
            hist = hist[hist > 0]
            features['entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        else:
            features['entropy'] = 0
        
        # Above mean streak (how many recent games above mean)
        above_mean_count = 0
        for i in range(min(n, 10)):
            if arr[-(i+1)] >= mean:
                above_mean_count += 1
            else:
                break
        features['above_mean_streak'] = above_mean_count
        
        # Volatility trend (is volatility increasing or decreasing?)
        if n >= 6:
            recent_vol = np.std(arr[-3:], ddof=1) if len(arr[-3:]) > 1 else 0
            older_vol = np.std(arr[:-3], ddof=1) if len(arr[:-3]) > 1 else 0
            features['volatility_trend'] = (recent_vol - older_vol) / (older_vol + 1e-6)
        else:
            features['volatility_trend'] = 0
        
        return features
    
    @classmethod
    def compute_threshold_features(cls, values: List[float], thresholds: List[int]) -> Dict[str, float]:
        """Compute threshold-specific predictive features."""
        features = {}
        arr = np.array(values)
        n = len(arr)
        
        if n == 0:
            return {}
        
        mean = np.mean(arr)
        
        for t in thresholds:
            # Historical hit rate
            hit_rate = np.mean(arr >= t) * 100
            features[f'hr_{t}'] = hit_rate
            
            # Recent hit rate (last 5)
            recent_hr = np.mean(arr[-5:] >= t) * 100 if n >= 5 else hit_rate
            features[f'hr_recent_{t}'] = recent_hr
            
            # Hit rate momentum (recent vs older)
            if n >= 6:
                old_hr = np.mean(arr[:-5] >= t) * 100
                features[f'hr_momentum_{t}'] = recent_hr - old_hr
            else:
                features[f'hr_momentum_{t}'] = 0
            
            # Distance metrics
            features[f'dist_{t}'] = mean - t
            features[f'dist_pct_{t}'] = ((mean - t) / (t + 1e-6)) * 100
            
            # Margin analysis (avg margin when hitting)
            hits = arr[arr >= t]
            if len(hits) > 0:
                features[f'avg_margin_{t}'] = np.mean(hits - t)
                features[f'min_margin_{t}'] = np.min(hits - t)
            else:
                features[f'avg_margin_{t}'] = 0
                features[f'min_margin_{t}'] = 0
            
            # Miss proximity (how close misses were)
            misses = arr[arr < t]
            if len(misses) > 0:
                features[f'miss_proximity_{t}'] = np.mean(t - misses)
            else:
                features[f'miss_proximity_{t}'] = 0
        
        return features
    
    @classmethod
    def compute_game_context_features(cls, df: pd.DataFrame) -> Dict[str, float]:
        """Extract game context and situational features."""
        features = {}
        n = len(df)
        
        # Minutes features
        if 'MIN' in df.columns:
            minutes = df['MIN'].apply(cls.parse_minutes).values
            features['avg_min'] = np.mean(minutes)
            features['min_min'] = np.min(minutes)
            features['max_min'] = np.max(minutes)
            features['min_std'] = np.std(minutes, ddof=1) if n > 1 else 0
            features['recent_min'] = np.mean(minutes[-3:]) if n >= 3 else np.mean(minutes)
            features['min_trend'] = features['recent_min'] - features['avg_min']
            
            # Minutes restriction detection
            features['min_restricted'] = 1 if features['recent_min'] < PredictionConfig.MIN_NORMAL_MINUTES else 0
            
            # Minutes consistency
            features['min_cv'] = features['min_std'] / (features['avg_min'] + 1e-6)
        else:
            for key in ['avg_min', 'min_min', 'max_min', 'min_std', 'recent_min', 'min_trend', 'min_restricted', 'min_cv']:
                features[key] = 0
        
        # Shooting efficiency
        if all(col in df.columns for col in ['FGM', 'FGA']):
            total_fga = df['FGA'].sum()
            features['fg_pct'] = (df['FGM'].sum() / (total_fga + 1e-6)) * 100
            if n >= 5:
                features['recent_fg_pct'] = (df['FGM'].iloc[-5:].sum() / (df['FGA'].iloc[-5:].sum() + 1e-6)) * 100
            else:
                features['recent_fg_pct'] = features['fg_pct']
            features['fg_pct_trend'] = features['recent_fg_pct'] - features['fg_pct']
        
        if all(col in df.columns for col in ['FG3M', 'FG3A']):
            features['fg3_pct'] = (df['FG3M'].sum() / (df['FG3A'].sum() + 1e-6)) * 100
        
        if all(col in df.columns for col in ['FTM', 'FTA']):
            features['ft_pct'] = (df['FTM'].sum() / (df['FTA'].sum() + 1e-6)) * 100
        
        # Usage proxy
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV']):
            if 'MIN' in df.columns:
                minutes = df['MIN'].apply(cls.parse_minutes).values
                total_min = np.sum(minutes)
                if total_min > 0:
                    usage = (df['FGA'].sum() + df['FTA'].sum() * 0.44 + df['TOV'].sum()) / total_min
                    features['usage_rate'] = usage * 48
                else:
                    features['usage_rate'] = 0
            else:
                features['usage_rate'] = 0
        
        # Plus/minus trend (if available)
        if 'PLUS_MINUS' in df.columns:
            pm = df['PLUS_MINUS'].values
            features['avg_plus_minus'] = np.mean(pm)
            features['recent_plus_minus'] = np.mean(pm[-3:]) if n >= 3 else np.mean(pm)
        
        # Home/Away split
        if 'MATCHUP' in df.columns:
            home_mask = df['MATCHUP'].str.contains('vs.', na=False)
            features['home_ratio'] = home_mask.mean()
            
            # Performance by venue (if we have enough games)
            if home_mask.sum() >= 2 and (~home_mask).sum() >= 2:
                features['venue_differential'] = 1  # Flag that we have venue data
            else:
                features['venue_differential'] = 0
        else:
            features['home_ratio'] = 0.5
            features['venue_differential'] = 0
        
        # Rest days estimation (from game dates)
        if 'GAME_DATE' in df.columns:
            try:
                dates = pd.to_datetime(df['GAME_DATE'])
                if len(dates) >= 2:
                    # Days between last two games
                    rest_days = (dates.iloc[-1] - dates.iloc[-2]).days
                    features['last_rest_days'] = rest_days
                    features['is_b2b'] = 1 if rest_days <= 1 else 0
                    
                    # Average rest
                    all_rest = dates.diff().dt.days.dropna()
                    features['avg_rest_days'] = all_rest.mean()
                else:
                    features['last_rest_days'] = 2
                    features['is_b2b'] = 0
                    features['avg_rest_days'] = 2
            except:
                features['last_rest_days'] = 2
                features['is_b2b'] = 0
                features['avg_rest_days'] = 2
        else:
            features['last_rest_days'] = 2
            features['is_b2b'] = 0
            features['avg_rest_days'] = 2
        
        # Blowout game detection
        if 'PLUS_MINUS' in df.columns:
            pm = df['PLUS_MINUS'].abs().values
            features['blowout_ratio'] = np.mean(pm >= PredictionConfig.BLOWOUT_THRESHOLD)
        else:
            features['blowout_ratio'] = 0
        
        return features
    
    @classmethod
    def compute_advanced_features(cls, values: List[float], df: pd.DataFrame, stat_col: str) -> Dict[str, float]:
        """Compute advanced cross-stat and derived features."""
        features = {}
        arr = np.array(values)
        n = len(arr)
        
        # Per-minute stats (if minutes available)
        if 'MIN' in df.columns and stat_col in df.columns:
            minutes = df['MIN'].apply(cls.parse_minutes).values
            stats = df[stat_col].values
            
            per_min = []
            for m, s in zip(minutes, stats):
                if m > 0:
                    per_min.append(s / m * 36)  # Per 36 minutes
            
            if per_min:
                features['per36'] = np.mean(per_min)
                features['per36_recent'] = np.mean(per_min[-3:]) if len(per_min) >= 3 else features['per36']
                features['per36_cv'] = np.std(per_min, ddof=1) / (features['per36'] + 1e-6) if len(per_min) > 1 else 0
        
        # Correlation with other stats
        stat_correlations = {
            'PTS': ['FGA', 'FTA', 'MIN'],
            'REB': ['MIN', 'OREB', 'DREB'],
            'AST': ['MIN', 'TOV'],
            'STL': ['MIN'],
            'BLK': ['MIN'],
            '3PM': ['FG3A', 'MIN'],
            'FG3M': ['FG3A', 'MIN'],
            'PRA': ['MIN', 'FGA']
        }
        
        corr_cols = stat_correlations.get(stat_col, ['MIN'])
        for corr_col in corr_cols:
            if corr_col in df.columns and stat_col in df.columns and n >= 5:
                try:
                    corr = df[stat_col].corr(df[corr_col])
                    features[f'corr_{corr_col.lower()}'] = corr if not np.isnan(corr) else 0
                except:
                    features[f'corr_{corr_col.lower()}'] = 0
        
        # Game-to-game volatility
        if n >= 3:
            diffs = np.diff(arr)
            features['game_volatility'] = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
            features['avg_game_change'] = np.mean(np.abs(diffs))
            features['max_drop'] = np.min(diffs)
            features['max_gain'] = np.max(diffs)
        else:
            features['game_volatility'] = 0
            features['avg_game_change'] = 0
            features['max_drop'] = 0
            features['max_gain'] = 0
        
        # Bayesian prior strength (games played confidence)
        features['data_confidence'] = min(1.0, n / 15)  # Max confidence at 15 games
        
        return features
    
    @classmethod
    def extract_all_features(cls, df: pd.DataFrame, stat_col: str, thresholds: List[int]) -> Dict[str, float]:
        """Extract all 80+ features for prediction."""
        if stat_col not in df.columns:
            return {}
        
        values = df[stat_col].tolist()
        n = len(values)
        
        features = {}
        
        # Core statistics
        features['mean'] = np.mean(values) if values else 0
        features['median'] = np.median(values) if values else 0
        features['std'] = np.std(values, ddof=1) if n > 1 else 0
        features['min'] = np.min(values) if values else 0
        features['max'] = np.max(values) if values else 0
        features['last_game'] = values[-1] if values else 0
        features['second_last'] = values[-2] if n >= 2 else features['last_game']
        features['games_count'] = n
        
        # All feature groups
        features.update(cls.compute_rolling_features(values))
        features.update(cls.compute_trend_features(values))
        features.update(cls.compute_consistency_features(values))
        features.update(cls.compute_threshold_features(values, thresholds))
        features.update(cls.compute_game_context_features(df))
        features.update(cls.compute_advanced_features(values, df, stat_col))
        
        return features


# =============================================================================
# ENSEMBLE ML PREDICTOR
# =============================================================================

class UltimateEnsemblePredictor:
    """
    State-of-the-art ensemble predictor using multiple ML algorithms.
    Dynamically adjusts weights based on stat characteristics and data quality.
    """
    
    def __init__(self):
        self.feature_names = None
        self.scalers = {}
    
    def _create_training_data(self, df: pd.DataFrame, stat_col: str, 
                               thresholds: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Create training data using sliding window approach with temporal split."""
        if len(df) < PredictionConfig.MIN_GAMES_FOR_ML:  # Need at least 6 games
            return None, None, []
        
        X_list = []
        y_list = []
        feature_names = None
        
        # Look-ahead bias fix: stop one game before the end to predict the last game
        for end_idx in range(5, len(df) - 1):  # Stop before last game
            window_df = df.iloc[:end_idx].copy()
            features = UltimateFeatureEngineer.extract_all_features(window_df, stat_col, thresholds)
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            X_list.append([features.get(k, 0) for k in feature_names])
            y_list.append(df[stat_col].iloc[end_idx])  # Predict the next game (end_idx)
        
        if len(X_list) < 4:
            return None, None, []
        
        return np.array(X_list), np.array(y_list), feature_names
    
    def _get_models(self, stat_type: str) -> List[Tuple[str, Any, float]]:
        """Get models with weights adjusted for stat type."""
        models = []
        char = PredictionConfig.STAT_CHARACTERISTICS.get(stat_type, {})
        variance_type = char.get('variance', 'medium')
        
        # Adjust weights based on variance
        weight_adj = 1.0
        if variance_type == 'high':
            weight_adj = 0.9  # Slightly lower ML weight for high variance stats
        
        # XGBoost (primary)
        if XGBOOST_AVAILABLE:
            xgb = XGBRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
            models.append(('xgb', xgb, 0.35 * weight_adj))
        
        # LightGBM (fast and accurate)
        if LIGHTGBM_AVAILABLE:
            lgbm = LGBMRegressor(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            )
            models.append(('lgbm', lgbm, 0.25 * weight_adj))
        
        # Sklearn models
        if SKLEARN_AVAILABLE:
            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            models.append(('gb', gb, 0.20 * weight_adj))
            
            # Bayesian Ridge (provides uncertainty)
            br = BayesianRidge()
            models.append(('bayesian', br, 0.10 * weight_adj))
            
            # Ridge for stability
            ridge = Ridge(alpha=1.0)
            models.append(('ridge', ridge, 0.10 * weight_adj))
        
        return models
    
    def _train_and_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_current: np.ndarray, stat_type: str, cat: str) -> Dict:
        """Train models and generate predictions."""
        models = self._get_models(stat_type)
        
        if not models:
            return {'predictions': [], 'weights': [], 'uncertainties': []}
        
        predictions = []
        weights = []
        uncertainties = []
        
        # Scale features for linear models
        scaler = RobustScaler() if SKLEARN_AVAILABLE else None
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_current_scaled = scaler.transform(X_current)
            self.scalers[cat] = scaler
        else:
            X_train_scaled = X_train
            X_current_scaled = X_current
        
        for name, model, weight in models:
            try:
                # Use scaled data for linear models
                if name in ['ridge', 'bayesian'] and scaler:
                    model.fit(X_train_scaled, y_train)
                    pred = model.predict(X_current_scaled)[0]
                    
                    # Get uncertainty from Bayesian Ridge
                    if name == 'bayesian' and hasattr(model, 'predict'):
                        try:
                            _, std = model.predict(X_current_scaled, return_std=True)
                            uncertainties.append(std[0])
                        except:
                            uncertainties.append(np.std(y_train))
                    else:
                        uncertainties.append(np.std(y_train) * 0.5)
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_current)[0]
                    uncertainties.append(np.std(y_train) * 0.5)
                
                predictions.append(pred)
                weights.append(weight)
                
            except Exception as e:
                continue
        
        return {
            'predictions': predictions,
            'weights': weights,
            'uncertainties': uncertainties
        }
    
    def predict(self, df: pd.DataFrame, stat_col: str, thresholds: List[int], cat: str) -> Dict:
        """Generate ensemble prediction with uncertainty estimates."""
        # Extract current features
        features = UltimateFeatureEngineer.extract_all_features(df, stat_col, thresholds)
        
        if not features:
            return self._fallback_prediction(df, stat_col)
        
        # Create training data
        X_train, y_train, feature_names = self._create_training_data(df, stat_col, thresholds)
        self.feature_names = feature_names
        
        # Current feature vector
        X_current = np.array([[features.get(k, 0) for k in feature_names]]) if feature_names else None
        
        # Statistical baseline
        values = df[stat_col].tolist()
        stat_prediction = self._exponential_weighted_average(values)
        
        # ML predictions
        ml_result = None
        if X_train is not None and X_current is not None and len(X_train) >= 4:
            stat_type = cat if cat in PredictionConfig.STAT_CHARACTERISTICS else 'PTS'
            ml_result = self._train_and_predict(X_train, y_train, X_current, stat_type, cat)
        
        # Combine predictions
        if ml_result and ml_result['predictions']:
            preds = np.array(ml_result['predictions'])
            weights = np.array(ml_result['weights'])
            weights = weights / weights.sum()
            
            ml_prediction = np.average(preds, weights=weights)
            ml_uncertainty = np.sqrt(np.average((preds - ml_prediction)**2, weights=weights))
            
            # Adaptive blending based on data quality
            data_confidence = features.get('data_confidence', 0.5)
            ml_weight = 0.6 + 0.2 * data_confidence  # 60-80% ML weight
            stat_weight = 1 - ml_weight
            
            final_prediction = ml_weight * ml_prediction + stat_weight * stat_prediction
        else:
            ml_prediction = None
            ml_uncertainty = None
            final_prediction = stat_prediction
        
        return {
            'ml_prediction': round(ml_prediction, 2) if ml_prediction else None,
            'stat_prediction': round(stat_prediction, 2),
            'final_prediction': round(final_prediction, 2),
            'uncertainty': round(ml_uncertainty, 2) if ml_uncertainty else None,
            'features': features,
            'model_used': 'ensemble' if ml_prediction else 'statistical',
            'feature_names': feature_names
        }
    
    def _exponential_weighted_average(self, values: List[float]) -> float:
        """Exponential weighted average with decay."""
        if not values:
            return 0.0
        n = len(values)
        if n == 1:
            return float(values[0])
        
        # Exponential decay weights
        alpha = 0.15
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()
        
        return float(np.average(values, weights=weights))
    
    def _fallback_prediction(self, df: pd.DataFrame, stat_col: str) -> Dict:
        """Fallback when ML not possible."""
        values = df[stat_col].tolist() if stat_col in df.columns else []
        stat_pred = self._exponential_weighted_average(values)
        
        return {
            'ml_prediction': None,
            'stat_prediction': round(stat_pred, 2),
            'final_prediction': round(stat_pred, 2),
            'uncertainty': None,
            'features': {},
            'model_used': 'fallback',
            'feature_names': []
        }


# =============================================================================
# ADVANCED PROBABILITY ENGINE
# =============================================================================

class BayesianProbabilityEngine:
    """
    Advanced probability calculations using Bayesian methods,
    multiple distributions, and Monte Carlo simulation.
    """
    
    @staticmethod
    def negative_binomial_prob(mean: float, variance: float, threshold: int) -> float:
        """P(X >= threshold) using Negative Binomial (better for overdispersed data)."""
        if mean <= 0 or variance <= mean:
            return BayesianProbabilityEngine.poisson_prob(mean, threshold)
        
        # NB parameters from mean and variance
        r = mean ** 2 / (variance - mean)
        p = mean / variance
        
        try:
            prob = 1 - nbinom.cdf(threshold - 1, r, p)
            return round(prob * 100, 1)
        except:
            return BayesianProbabilityEngine.poisson_prob(mean, threshold)
    
    @staticmethod
    def poisson_prob(expected: float, threshold: int) -> float:
        """P(X >= threshold) using Poisson distribution."""
        if expected <= 0:
            return 0.0
        prob = 1 - poisson.cdf(threshold - 1, expected)
        return round(prob * 100, 1)
    
    @staticmethod
    def normal_prob(expected: float, std: float, threshold: int) -> float:
        """P(X >= threshold) using Normal distribution."""
        if std <= 0:
            return 100.0 if expected >= threshold else 0.0
        z = (threshold - expected) / std
        prob = 1 - norm.cdf(z)
        return round(prob * 100, 1)
    
    @staticmethod
    def empirical_prob(values: List[float], threshold: int) -> float:
        """Empirical probability from historical data."""
        if not values:
            return 50.0
        return round(np.mean([1 if v >= threshold else 0 for v in values]) * 100, 1)
    
    @staticmethod
    def bayesian_prob(values: List[float], threshold: int) -> float:
        """
        Bayesian probability using Beta-Binomial model.
        Incorporates prior uncertainty when data is limited.
        """
        if not values:
            return 50.0
        
        n = len(values)
        successes = sum(1 for v in values if v >= threshold)
        
        # Prior: Beta(1, 1) = uniform
        alpha_prior = 1
        beta_prior = 1
        
        # Posterior: Beta(alpha + successes, beta + failures)
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (n - successes)
        
        # Expected value of posterior
        expected_prob = alpha_post / (alpha_post + beta_post)
        
        return round(expected_prob * 100, 1)
    
    @staticmethod
    def monte_carlo_prob(expected: float, std: float, threshold: int, 
                         n_sims: int = None, seed: Optional[int] = 42) -> Tuple[float, float, float]:
        """
        Monte Carlo simulation for probability with confidence interval.
        seed=42: Reproducible (default for production)
        seed=None: Truly random (use for sensitivity testing)
        """
        n_sims = n_sims or PredictionConfig.MC_SIMULATIONS
        
        # Use Generator for better randomness control
        rng = np.random.default_rng(seed)
        
        if std <= 0:
            prob = 100.0 if expected >= threshold else 0.0
            return prob, prob, prob
        
        # Simulate outcomes
        simulations = rng.normal(expected, std, n_sims)
        simulations = np.maximum(0, simulations)  # Floor at 0
        
        # Calculate probability
        hits = np.mean(simulations >= threshold)
        
        # Bootstrap confidence interval
        bootstrap_probs = []
        for _ in range(100):
            sample = rng.choice(simulations, size=n_sims, replace=True)
            bootstrap_probs.append(np.mean(sample >= threshold))
        
        lower_ci = np.percentile(bootstrap_probs, 2.5) * 100
        upper_ci = np.percentile(bootstrap_probs, 97.5) * 100
        
        return round(hits * 100, 1), round(lower_ci, 1), round(upper_ci, 1)
    
    @staticmethod
    def select_best_distribution(values: np.ndarray) -> str:
        """Select distribution based on goodness-of-fit tests."""
        if len(values) < 3:
            return 'empirical'  # Not enough data for fit testing
        
        try:
            # Shapiro-Wilk test for normality
            _, p_normal = shapiro(values) if len(values) <= 5000 else (0, 0)
            
            # Variance-to-mean ratio for Poisson fit (requires non-negative values)
            if np.all(values >= 0):
                vmr = np.var(values) / (np.mean(values) + 1e-6)
                
                if p_normal > 0.05:
                    return 'normal'  # Good normal fit
                elif 0.8 < vmr < 1.2:
                    return 'poisson'  # Good Poisson fit
                elif vmr > 1.2:
                    return 'neg_binomial'  # Overdispersed - negative binomial
                else:
                    return 'empirical'  # Use historical when distribution unclear
            else:
                # Data has negative values, probably differences or ratios
                return 'normal' if p_normal > 0.05 else 'empirical'
        except:
            # Fallback if tests fail
            return 'empirical'

    @classmethod
    def combined_probability(cls, expected: float, std: float, values: List[float], 
                            threshold: int, variance: float = None) -> Dict[str, Any]:
        """
        Calculate combined probability using multiple methods with distribution fit testing.
        Adaptive weighting based on stat characteristics and data fit.
        """
        n = len(values)
        variance = variance if variance else (std ** 2)
        
        # Select best-fitting distribution
        best_dist = cls.select_best_distribution(np.array(values))
        
        # Calculate all probabilities
        probs = {
            'poisson': cls.poisson_prob(expected, threshold),
            'normal': cls.normal_prob(expected, std, threshold),
            'empirical': cls.empirical_prob(values, threshold),
            'bayesian': cls.bayesian_prob(values, threshold),
            'neg_binomial': cls.negative_binomial_prob(expected, variance, threshold),
            'best_distribution': 'unknown'
        }
        
        # Assign best distribution based on fit
        probs['best_distribution'] = best_dist
        if best_dist in probs:
            probs['best_dist_prob'] = probs[best_dist]
        
        # Monte Carlo with confidence interval
        mc_prob, mc_lower, mc_upper = cls.monte_carlo_prob(expected, std, threshold, seed=None)
        probs['monte_carlo'] = mc_prob
        probs['mc_ci_lower'] = mc_lower
        probs['mc_ci_upper'] = mc_upper
        
        # Adaptive weighting based on data quantity and distribution fit
        if n >= 10:
            # More data = trust empirical and Monte Carlo more
            weights = {
                'empirical': 0.30,
                'monte_carlo': 0.25,
                'bayesian': 0.15,
                'normal': 0.15,
                'poisson': 0.10,
                'neg_binomial': 0.05
            }
        elif n >= 5:
            # Medium data = balanced
            weights = {
                'bayesian': 0.25,
                'monte_carlo': 0.25,
                'empirical': 0.20,
                'normal': 0.15,
                'poisson': 0.10,
                'neg_binomial': 0.05
            }
        else:
            # Limited data = rely on Bayesian and parametric
            weights = {
                'bayesian': 0.35,
                'normal': 0.25,
                'poisson': 0.20,
                'monte_carlo': 0.15,
                'empirical': 0.05,
                'neg_binomial': 0.00
            }
        
        # Weighted combination
        combined = sum(probs[k] * weights[k] for k in weights.keys())
        
        # Confidence based on Monte Carlo CI width
        ci_width = mc_upper - mc_lower
        confidence_penalty = min(20, ci_width / 5)
        
        probs['combined'] = round(combined, 1)
        probs['confidence_adjustment'] = round(-confidence_penalty, 1)
        
        return probs


# =============================================================================
# CONFIDENCE CALCULATOR
# =============================================================================

def calculate_ultimate_confidence(features: Dict[str, float], pred_result: Dict, 
                                  prob_data: Dict, stat_type: str) -> Dict:
    """
    Calculate comprehensive confidence score (0-100) with breakdown.
    Considers multiple factors for robust confidence estimation.
    Returns dictionary with 'confidence' and 'breakdown' keys.
    """
    factors = {}
    
    # 1. Data sufficiency (0-25 points)
    games = features.get('games_count', 0)
    factors['data_sufficiency'] = min(25, max(0, (games / 15) * 25))
    
    # 2. Consistency (0-30 points)
    cv = features.get('cv', 1)
    factors['consistency'] = max(0, 30 * (1 - min(cv / 0.5, 1)))
    
    # 3. Trend clarity (0-15 points)
    r2 = features.get('trend_r2', 0)
    factors['trend_clarity'] = r2 * 15
    
    # 4. Model agreement (0-20 points)
    ml_pred = pred_result.get('ml_prediction')
    stat_pred = pred_result.get('stat_prediction')
    if ml_pred is not None and stat_pred is not None:
        agreement = 1 - abs(ml_pred - stat_pred) / (stat_pred + 1e-6)
        factors['model_agreement'] = max(0, agreement * 20)
    else:
        factors['model_agreement'] = 0
    
    # 5. Probability precision (0-10 points)
    ci_width = prob_data.get('mc_ci_upper', 100) - prob_data.get('mc_ci_lower', 0)
    factors['probability_precision'] = max(0, 10 * (1 - min(ci_width / 50, 1)))
    
    # 6. Trend significance (0-10 points)
    trend_sig = features.get('trend_significant', 0)
    factors['trend_significance'] = trend_sig * 10
    
    # Calculate total
    total = sum(factors.values())
    
    # Clamp to reasonable bounds
    confidence = int(np.clip(total, 20, 95))
    
    return {
        'confidence': confidence,
        'breakdown': factors
    }


def calculate_ultimate_confidence_legacy(features: Dict[str, float], pred_result: Dict, 
                                  prob_data: Dict, stat_type: str) -> int:
    """
    Legacy confidence calculator for backward compatibility.
    """
    confidence = 50  # Base
    
    # 1. Data quantity (up to +15)
    games = features.get('games_count', 0)
    if games >= 15:
        confidence += 15
    elif games >= 10:
        confidence += 10
    elif games >= 7:
        confidence += 5
    elif games < 5:
        confidence -= 10
    
    # 2. Consistency (up to +20)
    cv = features.get('cv', 1)
    if cv < 0.15:
        confidence += 20
    elif cv < 0.25:
        confidence += 15
    elif cv < 0.35:
        confidence += 10
    elif cv < 0.50:
        confidence += 5
    elif cv > 0.70:
        confidence -= 10
    
    # 3. Trend stability (up to +10)
    trend_r2 = features.get('trend_r2', 0)
    if trend_r2 > 0.7:
        confidence += 10
    elif trend_r2 > 0.4:
        confidence += 5
    
    # 4. Model quality (up to +10)
    if pred_result.get('model_used') == 'ensemble':
        confidence += 10
        
        # Uncertainty penalty
        uncertainty = pred_result.get('uncertainty', 0)
        if uncertainty:
            mean = features.get('mean', 1)
            uncertainty_ratio = uncertainty / (mean + 1e-6)
            if uncertainty_ratio > 0.3:
                confidence -= 10
            elif uncertainty_ratio > 0.2:
                confidence -= 5
    
    # 5. Floor reliability (up to +10)
    floor_pct = features.get('floor_pct', 0)
    if floor_pct > 85:
        confidence += 10
    elif floor_pct > 70:
        confidence += 5
    
    # 6. Minutes stability (up to +5)
    if features.get('min_restricted', 0) == 0 and features.get('min_cv', 1) < 0.2:
        confidence += 5
    elif features.get('min_restricted', 0) == 1:
        confidence -= 10
    
    # 7. Monte Carlo CI width penalty
    ci_adjustment = prob_data.get('confidence_adjustment', 0)
    confidence += int(ci_adjustment)
    
    # 8. High variance stat penalty
    char = PredictionConfig.STAT_CHARACTERISTICS.get(stat_type, {})
    if char.get('variance') == 'high':
        confidence -= 5
    
    return max(15, min(95, confidence))


# =============================================================================
# INSIGHT GENERATOR
# =============================================================================

def generate_insights(features: Dict[str, float], prob_data: Dict, 
                      expected: float, threshold_hit_rates: Dict,
                      venue_splits: Optional[Dict] = None,
                      is_home_game: Optional[bool] = None,
                      context_insights: Optional[Dict] = None) -> List[str]:
    """Generate actionable insights based on analysis."""
    insights = []
    
    # === GAME CONTEXT INSIGHTS (Priority) ===
    if context_insights:
        # B2B Analysis
        b2b = context_insights.get('b2b', {})
        if b2b.get('impact') == 'negative':
            insights.append(f"💤 {b2b.get('reason', 'Back-to-back')}")
        
        # Rest Days
        rest = context_insights.get('rest', {})
        rest_reason = rest.get('reason', '')
        if 'Well rested' in rest_reason:
            insights.append(f"⚡ {rest_reason}")
        
        # Pace
        pace = context_insights.get('pace', {})
        pace_reason = pace.get('reason', '')
        if 'Fast' in pace_reason:
            insights.append(f"🚀 {pace_reason}")
        elif 'Slow' in pace_reason:
            insights.append(f"🐢 {pace_reason}")
        
        # Clutch
        clutch = context_insights.get('clutch', {})
        clutch_reason = clutch.get('reason', '')
        if 'performer' in clutch_reason.lower():
            insights.append(f"🔥 Clutch performer")
        elif 'concern' in clutch_reason.lower():
            insights.append(f"😰 {clutch_reason}")
        
        # Minutes
        minutes = context_insights.get('minutes', {})
        minutes_reason = minutes.get('reason', '')
        if 'boost' in minutes_reason.lower():
            insights.append(f"⏱️ {minutes_reason}")
        elif 'concern' in minutes_reason.lower():
            insights.append(f"⚠️ {minutes_reason}")
        
        # Opponent Matchup
        opp = context_insights.get('opponent', {})
        opp_reason = opp.get('reason', '')
        if 'Favorable' in opp_reason:
            insights.append(f"🎯 {opp_reason} matchup")
        elif 'Tough' in opp_reason:
            insights.append(f"🛡️ {opp_reason} matchup")
    
    # === HOME/AWAY INSIGHTS (Priority) ===
    if venue_splits and is_home_game is not None:
        home_avg = venue_splits.get('home_avg', 0)
        away_avg = venue_splits.get('away_avg', 0)
        diff = venue_splits.get('differential', 0)
        significant = venue_splits.get('significant', False)
        venue_impact = venue_splits.get('venue_impact', 'neutral')
        
        if significant and abs(diff) > 0:
            if is_home_game:
                if diff > 0:
                    # Player is better at home and playing home
                    insights.append(f"🏠 HOME BOOST: +{diff:.1f} avg at home ({venue_impact})")
                else:
                    # Player is worse at home but playing home
                    insights.append(f"🏠 Caution: {diff:.1f} worse at home vs road")
            else:
                if diff < 0:
                    # Player is better on road and playing away
                    insights.append(f"✈️ ROAD WARRIOR: +{abs(diff):.1f} better on road ({venue_impact})")
                else:
                    # Player is worse on road and playing away
                    insights.append(f"✈️ Road concern: -{diff:.1f} worse away vs home")
        elif venue_splits.get('home_games', 0) >= 2 and venue_splits.get('away_games', 0) >= 2:
            # Have venue data but not significant
            if is_home_game:
                insights.append(f"🏠 Playing HOME (avg: {home_avg:.1f})")
            else:
                insights.append(f"✈️ Playing AWAY (avg: {away_avg:.1f})")
    
    # Trend insights
    momentum = features.get('momentum_5', 0)
    if momentum > 0.15:
        insights.append("📈 Strong upward trend - player heating up!")
    elif momentum > 0.08:
        insights.append("📈 Trending upward recently")
    elif momentum < -0.15:
        insights.append("📉 Significant downward trend - caution advised")
    elif momentum < -0.08:
        insights.append("📉 Trending downward recently")
    
    # Hot/cold streaks
    streak_len = features.get('streak_length', 0)
    streak_dir = features.get('streak_direction', 0)
    if streak_len >= 4:
        if streak_dir > 0:
            insights.append(f"🔥 On a {streak_len}-game hot streak!")
        else:
            insights.append(f"❄️ On a {streak_len}-game cold streak")
    
    # Consistency
    consistency = features.get('consistency_score', 50)
    if consistency > 75:
        insights.append("✅ Highly consistent performer - reliable floor")
    elif consistency < 35:
        insights.append("⚠️ High variance player - boom or bust potential")
    
    # Minutes situation
    if features.get('min_restricted', 0) == 1:
        insights.append("⏱️ Playing reduced minutes recently - monitor closely")
    elif features.get('min_trend', 0) > 3:
        insights.append("⬆️ Minutes trending up - could boost production")
    elif features.get('min_trend', 0) < -3:
        insights.append("⬇️ Minutes trending down - could limit ceiling")
    
    # Back-to-back
    if features.get('is_b2b', 0) == 1:
        insights.append("📅 Back-to-back game - slight fatigue factor")
    
    # Mean reversion
    mr_signal = features.get('mean_reversion_signal', 0)
    if mr_signal > 0.15:
        insights.append("🎯 Due for positive regression to mean")
    elif mr_signal < -0.15:
        insights.append("⚖️ Running hot - regression risk")
    
    # Breakout potential
    if features.get('breakout_score', 0) > 0.3:
        insights.append("💥 Recent breakout performance - elevated ceiling")
    
    # Data confidence
    if features.get('games_count', 0) < 5:
        insights.append("📊 Limited data - predictions less certain")
    
    return insights[:6]  # Limit to top 6 insights


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

# ============================================================================
# ENHANCED GAME CONTEXT ANALYZERS
# ============================================================================

class GameContextAnalyzer:
    """Analyzes game context factors: B2B, rest days, opponent strength"""
    
    @staticmethod
    def is_back_to_back(df: pd.DataFrame) -> Tuple[bool, int]:
        """Check if the upcoming game is a back-to-back. Returns (is_b2b, days_since_last_game)"""
        if df is None or len(df) < 2:
            return False, 999
        
        df_sorted = df.sort_values('GAME_DATE', ascending=False)
        if 'GAME_DATE' not in df_sorted.columns:
            return False, 999
        
        try:
            df_sorted['GAME_DATE'] = pd.to_datetime(df_sorted['GAME_DATE'])
            last_game = df_sorted.iloc[0]['GAME_DATE']
            prev_game = df_sorted.iloc[1]['GAME_DATE']
            days_diff = (last_game - prev_game).days
            
            is_b2b = days_diff == 1
            return is_b2b, days_diff
        except:
            return False, 999
    
    @staticmethod
    def calculate_b2b_fatigue(df: pd.DataFrame, col: str) -> Tuple[float, str]:
        """Calculate fatigue factor for back-to-back games"""
        is_b2b, days_diff = GameContextAnalyzer.is_back_to_back(df)
        
        if not is_b2b:
            return 1.0, "Normal rest"
        
        # Get historical B2B performance if available
        if df is not None and len(df) >= 2 and 'GAME_DATE' in df.columns:
            try:
                df_sorted = df.sort_values('GAME_DATE', ascending=False).copy()
                df_sorted['GAME_DATE'] = pd.to_datetime(df_sorted['GAME_DATE'])
                
                # Find B2B games
                df_sorted['days_diff'] = df_sorted['GAME_DATE'].diff(-1).days
                b2b_games = df_sorted[df_sorted['days_diff'] == 1]
                
                if len(b2b_games) >= 2 and col in b2b_games.columns:
                    b2b_avg = b2b_games[col].mean()
                    non_b2b_games = df_sorted[df_sorted['days_diff'] != 1]
                    non_b2b_avg = non_b2b_games[col].mean() if len(non_b2b_games) > 0 else b2b_avg
                    
                    if non_b2b_avg > 0:
                        fatigue_factor = b2b_avg / non_b2b_avg
                        return fatigue_factor, f"B2B ({len(b2b_games)} games)"
            except:
                pass
        
        # Default fatigue factor for B2B (typically 2-5% worse)
        return 0.96, "B2B (default)"
    
    @staticmethod
    def get_rest_days(df: pd.DataFrame) -> int:
        """Get number of days since last game"""
        _, days_diff = GameContextAnalyzer.is_back_to_back(df)
        return days_diff
    
    @staticmethod
    def calculate_rest_advantage(df: pd.DataFrame, col: str) -> Tuple[float, str]:
        """Calculate rest day advantage/disadvantage"""
        rest_days = GameContextAnalyzer.get_rest_days(df)
        
        if rest_days >= 999:
            return 1.0, "Unknown rest"
        
        if rest_days == 1:
            return GameContextAnalyzer.calculate_b2b_fatigue(df, col)
        elif rest_days >= 4:
            return 1.05, f"Well rested ({rest_days} days)"
        elif rest_days >= 2:
            return 1.02, f"Normal rest ({rest_days} days)"
        else:
            return 1.0, f"Rest: {rest_days} days"
    
    @staticmethod
    def get_opponent_rank_adjustment(opponent_team_id: int, col: str, 
                                     full_season_df: pd.DataFrame) -> Tuple[float, str]:
        """Calculate adjustment based on opponent's defensive ranking"""
        if full_season_df is None or opponent_team_id is None:
            return 1.0, "No opponent data"
        
        try:
            # Calculate opponent's average allowed for this stat
            # This is a simplified version - in production you'd use defensive ratings
            if 'TEAM_ID' not in full_season_df.columns:
                return 1.0, "No team data"
            
            opponent_games = full_season_df[full_season_df['TEAM_ID'] == opponent_team_id]
            if len(opponent_games) < 5:
                return 1.0, "Limited opponent data"
            
            # Calculate average stat allowed (simplified)
            stat_col = 'OPP_' + col if 'OPP_' + col in opponent_games.columns else col
            if stat_col not in opponent_games.columns:
                return 1.0, f"No {col} data"
            
            opp_avg = opponent_games[stat_col].mean()
            league_avg = full_season_df[stat_col].mean() if stat_col in full_season_df.columns else opp_avg
            
            if league_avg > 0:
                # Lower is better for defense
                ratio = league_avg / opp_avg
                if ratio > 1.1:
                    return 1.08, "Favorable matchup"
                elif ratio < 0.9:
                    return 0.92, "Tough matchup"
            
            return 1.0, "Average matchup"
        except:
            return 1.0, "Calculation error"
    
    @staticmethod
    def calculate_pace_adjustment(df: pd.DataFrame, col: str, 
                                  team_pace: Optional[float] = None,
                                  league_avg_pace: float = 100.0) -> Tuple[float, str]:
        """
        Calculate pace-adjusted stat projection.
        Normalizes stats per 100 possessions for fair comparison.
        """
        if df is None or len(df) < 3:
            return 1.0, "Insufficient pace data"
        
        try:
            # Get team's average pace if not provided
            if team_pace is None:
                if 'PACE' in df.columns:
                    team_pace = df['PACE'].mean()
                elif 'possessions' in df.columns:
                    # Calculate pace from possessions if available
                    team_pace = df['possessions'].mean() / df['MIN'].mean() * 48 if 'MIN' in df.columns else None
            
            if team_pace is None:
                team_pace = league_avg_pace
            
            # Calculate pace factor (how team's pace compares to league average)
            pace_factor = league_avg_pace / team_pace if team_pace > 0 else 1.0
            
            # Adjust: faster pace = more opportunities = higher projected stats
            if pace_factor > 1.05:
                return round(pace_factor, 3), f"Fast pace ({pace_factor:.2f}x)"
            elif pace_factor < 0.95:
                return round(pace_factor, 3), f"Slow pace ({pace_factor:.2f}x)"
            else:
                return 1.0, "Neutral pace"
        except:
            return 1.0, "Pace calc error"
    
    @staticmethod
    def calculate_clutch_performance(df: pd.DataFrame, col: str) -> Tuple[float, str]:
        """
        Analyze player's performance in clutch time (last 5 mins within 5 points).
        Returns adjustment factor based on historical clutch performance.
        """
        if df is None or len(df) < 5:
            return 1.0, "No clutch data"
        
        # Check if we have clutch data (would need special NBA API for real clutch stats)
        # For now, use a simple proxy: 4th quarter performance
        try:
            if 'Q4_' + col in df.columns:
                q4_col = 'Q4_' + col
            elif 'fourth_quarter' in df.columns:
                q4_col = 'fourth_quarter'
            else:
                # Use last 2 games as proxy for clutch readiness
                recent = df.head(2)
                if col in recent.columns:
                    recent_avg = recent[col].mean()
                    season_avg = df[col].mean()
                    if season_avg > 0 and recent_avg > 0:
                        ratio = recent_avg / season_avg
                        if ratio > 1.1:
                            return 1.05, "Heating up (clutch form)"
                        elif ratio < 0.9:
                            return 0.95, "Cold (clutch concern)"
                return 1.0, "Average clutch"
            
            if q4_col in df.columns:
                clutch_avg = df[q4_col].mean()
                season_avg = df[col].mean()
                if season_avg > 0 and clutch_avg > 0:
                    ratio = clutch_avg / season_avg
                    if ratio > 1.1:
                        return 1.08, "Clutch performer (+8%)"
                    elif ratio < 0.9:
                        return 0.92, "Clutch struggle (-8%)"
            
            return 1.0, "Average clutch"
        except:
            return 1.0, "Clutch calc error"
    
    @staticmethod
    def calculate_minutes_weight(df: pd.DataFrame, col: str) -> Tuple[float, str]:
        """
        Weight recent games by minutes played - more minutes = more reliable data.
        """
        if df is None or len(df) < 3:
            return 1.0, "Insufficient minutes data"
        
        try:
            if 'MIN' not in df.columns:
                return 1.0, "No minutes data"
            
            # Calculate weighted average (recent games weighted by minutes)
            weights = df['MIN'].values
            if sum(weights) == 0:
                return 1.0, "No minutes played"
            
            weighted_avg = sum(w * v for w, v in zip(weights, df[col].values)) / sum(weights)
            simple_avg = df[col].mean()
            
            if simple_avg > 0:
                ratio = weighted_avg / simple_avg
                if ratio > 1.02:
                    return round(ratio, 3), "High minutes boost"
                elif ratio < 0.98:
                    return round(ratio, 3), "Low minutes concern"
            
            return 1.0, "Normal minutes"
        except:
            return 1.0, "Minutes calc error"
    
    @staticmethod
    def generate_context_insights(df: pd.DataFrame, col: str, 
                                opponent_team_id: Optional[int],
                                full_season_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive context insights"""
        insights = {}
        
        # B2B Analysis
        fatigue_factor, fatigue_reason = GameContextAnalyzer.calculate_b2b_fatigue(df, col)
        insights['b2b'] = {
            'factor': fatigue_factor,
            'reason': fatigue_reason,
            'impact': 'negative' if fatigue_factor < 0.98 else 'neutral'
        }
        
        # Rest Days
        rest_factor, rest_reason = GameContextAnalyzer.calculate_rest_advantage(df, col)
        insights['rest'] = {
            'factor': rest_factor,
            'reason': rest_reason,
            'days': GameContextAnalyzer.get_rest_days(df)
        }
        
        # Pace Adjustment
        pace_factor, pace_reason = GameContextAnalyzer.calculate_pace_adjustment(df, col)
        insights['pace'] = {
            'factor': pace_factor,
            'reason': pace_reason
        }
        
        # Clutch Performance
        clutch_factor, clutch_reason = GameContextAnalyzer.calculate_clutch_performance(df, col)
        insights['clutch'] = {
            'factor': clutch_factor,
            'reason': clutch_reason
        }
        
        # Minutes Weight
        minutes_factor, minutes_reason = GameContextAnalyzer.calculate_minutes_weight(df, col)
        insights['minutes'] = {
            'factor': minutes_factor,
            'reason': minutes_reason
        }
        
        # Opponent Analysis
        if opponent_team_id and full_season_df is not None:
            opp_factor, opp_reason = GameContextAnalyzer.get_opponent_rank_adjustment(
                opponent_team_id, col, full_season_df
            )
            insights['opponent'] = {
                'factor': opp_factor,
                'reason': opp_reason,
                'team_id': opponent_team_id
            }
        else:
            insights['opponent'] = {
                'factor': 1.0,
                'reason': 'No opponent data',
                'team_id': None
            }
        
        return insights


class InjuryAnalyzer:
    """Analyze injury impact on predictions"""
    
    # Common starters and their impacts (simplified - in production use real injury API)
    STAR_PLAYERS = {
        1629630: {"name": "Ja Morant", "impact": 0.15},   # Grizzlies
        203999: {"name": "Kevin Durant", "impact": 0.12},  # Suns
        2544: {"name": "LeBron James", "impact": 0.15},   # Lakers
        201939: {"name": "Stephen Curry", "impact": 0.15}, # Warriors
        201942: {"name": "Giannis", "impact": 0.15},       # Bucks
    }
    
    @staticmethod
    def check_injuries(player_id: int, team_id: int) -> Tuple[bool, str, float]:
        """Check if key player is out - returns (is_out, player_name, impact_factor)"""
        # This would connect to real injury API in production
        # For now, return placeholder - user can integrate injury API
        return False, "", 1.0


class PredictionValidator:
    """Track and validate predictions against actual results"""
    
    def __init__(self):
        self.predictions_history = []
    
    def record_prediction(self, prediction: Dict, actual: Optional[Dict] = None):
        """Record a prediction for later validation"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual
        }
        self.predictions_history.append(record)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate validation metrics from recorded predictions"""
        if not self.predictions_history:
            return {'error': 'No predictions recorded'}
        
        # Filter to completed predictions
        completed = [p for p in self.predictions_history if p.get('actual') is not None]
        if not completed:
            return {'error': 'No completed predictions'}
        
        correct = 0
        total = len(completed)
        errors = []
        
        for record in completed:
            pred = record['prediction']
            actual = record['actual']
            
            cat = pred.get('category')
            threshold = pred.get('threshold')
            predicted_over = pred.get('over_probability', 0.5) > 0.5
            actual_over = actual.get(cat, 0) > threshold
            
            if predicted_over == actual_over:
                correct += 1
            else:
                diff = abs(pred.get('expected', 0) - actual.get(cat, 0))
                errors.append(diff)
        
        accuracy = correct / total if total > 0 else 0
        mae = sum(errors) / len(errors) if errors else 0
        
        return {
            'total_predictions': total,
            'accuracy': accuracy,
            'mean_absolute_error': mae,
            'sample_size': total
        }
    
    def get_top_features(self) -> List[Dict]:
        """Analyze which features correlate with correct predictions"""
        # Placeholder for feature importance analysis
        return []


# Global validator instance
_prediction_validator = PredictionValidator()


def generate_predictions(last_n_df: pd.DataFrame, milestones: Dict[str, List[int]], 
                         matchup_multipliers: Optional[Dict[str, float]] = None,
                         game_context: Optional[Dict[str, Any]] = None,
                         full_season_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Generate ultimate ML-powered predictions for all stat categories.
    
    Args:
        last_n_df: DataFrame of recent games (sorted oldest → newest)
        milestones: Dict of category → list of thresholds
        matchup_multipliers: Optional dict of category → float multiplier
        game_context: Optional dict with 'is_home_game', 'opponent_team_id'
        full_season_df: Optional full season DataFrame for comprehensive analysis
    
    Returns:
        Comprehensive predictions with probabilities, insights, and confidence scores
    """
    if matchup_multipliers is None:
        matchup_multipliers = {}
    if game_context is None:
        game_context = {}
    
    # Get upcoming game venue
    is_home_game = game_context.get('is_home_game', True)
    
    # Use full season data for home/away analysis if available
    venue_analysis_df = full_season_df if full_season_df is not None else last_n_df
    
    predictor = UltimateEnsemblePredictor()
    predictions = {}
    
    for cat, thresholds in milestones.items():
        col = 'FG3M' if cat == '3PM' else cat
        
        if col not in last_n_df.columns:
            continue
        
        values = last_n_df[col].tolist()
        if not values:
            continue
        
        # Get ML prediction
        pred_result = predictor.predict(last_n_df, col, thresholds, cat)
        features = pred_result['features']
        
        if not features:
            continue
        
        # === HOME/AWAY ANALYSIS ===
        venue_splits = HomeAwayAnalyzer.calculate_venue_splits(venue_analysis_df, col)
        venue_factor, venue_reason = HomeAwayAnalyzer.get_venue_adjustment(venue_splits, is_home_game)
        venue_hit_rates = HomeAwayAnalyzer.calculate_venue_hit_rates(venue_analysis_df, col, thresholds)
        
        # === GAME CONTEXT ANALYSIS (B2B, Rest, Opponent) ===
        opponent_team_id = game_context.get('opponent_team_id')
        context_insights = GameContextAnalyzer.generate_context_insights(
            last_n_df, col, opponent_team_id, full_season_df
        )
        
        # Calculate combined context factor
        context_factor = (
            context_insights.get('b2b', {}).get('factor', 1.0) *
            context_insights.get('rest', {}).get('factor', 1.0) *
            context_insights.get('pace', {}).get('factor', 1.0) *
            context_insights.get('clutch', {}).get('factor', 1.0) *
            context_insights.get('minutes', {}).get('factor', 1.0) *
            context_insights.get('opponent', {}).get('factor', 1.0)
        )
        
        # Get base prediction
        raw_expected = pred_result['final_prediction']
        
        # Apply venue adjustment
        venue_adjusted = raw_expected * venue_factor
        
        # Apply context adjustment (B2B, rest, opponent)
        context_adjusted = venue_adjusted * context_factor
        
        # Apply matchup multiplier on top
        matchup_mult = matchup_multipliers.get(cat, 1.0)
        has_matchup = cat in matchup_multipliers
        expected = round(context_adjusted * matchup_mult, 1)
        
        # Get stats
        std_dev = features.get('std', 0)
        variance = std_dev ** 2
        
        # Calculate probabilities for each threshold
        over_probs = {}
        over_probs_detailed = {}
        threshold_hit_rates = {}
        
        for t in thresholds:
            prob_data = BayesianProbabilityEngine.combined_probability(
                expected, std_dev, values, t, variance
            )
            over_probs[str(t)] = prob_data['combined']
            over_probs_detailed[str(t)] = prob_data
            threshold_hit_rates[str(t)] = features.get(f'hr_{t}', 0)
        
        # Calculate confidence
        median_threshold = thresholds[len(thresholds) // 2]
        median_prob_data = over_probs_detailed.get(str(median_threshold), {})
        confidence_result = calculate_ultimate_confidence(features, pred_result, median_prob_data, cat)
        confidence = confidence_result['confidence']
        
        if has_matchup:
            confidence = min(95, confidence + 5)
        
        # Boost confidence if we have good venue data
        if venue_splits.get('significant', False):
            confidence = min(95, confidence + 3)
        
        # Generate insights (now with venue and context info)
        insights = generate_insights(features, median_prob_data, expected, threshold_hit_rates,
                                     venue_splits, is_home_game, context_insights)
        
        # Floor & ceiling with uncertainty
        uncertainty = pred_result.get('uncertainty') or std_dev or 0
        floor_val = round(max(0, expected - uncertainty * 1.3), 1)
        ceiling_val = round(expected + uncertainty * 1.3, 1)
        
        # Best bet analysis
        best_bet = None
        best_value = None
        for t in thresholds:
            prob = over_probs[str(t)]
            if prob >= 70 and expected > t:
                margin = expected - t
                if best_bet is None or margin < (expected - best_bet):
                    best_bet = t
                    best_value = prob
        
        predictions[cat] = {
            # Core predictions
            'expected': expected,
            'raw_expected': raw_expected,
            'ml_prediction': pred_result.get('ml_prediction'),
            'stat_prediction': pred_result.get('stat_prediction'),
            
            # Venue adjustments
            'venue_factor': round(venue_factor, 4),
            'venue_adjusted': round(venue_adjusted, 1),
            'venue_reason': venue_reason,
            'is_home_game': is_home_game,
            
            # Context adjustments (B2B, Rest, Opponent)
            'context_factor': round(context_factor, 4),
            'context_insights': context_insights,
            
            # Matchup adjustments
            'matchup_factor': matchup_mult,
            'matchup_applied': has_matchup,
            
            # Uncertainty & confidence
            'std_dev': round(std_dev, 1),
            'uncertainty': round(uncertainty, 2) if uncertainty else None,
            'confidence': confidence,
            'confidence_breakdown': confidence_result.get('breakdown', {}),
            'floor': floor_val,
            'ceiling': ceiling_val,
            
            # Probabilities
            'over_probs': over_probs,
            'over_probs_detailed': over_probs_detailed,
            
            # Model info
            'model_used': pred_result.get('model_used', 'statistical'),
            
            # Insights
            'insights': insights,
            
            # Best bet suggestion
            'best_bet': {
                'threshold': best_bet,
                'probability': best_value
            } if best_bet else None,
            
            # Advanced analytics
            'advanced_stats': {
                'trend_slope': round(features.get('trend_slope', 0), 3),
                'trend_r2': round(features.get('trend_r2', 0), 3),
                'momentum_3': round(features.get('momentum_3', 0), 3),
                'momentum_5': round(features.get('momentum_5', 0), 3),
                'consistency': round(features.get('consistency_score', 0), 1),
                'reliability': round(features.get('reliability_index', 0), 1),
                'hot_cold_streak': features.get('streak_length', 0) * features.get('streak_direction', 0),
                'cv': round(features.get('cv', 0), 3),
                'roll_avg_3': round(features.get('roll_mean_3', 0), 1),
                'roll_avg_5': round(features.get('roll_mean_5', 0), 1),
                'roll_avg_10': round(features.get('roll_mean_10', 0), 1),
                'last_game': features.get('last_game', 0),
                'per_36': round(features.get('per36', 0), 1),
                'games_played': features.get('games_count', 0),
                'floor_reliability': round(features.get('floor_pct', 0), 1),
                'is_b2b': features.get('is_b2b', 0),
                'min_restricted': features.get('min_restricted', 0)
            },
            
            # Home/Away splits
            'venue_splits': {
                'home_avg': venue_splits.get('home_avg', 0),
                'away_avg': venue_splits.get('away_avg', 0),
                'differential': venue_splits.get('differential', 0),
                'home_games': venue_splits.get('home_games', 0),
                'away_games': venue_splits.get('away_games', 0),
                'significant': venue_splits.get('significant', False),
                'venue_impact': venue_splits.get('venue_impact', 'neutral'),
                'hit_rates': venue_hit_rates
            }
        }
    
    # Convert all numpy types to native Python types for JSON serialization
    return convert_numpy_types(predictions)


# =============================================================================
# BACKTESTING & VALIDATION
# =============================================================================

def backtest_engine(df: pd.DataFrame, stat_col: str, thresholds: List[int],
                   test_window: int = 10) -> Dict:
    """
    Backtest the last N games to evaluate prediction accuracy.
    
    Returns:
        - Brier scores for each threshold
        - Calibration metrics
        - MAE/RMSE for point predictions
    """
    if len(df) < test_window + PredictionConfig.MIN_GAMES_FOR_ML:
        return {'error': 'Insufficient data for backtesting'}
    
    results = {
        'predictions': [],
        'actuals': [],
        'thresholds': {t: {'pred_probs': [], 'hits': []} for t in thresholds}
    }
    
    # Test on last N games
    for i in range(len(df) - test_window, len(df)):
        train_df = df.iloc[:i].copy()
        actual = df[stat_col].iloc[i]
        
        # Generate prediction using training data only
        # Only include the specific stat column we're testing
        if stat_col not in train_df.columns:
            continue
            
        # Create milestones for just this stat
        milestones = {stat_col: thresholds}
        
        try:
            pred_dict = generate_predictions(
                last_n_df=train_df,
                milestones=milestones,
                matchup_multipliers={},
                game_context={'is_home_game': True},
                full_season_df=train_df
            )
        except:
            # Skip if prediction fails
            continue
        
        if stat_col not in pred_dict:
            continue
        
        expected = pred_dict[stat_col]['expected']
        results['predictions'].append(expected)
        results['actuals'].append(actual)
        
        # Threshold probabilities
        for t in thresholds:
            prob = pred_dict[stat_col]['over_probs'][str(t)]
            hit = 1 if actual >= t else 0
            
            results['thresholds'][t]['pred_probs'].append(prob)
            results['thresholds'][t]['hits'].append(hit)
    
    # Calculate metrics
    metrics = {}
    
    if len(results['predictions']) == 0:
        return {'error': 'No valid predictions generated during backtest'}
    
    # Point prediction accuracy
    preds = np.array(results['predictions'])
    actuals = np.array(results['actuals'])
    metrics['mae'] = float(np.mean(np.abs(preds - actuals)))
    metrics['rmse'] = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    metrics['mape'] = float(np.mean(np.abs((preds - actuals) / (actuals + 1e-6))) * 100)
    
    # Threshold metrics
    metrics['thresholds'] = {}
    for t in thresholds:
        probs = np.array(results['thresholds'][t]['pred_probs'])
        hits = np.array(results['thresholds'][t]['hits'])
        
        if len(probs) == 0:
            continue
            
        # Brier score (0 = perfect, 1 = worst)
        brier = float(np.mean((probs/100 - hits) ** 2))
        
        # Calibration
        calibration = {}
        bins = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        for low, high in bins:
            mask = (probs >= low) & (probs < high)
            if mask.sum() > 0:
                actual_rate = float(hits[mask].mean() * 100)
                predicted_rate = (low + high) / 2
                calibration[f'{low}-{high}%'] = {
                    'predicted': predicted_rate,
                    'actual': actual_rate,
                    'error': abs(predicted_rate - actual_rate),
                    'count': int(mask.sum())
                }
        
        metrics['thresholds'][t] = {
            'brier_score': round(brier, 4),
            'calibration': calibration
        }
    
    return metrics


def evaluate_prediction_accuracy(df: pd.DataFrame, stat_col: str, 
                               test_size: int = 10) -> Dict:
    """
    Quick evaluation of prediction accuracy for a given stat column.
    """
    thresholds = [20, 25, 30]  # Default thresholds for evaluation
    if stat_col in ['REB', 'AST']:
        thresholds = [5, 10, 15]
    elif stat_col in ['STL', 'BLK']:
        thresholds = [1, 2, 3]
    elif stat_col == '3PM':
        thresholds = [1, 2, 3]
    
    # Filter to only include rows with valid data for the stat
    valid_df = df[df[stat_col].notna()].copy()
    
    if len(valid_df) < test_size + PredictionConfig.MIN_GAMES_FOR_ML:
        return {'error': f'Need at least {test_size + PredictionConfig.MIN_GAMES_FOR_ML} games for evaluation, got {len(valid_df)}'}
    
    return backtest_engine(valid_df, stat_col, thresholds, test_window=test_size)


# =============================================================================
# LEGACY FUNCTIONS (Backward Compatibility)
# =============================================================================

def weighted_average(values: List[float]) -> float:
    """Legacy weighted average function."""
    if not values:
        return 0.0
    n = len(values)
    alpha = 0.15
    weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()
    return round(float(np.average(values, weights=weights)), 1)


def calculate_confidence(values: List[float], has_matchup: bool = False) -> int:
    """Legacy confidence calculation."""
    if len(values) < 2:
        return 50
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    if mean == 0:
        return 50
    cv = std / mean
    confidence = max(30, min(95, int(95 - (cv * 65))))
    if has_matchup:
        confidence = min(95, confidence + 3)
    return confidence


def poisson_over_probability(lam: float, threshold: int) -> float:
    """Legacy Poisson probability function."""
    if lam <= 0:
        return 0.0
    prob = 1 - poisson.cdf(threshold - 1, lam)
    return round(prob * 100, 1)
