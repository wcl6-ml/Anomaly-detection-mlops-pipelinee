import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift between reference and production data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold_ks: float = 0.05, 
                 threshold_psi: float = 0.2):
        """
        Args:
            reference_data: Training/reference dataset
            threshold_ks: KS test p-value threshold (lower = more sensitive)
            threshold_psi: PSI threshold (higher = drift)
        """
        self.reference_data = reference_data
        self.threshold_ks = threshold_ks
        self.threshold_psi = threshold_psi
        self.feature_names = reference_data.columns.tolist()
        print(self.feature_names)
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                    buckets: int = 10) -> float:
        """
        Calculate Population Stability Index using reference-based binning.
        
        Args:
            expected: The reference/training data for a feature.
            actual: The new/production data for a feature.
            buckets: Number of bins to create.
        """
        # 1. Define bins based on the distribution of the EXPECTED (reference) data
        # We use quantiles to ensure each bin in 'expected' has ~10% of the data
        breakpoints = np.percentile(expected, np.arange(0, 100 + 100/buckets, 100/buckets))
        
        # Handle non-unique breakpoints (common in features with many zeros)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            # If the feature has no variance, PSI isn't meaningful
            return 0.0

        # 2. Categorize data into these fixed bins
        # np.histogram uses the reference breakpoints for both datasets
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # 3. Convert to percentages (proportions)
        expected_percents = expected_counts / len(expected)
        actual_percents = actual_counts / len(actual)

        # 4. Handle zeros to avoid infinity in log (Smoothing)
        # Using 1e-4 is a standard practice in PSI calculation
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        # 5. Calculate PSI formula: (Actual% - Expected%) * ln(Actual% / Expected%)
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

        return float(psi_value)
    
    def detect_drift(self, batch_data: pd.DataFrame) -> Dict[str, any]:
            """
            Detect drift in batch data compared to reference using 
            dynamic PSI thresholding.
            """
            results = {
                'drift_detected': False,
                'feature_drifts': {},
                'overall_psi': 0.0,
                'drifted_features': []
            }
            
            psi_values = []
            
            for feature in self.feature_names:
                ref_values = self.reference_data[feature].values
                batch_values = batch_data[feature].values
                
                # 1. KS test for statistical difference
                ks_stat, p_value = stats.ks_2samp(ref_values, batch_values)
                
                # 2. PSI calculation
                psi = self.calculate_psi(ref_values, batch_values)
                psi_values.append(psi)
                
                # 3. Individual feature drift (still useful for granular logging)
                # Note: Using the same threshold_psi here
                feature_drift = p_value < self.threshold_ks or psi > self.threshold_psi
                
                results['feature_drifts'][feature] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(p_value),
                    'psi': float(psi),
                    'drift': bool(feature_drift)
                }
                
                if feature_drift:
                    results['drifted_features'].append(feature)
            
            # 4. Overall metrics calculation
            results['overall_psi'] = float(np.mean(psi_values))
            
            # 5. Dynamic Drift Logic:
            # We flag the WHOLE batch as drifted only if the mean PSI 
            # exceeds our dynamically set threshold (e.g., 0.2 + noise).
            results['drift_detected'] = results['overall_psi'] > self.threshold_psi
            
            logger.info(
                f"Drift Check: Overall PSI={results['overall_psi']:.4f}, "
                f"Threshold={self.threshold_psi:.4f}, "
                f"Drift Detected={results['drift_detected']}"
            )
            
            return results