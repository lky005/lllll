#!/usr/bin/env python3
"""
Tests for anomaly detection functionality in QCI/5QI Analysis Pipeline.

Tests cover:
- Threshold classification (OK/WARN/CRIT) for mismatch rates
- Outlier detection behavior with deterministic DataFrames
- Boxplot-safe handling with missing or sparse KPI data
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from analyze_qos import QCIAnalyzer


class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test mapping file
        self.mapping_file = self.temp_path / "test_mapping.yaml"
        with open(self.mapping_file, 'w') as f:
            f.write("""
qci_to_5qi:
  "1": [1]
  "2": [2] 
  "3": [3]
  "9": [9]

5qi_to_qci:
  "1": [1]
  "2": [2]
  "3": [3]
  "9": [9]
""")
        
        # Create test config file with specific thresholds
        self.config_file = self.temp_path / "test_config.yaml"
        with open(self.config_file, 'w') as f:
            f.write("""
thresholds:
  mismatch_rate_warn: 0.2    # 20%
  mismatch_rate_crit: 0.5    # 50%
  kpi_outlier_zscore: 1.5    # Lower threshold for easier testing

top_n_entities: 5
time_windows: ["5min"]
charts:
  enable_time_series: false
  enable_kpi_boxplots: false
  enable_stacked_mismatch: false
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_threshold_classification_ok(self):
        """Test that entities with low mismatch rates are classified as OK."""
        # Create analyzer with test config
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create test data with no mismatches (all consistent)
        test_data = pd.DataFrame({
            'qci': [1, 1, 2, 2, 3, 3],
            '5qi': [1, 1, 2, 2, 3, 3],  # All match expected mapping
            'imsi': ['entity1', 'entity1', 'entity2', 'entity2', 'entity3', 'entity3']
        })
        
        analyzer.data = test_data
        result = analyzer.classify_entity_anomalies()
        
        # All entities should be classified as OK
        self.assertEqual(len(result['entity_scores']), 3)
        for score in result['entity_scores']:
            self.assertEqual(score['severity'], 'OK')
            self.assertEqual(score['mismatch_rate'], 0.0)
        
        # Summary should show all OK
        self.assertEqual(result['summary']['ok'], 3)
        self.assertEqual(result['summary']['warn'], 0)
        self.assertEqual(result['summary']['crit'], 0)
    
    def test_threshold_classification_warn(self):
        """Test that entities with medium mismatch rates are classified as WARN."""
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create test data with 25% mismatch rate (between warn 20% and crit 50%)
        test_data = pd.DataFrame({
            'qci': [1, 1, 1, 1],
            '5qi': [1, 1, 1, 2],  # Last one is mismatch (25% rate)
            'imsi': ['entity1', 'entity1', 'entity1', 'entity1']
        })
        
        analyzer.data = test_data
        result = analyzer.classify_entity_anomalies()
        
        # Entity should be classified as WARN
        self.assertEqual(len(result['entity_scores']), 1)
        self.assertEqual(result['entity_scores'][0]['severity'], 'WARN')
        self.assertEqual(result['entity_scores'][0]['mismatch_rate'], 0.25)
        
        # Summary should show 1 WARN
        self.assertEqual(result['summary']['warn'], 1)
    
    def test_threshold_classification_crit(self):
        """Test that entities with high mismatch rates are classified as CRIT."""
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create test data with 75% mismatch rate (above crit 50%)
        test_data = pd.DataFrame({
            'qci': [1, 1, 1, 1],
            '5qi': [1, 2, 3, 9],  # 3 mismatches out of 4 (75% rate)
            'imsi': ['entity1', 'entity1', 'entity1', 'entity1']
        })
        
        analyzer.data = test_data
        result = analyzer.classify_entity_anomalies()
        
        # Entity should be classified as CRIT
        self.assertEqual(len(result['entity_scores']), 1)
        self.assertEqual(result['entity_scores'][0]['severity'], 'CRIT')
        self.assertEqual(result['entity_scores'][0]['mismatch_rate'], 0.75)
        
        # Summary should show 1 CRIT
        self.assertEqual(result['summary']['crit'], 1)
    
    def test_outlier_detection_with_deterministic_data(self):
        """Test KPI outlier detection with deterministic data."""
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create deterministic test data with one clear outlier
        # All data points in same time window (within 5 minutes)
        base_time = pd.Timestamp('2024-01-01 10:00:00')
        test_data = pd.DataFrame({
            'qci': [1, 1, 1, 1, 1],
            '5qi': [1, 1, 1, 1, 1],
            'imsi': ['e1', 'e2', 'e3', 'e4', 'e5'],
            'throughput': [10.0, 10.0, 10.0, 10.0, 100.0],  # Clear outlier
            'timestamp': [base_time + pd.Timedelta(seconds=i) for i in range(5)]  # All within same 5min window
        })
        
        analyzer.data = test_data
        result = analyzer.detect_kpi_outliers()
        
        # Debug output
        print(f"Test data shape: {test_data.shape}")
        print(f"Timestamps range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
        print(f"Outliers detected: {len(result['outliers'])}")
        if result['outliers']:
            for outlier in result['outliers']:
                print(f"Outlier: {outlier}")
        
        # Should detect the outlier
        self.assertGreaterEqual(len(result['outliers']), 1)
        
        # Check outlier details
        outlier = result['outliers'][0]
        self.assertEqual(outlier['kpi'], 'throughput')
        self.assertEqual(outlier['value'], 100.0)
        self.assertGreater(outlier['z_score'], 1.5)  # Should be around 2.0
    
    def test_kpi_analysis_with_missing_data(self):
        """Test that KPI analysis handles missing/sparse data gracefully."""
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create sparse data with lots of missing values
        test_data = pd.DataFrame({
            'qci': [1, 1, 2, 2, 3],
            '5qi': [1, 1, 2, 2, 3],
            'imsi': ['e1', 'e2', 'e3', 'e4', 'e5'],
            'throughput': [10.0, np.nan, np.nan, 20.0, np.nan],  # Sparse data
            'latency': [np.nan, 50.0, np.nan, np.nan, 30.0],    # Sparse data
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min')
        })
        
        analyzer.data = test_data
        
        # Should not crash and should handle missing data gracefully
        kpi_stats = analyzer.compute_kpi_stats()
        outliers = analyzer.detect_kpi_outliers()
        
        # Should generate some stats despite missing data
        self.assertIsInstance(kpi_stats, dict)
        self.assertIsInstance(outliers, dict)
        
        # Outlier detection should handle insufficient data gracefully
        self.assertIsInstance(outliers['outliers'], list)
    
    def test_boxplot_safe_handling(self):
        """Test that boxplot creation handles edge cases safely."""
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(self.config_file)
        )
        
        # Create minimal data that might cause boxplot issues
        test_data = pd.DataFrame({
            'qci': [1],
            '5qi': [1],
            'imsi': ['e1'],
            'throughput': [10.0],  # Single data point
            'timestamp': [pd.Timestamp('2024-01-01')]
        })
        
        analyzer.data = test_data
        
        # Should not crash when creating charts with minimal data
        try:
            analyzer.create_kpi_boxplots()
            success = True
        except Exception as e:
            success = False
            print(f"Boxplot creation failed: {e}")
        
        self.assertTrue(success, "Boxplot creation should handle minimal data gracefully")
    
    def test_config_loading_with_defaults(self):
        """Test that configuration loading works with partial configs."""
        # Create partial config (missing some settings)
        partial_config_file = self.temp_path / "partial_config.yaml"
        with open(partial_config_file, 'w') as f:
            f.write("""
thresholds:
  mismatch_rate_warn: 0.1
# Missing other settings - should use defaults
""")
        
        analyzer = QCIAnalyzer(
            mapping_file=str(self.mapping_file),
            output_dir=str(self.temp_path / "output"),
            config_file=str(partial_config_file)
        )
        
        # Should have loaded the custom setting
        self.assertEqual(analyzer.config['thresholds']['mismatch_rate_warn'], 0.1)
        
        # Should have default values for missing settings
        self.assertEqual(analyzer.config['thresholds']['mismatch_rate_crit'], 0.15)
        self.assertEqual(analyzer.config['top_n_entities'], 10)


if __name__ == '__main__':
    unittest.main()