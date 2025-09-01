#!/usr/bin/env python3
"""
Unit tests for QCI/5QI Analysis Pipeline core functions.

Tests cover:
- compute_distributions: correct counts and percentages
- detect_mismatches: mismatch detection vs missing-mapping split
- parse_timestamp_column: ISO8601, epoch seconds/ms coercion  
- analyze_bearer_consistency: variability flags and top entities

Uses small in-memory DataFrames; no external files required.
"""

import pytest
import pandas as pd
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

# Add scripts directory to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from analyze_qos import QCIAnalyzer


class TestQCIAnalyzer:
    """Test suite for QCIAnalyzer core functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a QCIAnalyzer instance with temp output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create required subdirectories
            (temp_path / "artifacts").mkdir()
            (temp_path / "images").mkdir()
            
            analyzer = QCIAnalyzer(
                output_dir=temp_path,
                mapping_file=None,  # Will use empty mapping for tests
                window='5min'
            )
            
            # Set up minimal mapping for testing
            analyzer.qci_to_5qi_map = {
                "1": [1],
                "2": [2], 
                "3": [3],
                "5": [5]
            }
            
            # Disable logging for cleaner test output
            analyzer.logger.setLevel(logging.CRITICAL)
            
            yield analyzer
    
    def test_compute_distributions_basic(self, analyzer):
        """Test compute_distributions with basic QCI/5QI data."""
        # Create test data
        test_data = pd.DataFrame({
            'qci': [1, 1, 2, 2, 3, 5, 5, 5],
            '5qi': [1, 1, 2, 2, 3, 5, 5, 5],
            'imsi': ['123'] * 8,
            'bearer_id': ['ebi_1'] * 8
        })
        
        analyzer.data = test_data
        result = analyzer.compute_distributions()
        
        # Verify QCI distribution
        assert 'qci_distribution' in result
        qci_dist = result['qci_distribution']
        
        # Check counts
        assert qci_dist['counts'][1] == 2  # QCI 1 appears 2 times
        assert qci_dist['counts'][2] == 2  # QCI 2 appears 2 times  
        assert qci_dist['counts'][3] == 1  # QCI 3 appears 1 time
        assert qci_dist['counts'][5] == 3  # QCI 5 appears 3 times
        
        # Check percentages (should sum to 100)
        percentages = list(qci_dist['percentages'].values())
        assert abs(sum(percentages) - 100.0) < 0.01
        
        # Check specific percentages
        assert qci_dist['percentages'][1] == 25.0  # 2/8 * 100
        assert qci_dist['percentages'][5] == 37.5  # 3/8 * 100
        
        # Check total
        assert qci_dist['total_rows'] == 8
        
        # Verify 5QI distribution
        assert '5qi_distribution' in result
        qi5_dist = result['5qi_distribution']
        assert qi5_dist['total_rows'] == 8
        assert qi5_dist['counts'][5] == 3
    
    def test_compute_distributions_with_null_values(self, analyzer):
        """Test compute_distributions handling null values correctly."""
        # Create test data with null values
        test_data = pd.DataFrame({
            'qci': [1, 2, None, 3, None],
            '5qi': [1, None, 2, 3, None],
            'imsi': ['123'] * 5
        })
        
        analyzer.data = test_data
        result = analyzer.compute_distributions()
        
        # Should only count non-null values
        qci_dist = result['qci_distribution']
        assert qci_dist['total_rows'] == 3  # Only 3 non-null QCI values
        
        qi5_dist = result['5qi_distribution']
        assert qi5_dist['total_rows'] == 3  # Only 3 non-null 5QI values
    
    def test_detect_mismatches_with_violations(self, analyzer):
        """Test detect_mismatches correctly identifies mapping violations."""
        # Create test data with intentional mismatches
        test_data = pd.DataFrame({
            'qci': [1, 1, 2, 3, 5],
            '5qi': [1, 2, 2, 99, 5],  # QCI 1->5QI 2 and QCI 3->5QI 99 are mismatches
            'imsi': ['123', '456', '789', '101', '202'],
            'bearer_id': ['ebi_1', 'ebi_2', 'ebi_3', 'ebi_4', 'ebi_5']
        })
        
        analyzer.data = test_data
        result = analyzer.detect_mismatches()
        
        # Should detect 2 mismatches out of 5 comparisons
        assert result['mismatch_rate'] == 40.0  # 2/5 * 100
        assert result['total_comparisons'] == 5
        assert len(result['mismatches']) == 2
        
        # Check specific mismatches
        mismatches = result['mismatches']
        
        # Find the QCI 1 -> 5QI 2 mismatch
        qci1_mismatch = next((m for m in mismatches if m['qci'] == 1 and m['5qi'] == 2), None)
        assert qci1_mismatch is not None
        assert qci1_mismatch['expected_5qi'] == [1]
        
        # Find the QCI 3 -> 5QI 99 mismatch  
        qci3_mismatch = next((m for m in mismatches if m['qci'] == 3 and m['5qi'] == 99), None)
        assert qci3_mismatch is not None
        assert qci3_mismatch['expected_5qi'] == [3]
    
    def test_detect_mismatches_missing_mapping(self, analyzer):
        """Test detect_mismatches handling of unmapped QCI values."""
        # Create test data with QCI not in mapping
        test_data = pd.DataFrame({
            'qci': [1, 7, 8],  # QCI 7 and 8 not in test mapping
            '5qi': [1, 7, 8],
            'imsi': ['123', '456', '789'],
            'bearer_id': ['ebi_1', 'ebi_2', 'ebi_3']
        })
        
        analyzer.data = test_data
        result = analyzer.detect_mismatches()
        
        # Should only compare mapped QCI values (just QCI 1)
        # QCI 7 and 8 should be ignored since they're not in mapping
        assert result['total_comparisons'] == 3
        assert result['mismatch_rate'] == 0.0  # No mismatches for mapped values
        assert len(result['mismatches']) == 0
    
    def test_detect_mismatches_missing_columns(self, analyzer):
        """Test detect_mismatches when QCI or 5QI columns are missing."""
        # Test with missing 5QI column
        test_data = pd.DataFrame({
            'qci': [1, 2, 3],
            'imsi': ['123', '456', '789']
            # No 5qi column
        })
        
        analyzer.data = test_data
        result = analyzer.detect_mismatches()
        
        assert result['mismatch_rate'] == 0
        assert result['total_comparisons'] == 0
        assert len(result['mismatches']) == 0
    
    def test_parse_timestamp_column_iso8601(self, analyzer):
        """Test parse_timestamp_column with ISO8601 format."""
        test_data = pd.DataFrame({
            'timestamp': [
                '2024-07-31T14:42:13.123Z',
                '2024-07-31T14:42:14.234Z', 
                '2024-07-31T14:42:15.345Z'
            ],
            'qci': [1, 2, 3]
        })
        
        result = analyzer.parse_timestamp_column(test_data, 'auto')
        
        # Should successfully parse all timestamps
        assert not result['timestamp'].isna().any()
        assert all(isinstance(ts, pd.Timestamp) for ts in result['timestamp'])
        
        # Check specific parsed values
        first_ts = result['timestamp'].iloc[0]
        assert first_ts.year == 2024
        assert first_ts.month == 7
        assert first_ts.day == 31
    
    def test_parse_timestamp_column_epoch_seconds(self, analyzer):
        """Test parse_timestamp_column with epoch seconds."""
        # Use epoch timestamps (seconds since 1970-01-01)
        epoch_times = [1722434533, 1722434534, 1722434535]
        
        test_data = pd.DataFrame({
            'timestamp': epoch_times,
            'qci': [1, 2, 3]
        })
        
        result = analyzer.parse_timestamp_column(test_data, 'auto')
        
        # Should parse epoch timestamps correctly
        assert not result['timestamp'].isna().any()
        assert all(isinstance(ts, pd.Timestamp) for ts in result['timestamp'])
    
    def test_parse_timestamp_column_epoch_milliseconds(self, analyzer):
        """Test parse_timestamp_column with epoch milliseconds."""
        # Use epoch milliseconds
        epoch_ms = [1722434533123, 1722434534234, 1722434535345]
        
        test_data = pd.DataFrame({
            'timestamp': epoch_ms,
            'qci': [1, 2, 3]
        })
        
        result = analyzer.parse_timestamp_column(test_data, 'auto')
        
        # Should parse epoch milliseconds correctly
        assert not result['timestamp'].isna().any()
        assert all(isinstance(ts, pd.Timestamp) for ts in result['timestamp'])
    
    def test_parse_timestamp_column_auto_detection(self, analyzer):
        """Test parse_timestamp_column auto-detection of timestamp column."""
        # Test with 'time' column (should be auto-detected)
        test_data = pd.DataFrame({
            'time': ['2024-07-31T14:42:13Z', '2024-07-31T14:42:14Z'],
            'qci': [1, 2]
        })
        
        result = analyzer.parse_timestamp_column(test_data, 'auto')
        
        # Should auto-detect 'time' column and parse it
        assert not result['time'].isna().any()
        
        # Test with no timestamp column
        test_data_no_ts = pd.DataFrame({
            'qci': [1, 2],
            'data': ['a', 'b']
        })
        
        result_no_ts = analyzer.parse_timestamp_column(test_data_no_ts, 'auto')
        
        # Should return unchanged when no timestamp column found
        assert result_no_ts.equals(test_data_no_ts)
    
    def test_analyze_bearer_consistency_variability(self, analyzer):
        """Test analyze_bearer_consistency detects QCI/5QI variability."""
        # Create test data with bearer consistency issues
        test_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-07-31T14:42:13Z',
                '2024-07-31T14:42:14Z', 
                '2024-07-31T14:42:15Z',
                '2024-07-31T14:42:16Z'
            ]),
            'bearer_id': ['ebi_1', 'ebi_1', 'ebi_2', 'ebi_2'],
            'qci': [1, 2, 3, 3],  # ebi_1 has QCI variability (1 -> 2)
            '5qi': [1, 1, 3, 5],  # ebi_2 has 5QI variability (3 -> 5)
            'imsi': ['123'] * 4
        })
        
        analyzer.data = test_data
        analyzer.window = '5min'  # Wide window to catch all data
        result = analyzer.analyze_bearer_consistency()
        
        # Should detect anomalies
        anomalies = result['anomalies']
        assert len(anomalies) > 0
        
        # Check for QCI variability detection
        qci_anomalies = [a for a in anomalies if a['type'] == 'qci_variability']
        assert len(qci_anomalies) >= 1
        
        # Check for 5QI variability detection  
        qi5_anomalies = [a for a in anomalies if a['type'] == '5qi_variability']
        assert len(qi5_anomalies) >= 1
        
        # Verify anomaly details
        qci_anomaly = qci_anomalies[0]
        assert qci_anomaly['bearer_id'] == 'ebi_1'
        assert set(qci_anomaly['unique_qcis']) == {1, 2}
    
    def test_analyze_bearer_consistency_missing_columns(self, analyzer):
        """Test analyze_bearer_consistency with missing required columns."""
        # Test with missing timestamp
        test_data = pd.DataFrame({
            'bearer_id': ['ebi_1', 'ebi_2'],
            'qci': [1, 2]
            # No timestamp
        })
        
        analyzer.data = test_data
        result = analyzer.analyze_bearer_consistency()
        
        # Should return empty anomalies when timestamp missing
        assert result['anomalies'] == []
        
        # Test with missing bearer_id and imsi
        test_data_no_bearer = pd.DataFrame({
            'timestamp': pd.to_datetime(['2024-07-31T14:42:13Z', '2024-07-31T14:42:14Z']),
            'qci': [1, 2]
            # No bearer_id or imsi
        })
        
        analyzer.data = test_data_no_bearer
        result = analyzer.analyze_bearer_consistency()
        
        # Should return empty anomalies when no bearer identifier
        assert result['anomalies'] == []
    
    def test_analyze_bearer_consistency_consistent_data(self, analyzer):
        """Test analyze_bearer_consistency with consistent data (no anomalies)."""
        # Create test data with consistent QCI/5QI per bearer
        test_data = pd.DataFrame({
            'timestamp': pd.to_datetime([
                '2024-07-31T14:42:13Z',
                '2024-07-31T14:42:14Z',
                '2024-07-31T14:42:15Z',
                '2024-07-31T14:42:16Z'
            ]),
            'bearer_id': ['ebi_1', 'ebi_1', 'ebi_2', 'ebi_2'],
            'qci': [1, 1, 2, 2],  # Consistent QCI per bearer
            '5qi': [1, 1, 2, 2],  # Consistent 5QI per bearer
            'imsi': ['123'] * 4
        })
        
        analyzer.data = test_data
        result = analyzer.analyze_bearer_consistency()
        
        # Should detect no anomalies
        assert len(result['anomalies']) == 0


if __name__ == '__main__':
    pytest.main([__file__])