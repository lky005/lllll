#!/usr/bin/env python3
"""
QCI/5QI Analysis Pipeline

Analyzes uploaded logs for QCI/5QI patterns, detects inconsistencies and anomalies,
and generates a reproducible analysis report with supporting charts and artifacts.

Author: Auto-generated analysis pipeline
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import yaml
    from dateutil import parser as date_parser
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install dependencies with: pip install -r requirements.txt")
    sys.exit(1)

# Use Agg backend for matplotlib to avoid display issues
matplotlib.use('Agg')


class QCIAnalyzer:
    """Main QCI/5QI analysis engine."""
    
    def __init__(self, mapping_file: str, output_dir: str, window: str = "5min", verbose: bool = False, config_file: str = None):
        """Initialize analyzer with configuration."""
        self.mapping_file = mapping_file
        self.output_dir = Path(output_dir)
        self.window = window
        self.verbose = verbose
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)
        
        # Setup logging first
        self.setup_logging()
        
        # Load analysis configuration (with defaults)
        self.config = self.load_analysis_config(config_file)
        self._config_file = config_file or 'built-in defaults'
        
        # Load QCI/5QI mapping
        self.qci_to_5qi_map, self.qi5_to_qci_map = self.load_mapping()
        
        # Initialize data storage
        self.data = None
        self.file_info = []
        
    def load_analysis_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load analysis configuration with defaults."""
        # Default configuration
        default_config = {
            'thresholds': {
                'mismatch_rate_warn': 0.05,
                'mismatch_rate_crit': 0.15,
                'kpi_outlier_zscore': 3.0
            },
            'top_n_entities': 10,
            'time_windows': ["5min", "15min", "1H"],
            'charts': {
                'enable_time_series': True,
                'enable_kpi_boxplots': True,
                'enable_stacked_mismatch': True
            },
            'features': {
                'entity_scoring': True,
                'enhanced_kpi_stats': True,
                'variability_analysis': True
            }
        }
        
        # If no config file provided, return defaults
        if not config_file:
            return default_config
        
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.warning(f"Config file {config_file} not found, using defaults")
                return default_config
                
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Deep merge user config with defaults
            def deep_merge(default: dict, user: dict) -> dict:
                merged = default.copy()
                for key, value in user.items():
                    if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            
            merged_config = deep_merge(default_config, user_config)
            self.logger.info(f"Loaded configuration from {config_file}")
            return merged_config
            
        except Exception as e:
            self.logger.warning(f"Failed to load config file {config_file}: {e}, using defaults")
            return default_config
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "run.log")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def load_mapping(self) -> Tuple[Dict, Dict]:
        """Load QCI/5QI mapping from YAML file."""
        try:
            with open(self.mapping_file, 'r') as f:
                mapping_data = yaml.safe_load(f)
            
            qci_to_5qi = mapping_data.get('qci_to_5qi', {})
            qi5_to_qci = mapping_data.get('5qi_to_qci', {})
            
            self.logger.info(f"Loaded mapping with {len(qci_to_5qi)} QCI->5QI entries")
            return qci_to_5qi, qi5_to_qci
            
        except Exception as e:
            self.logger.error(f"Failed to load mapping file {self.mapping_file}: {e}")
            return {}, {}
    
    def discover_files(self, input_dir: str) -> List[Path]:
        """Discover CSV, TSV, and JSON files in input directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            self.logger.error(f"Input directory {input_dir} does not exist")
            return []
        
        extensions = ['.csv', '.tsv', '.json', '.jsonl']
        files = []
        
        for ext in extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"**/*{ext}"))
        
        self.logger.info(f"Discovered {len(files)} data files: {[f.name for f in files]}")
        return files
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and map common synonyms."""
        # Convert to lowercase and strip spaces
        df.columns = df.columns.str.lower().str.strip()
        
        # Map common synonyms
        column_mapping = {
            'five_qi': '5qi',
            'qi5': '5qi',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'ebi': 'bearer_id',
            'qfi': 'bearer_id',
            'tx_bytes': 'throughput',
            'rx_bytes': 'throughput',
            'rtp_jitter': 'latency',
            'loss_rate': 'packet_loss'
        }
        
        df = df.rename(columns=column_mapping)
        
        self.logger.debug(f"Normalized columns: {list(df.columns)}")
        return df
    
    def parse_timestamp_column(self, df: pd.DataFrame, time_col: str = 'auto') -> pd.DataFrame:
        """Parse timestamp column robustly."""
        if time_col == 'auto':
            # Auto-detect timestamp column
            timestamp_candidates = ['timestamp', 'time', 'datetime']
            time_col = None
            
            for col in timestamp_candidates:
                if col in df.columns:
                    time_col = col
                    break
        
        if time_col is None or time_col not in df.columns:
            self.logger.warning("No timestamp column found, skipping time-based analysis")
            return df
        
        try:
            # Try pandas datetime parsing first
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # If that fails, try dateutil parser
            if df[time_col].isna().all():
                df[time_col] = df[time_col].apply(
                    lambda x: date_parser.parse(str(x)) if pd.notna(x) else None
                )
            
            valid_timestamps = df[time_col].notna().sum()
            self.logger.info(f"Parsed {valid_timestamps}/{len(df)} timestamps successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamps in column {time_col}: {e}")
        
        return df
    
    def load_and_process_files(self, files: List[Path], time_col: str = 'auto') -> pd.DataFrame:
        """Load and process discovered files."""
        dataframes = []
        
        for file_path in files:
            try:
                self.logger.info(f"Processing file: {file_path.name}")
                
                if file_path.suffix.lower() == '.json':
                    # Handle JSON files
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                        
                elif file_path.suffix.lower() == '.jsonl':
                    # Handle JSON Lines files
                    df = pd.read_json(file_path, lines=True)
                    
                elif file_path.suffix.lower() == '.tsv':
                    # Handle TSV files
                    df = pd.read_csv(file_path, sep='\t')
                    
                else:
                    # Handle CSV files (default)
                    df = pd.read_csv(file_path)
                
                # Normalize column names
                df = self.normalize_column_names(df)
                
                # Parse timestamps
                df = self.parse_timestamp_column(df, time_col)
                
                # Add source file info
                df['_source_file'] = file_path.name
                
                dataframes.append(df)
                
                self.file_info.append({
                    'filename': file_path.name,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'has_qci': 'qci' in df.columns,
                    'has_5qi': '5qi' in df.columns,
                    'has_timestamp': 'timestamp' in df.columns
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process file {file_path.name}: {e}")
        
        if not dataframes:
            self.logger.error("No files could be processed successfully")
            return pd.DataFrame()
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        self.logger.info(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def compute_distributions(self) -> Dict[str, Any]:
        """Compute QCI and 5QI distributions."""
        results = {}
        
        # QCI distribution
        if 'qci' in self.data.columns:
            qci_counts = self.data['qci'].value_counts().sort_index()
            qci_total = len(self.data[self.data['qci'].notna()])
            qci_percentages = (qci_counts / qci_total * 100).round(2)
            
            results['qci_distribution'] = {
                'counts': qci_counts.to_dict(),
                'percentages': qci_percentages.to_dict(),
                'total_rows': qci_total
            }
            
            # Save to artifacts
            qci_df = pd.DataFrame({
                'qci': qci_counts.index,
                'count': qci_counts.values,
                'percentage': qci_percentages.values
            })
            qci_df.to_csv(self.output_dir / "artifacts" / "qci_distribution.csv", index=False)
        
        # 5QI distribution
        if '5qi' in self.data.columns:
            qi5_counts = self.data['5qi'].value_counts().sort_index()
            qi5_total = len(self.data[self.data['5qi'].notna()])
            qi5_percentages = (qi5_counts / qi5_total * 100).round(2)
            
            results['5qi_distribution'] = {
                'counts': qi5_counts.to_dict(),
                'percentages': qi5_percentages.to_dict(),
                'total_rows': qi5_total
            }
            
            # Save to artifacts
            qi5_df = pd.DataFrame({
                '5qi': qi5_counts.index,
                'count': qi5_counts.values,
                'percentage': qi5_percentages.values
            })
            qi5_df.to_csv(self.output_dir / "artifacts" / "5qi_distribution.csv", index=False)
        
        return results
    
    def detect_mismatches(self) -> Dict[str, Any]:
        """Detect QCI/5QI mapping mismatches."""
        if 'qci' not in self.data.columns or '5qi' not in self.data.columns:
            self.logger.warning("Both QCI and 5QI columns required for mismatch detection")
            return {'mismatch_rate': 0, 'total_comparisons': 0, 'mismatches': []}
        
        # Filter rows with both QCI and 5QI values
        both_present = self.data[self.data['qci'].notna() & self.data['5qi'].notna()].copy()
        
        if len(both_present) == 0:
            self.logger.warning("No rows with both QCI and 5QI values found")
            return {'mismatch_rate': 0, 'total_comparisons': 0, 'mismatches': []}
        
        # Check mismatches
        mismatches = []
        for idx, row in both_present.iterrows():
            qci_str = str(int(row['qci']))
            qi5_val = int(row['5qi'])
            
            expected_5qis = self.qci_to_5qi_map.get(qci_str, [])
            
            if expected_5qis and qi5_val not in expected_5qis:
                mismatches.append({
                    'row_index': idx,
                    'qci': int(row['qci']),
                    '5qi': qi5_val,
                    'expected_5qi': expected_5qis,
                    'imsi': row.get('imsi', 'N/A'),
                    'bearer_id': row.get('bearer_id', 'N/A'),
                    'timestamp': row.get('timestamp', 'N/A')
                })
        
        mismatch_rate = len(mismatches) / len(both_present) * 100
        
        results = {
            'mismatch_rate': round(mismatch_rate, 2),
            'total_comparisons': len(both_present),
            'mismatches': mismatches[:100]  # Limit to first 100 for report
        }
        
        # Save detailed mismatches to artifacts
        if mismatches:
            mismatch_df = pd.DataFrame(mismatches)
            mismatch_df.to_csv(self.output_dir / "artifacts" / "mismatches.csv", index=False)
        
        self.logger.info(f"Detected {len(mismatches)} mismatches out of {len(both_present)} comparisons ({mismatch_rate:.2f}%)")
        
        return results
    
    def classify_entity_anomalies(self) -> Dict[str, Any]:
        """Classify entities by mismatch rate thresholds and detect anomalies."""
        if 'qci' not in self.data.columns or '5qi' not in self.data.columns:
            return {'entity_scores': [], 'summary': {'ok': 0, 'warn': 0, 'crit': 0}}
        
        # Get thresholds from config
        warn_thresh = self.config['thresholds']['mismatch_rate_warn']
        crit_thresh = self.config['thresholds']['mismatch_rate_crit']
        
        # Group by entity (IMSI or bearer_id)
        entity_col = 'imsi' if 'imsi' in self.data.columns else 'bearer_id'
        if entity_col not in self.data.columns:
            return {'entity_scores': [], 'summary': {'ok': 0, 'warn': 0, 'crit': 0}}
        
        entity_scores = []
        both_present = self.data[self.data['qci'].notna() & self.data['5qi'].notna()]
        
        for entity_id, entity_data in both_present.groupby(entity_col):
            mismatches = 0
            total_comparisons = len(entity_data)
            
            for _, row in entity_data.iterrows():
                qci_str = str(int(row['qci']))
                qi5_val = int(row['5qi'])
                expected_5qis = self.qci_to_5qi_map.get(qci_str, [])
                
                if expected_5qis and qi5_val not in expected_5qis:
                    mismatches += 1
            
            mismatch_rate = mismatches / total_comparisons if total_comparisons > 0 else 0
            
            # Classify severity
            if mismatch_rate >= crit_thresh:
                severity = 'CRIT'
            elif mismatch_rate >= warn_thresh:
                severity = 'WARN' 
            else:
                severity = 'OK'
            
            entity_scores.append({
                'entity_id': entity_id,
                'entity_type': entity_col,
                'mismatch_count': mismatches,
                'total_comparisons': total_comparisons,
                'mismatch_rate': round(mismatch_rate, 4),
                'severity': severity
            })
        
        # Sort by mismatch rate descending
        entity_scores.sort(key=lambda x: x['mismatch_rate'], reverse=True)
        
        # Count by severity
        summary = {'ok': 0, 'warn': 0, 'crit': 0}
        for score in entity_scores:
            if score['severity'] == 'OK':
                summary['ok'] += 1
            elif score['severity'] == 'WARN':
                summary['warn'] += 1
            elif score['severity'] == 'CRIT':
                summary['crit'] += 1
        
        # Save detailed scoring to artifacts
        if entity_scores:
            score_df = pd.DataFrame(entity_scores)
            score_df.to_csv(self.output_dir / "artifacts" / "mismatch_by_entity.csv", index=False)
        
        return {'entity_scores': entity_scores, 'summary': summary}
    
    def detect_kpi_outliers(self) -> Dict[str, Any]:
        """Detect KPI outliers using z-score analysis."""
        kpi_columns = ['throughput', 'latency', 'packet_loss']
        available_kpis = [col for col in kpi_columns if col in self.data.columns]
        
        if not available_kpis:
            return {'outliers': []}
        
        # Get threshold from config
        zscore_thresh = self.config['thresholds']['kpi_outlier_zscore']
        
        outliers = []
        
        # Group by time window if timestamps available
        if 'timestamp' in self.data.columns and self.data['timestamp'].notna().any():
            # Use existing time_window column if available, otherwise create it
            if 'time_window' not in self.data.columns:
                self.data = self.data.copy()
                self.data['time_window'] = self.data['timestamp'].dt.floor(self.window)
            
            for window, window_data in self.data.groupby('time_window'):
                for kpi in available_kpis:
                    kpi_values = window_data[kpi].dropna()
                    if len(kpi_values) < 3:  # Need at least 3 points for meaningful z-score
                        continue
                    
                    mean_val = kpi_values.mean()
                    std_val = kpi_values.std()
                    
                    if std_val == 0:  # No variation
                        continue
                    
                    # Calculate z-scores
                    z_scores = (kpi_values - mean_val) / std_val
                    outlier_mask = abs(z_scores) > zscore_thresh
                    
                    if outlier_mask.any():
                        outlier_indices = z_scores[outlier_mask].index
                        for idx in outlier_indices:
                            row = window_data.loc[idx]
                            outliers.append({
                                'time_window': window,
                                'entity_id': row.get('imsi', row.get('bearer_id', 'N/A')),
                                'kpi': kpi,
                                'value': round(row[kpi], 3),
                                'z_score': round(abs(z_scores[idx]), 2),
                                'window_mean': round(mean_val, 3),
                                'window_std': round(std_val, 3),
                                'severity': 'HIGH' if abs(z_scores[idx]) > zscore_thresh * 1.5 else 'MEDIUM'
                            })
        
        # Save outliers to artifacts
        if outliers:
            outlier_df = pd.DataFrame(outliers)
            outlier_df.to_csv(self.output_dir / "artifacts" / "kpi_outliers.csv", index=False)
        
        self.logger.info(f"Detected {len(outliers)} KPI outliers")
        return {'outliers': outliers}
    
    def analyze_bearer_consistency(self) -> Dict[str, Any]:
        """Analyze consistency per bearer/IMSI over time windows."""
        if 'timestamp' not in self.data.columns:
            self.logger.warning("Timestamp column required for bearer consistency analysis")
            return {'anomalies': []}
        
        # Group by bearer and time window
        bearer_col = 'bearer_id' if 'bearer_id' in self.data.columns else 'imsi'
        if bearer_col not in self.data.columns:
            self.logger.warning("Neither bearer_id nor imsi column found for consistency analysis")
            return {'anomalies': []}
        
        # Create time windows
        self.data = self.data.copy()
        self.data['time_window'] = self.data['timestamp'].dt.floor(self.window)
        
        anomalies = []
        
        for (bearer, window), group in self.data.groupby([bearer_col, 'time_window']):
            # Check QCI variability
            if 'qci' in group.columns:
                unique_qcis = group['qci'].dropna().nunique()
                if unique_qcis > 1:
                    anomalies.append({
                        'bearer_id': bearer,
                        'time_window': window,
                        'type': 'qci_variability',
                        'unique_qcis': sorted(group['qci'].dropna().unique().astype(int).tolist()),
                        'row_count': len(group)
                    })
            
            # Check 5QI variability
            if '5qi' in group.columns:
                unique_5qis = group['5qi'].dropna().nunique()
                if unique_5qis > 1:
                    anomalies.append({
                        'bearer_id': bearer,
                        'time_window': window,
                        'type': '5qi_variability',
                        'unique_5qis': sorted(group['5qi'].dropna().unique().astype(int).tolist()),
                        'row_count': len(group)
                    })
        
        # Save anomalies to artifacts
        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            anomaly_df.to_csv(self.output_dir / "artifacts" / "anomalies_by_bearer.csv", index=False)
        
        self.logger.info(f"Detected {len(anomalies)} bearer consistency anomalies")
        
        return {'anomalies': anomalies[:50]}  # Limit for report
    
    def compute_kpi_stats(self) -> Dict[str, Any]:
        """Compute KPI statistics if available."""
        kpi_columns = ['throughput', 'latency', 'packet_loss']
        available_kpis = [col for col in kpi_columns if col in self.data.columns]
        
        if not available_kpis:
            self.logger.info("No KPI columns found, skipping KPI analysis")
            return {}
        
        results = {}
        
        # Enhanced stats with percentiles if enabled
        enhanced_stats = self.config['features']['enhanced_kpi_stats']
        
        # Stats per QCI
        if 'qci' in self.data.columns:
            qci_stats = {}
            qci_detailed = []
            
            for qci in self.data['qci'].dropna().unique():
                qci_data = self.data[self.data['qci'] == qci]
                qci_stats[int(qci)] = {}
                
                for kpi in available_kpis:
                    kpi_values = qci_data[kpi].dropna()
                    if len(kpi_values) > 0:
                        basic_stats = {
                            'mean': round(kpi_values.mean(), 3),
                            'median': round(kpi_values.median(), 3),
                            'min': round(kpi_values.min(), 3),
                            'max': round(kpi_values.max(), 3),
                            'count': len(kpi_values)
                        }
                        
                        if enhanced_stats:
                            basic_stats.update({
                                'p50': round(kpi_values.quantile(0.5), 3),
                                'p90': round(kpi_values.quantile(0.9), 3),
                                'p99': round(kpi_values.quantile(0.99), 3),
                                'std': round(kpi_values.std(), 3)
                            })
                        
                        qci_stats[int(qci)][kpi] = basic_stats
                        
                        # Detailed stats for CSV export
                        if enhanced_stats:
                            qci_detailed.append({
                                'qci': int(qci),
                                'kpi': kpi,
                                'count': len(kpi_values),
                                'mean': round(kpi_values.mean(), 3),
                                'median': round(kpi_values.median(), 3),
                                'min': round(kpi_values.min(), 3),
                                'max': round(kpi_values.max(), 3),
                                'std': round(kpi_values.std(), 3),
                                'p50': round(kpi_values.quantile(0.5), 3),
                                'p90': round(kpi_values.quantile(0.9), 3),
                                'p99': round(kpi_values.quantile(0.99), 3)
                            })
            
            results['qci_kpi_stats'] = qci_stats
            
            # Save detailed QCI stats to CSV
            if enhanced_stats and qci_detailed:
                qci_df = pd.DataFrame(qci_detailed)
                qci_df.to_csv(self.output_dir / "artifacts" / "kpi_stats_by_qci.csv", index=False)
        
        # Stats per 5QI
        if '5qi' in self.data.columns:
            qi5_stats = {}
            qi5_detailed = []
            
            for qi5 in self.data['5qi'].dropna().unique():
                qi5_data = self.data[self.data['5qi'] == qi5]
                qi5_stats[int(qi5)] = {}
                
                for kpi in available_kpis:
                    kpi_values = qi5_data[kpi].dropna()
                    if len(kpi_values) > 0:
                        basic_stats = {
                            'mean': round(kpi_values.mean(), 3),
                            'median': round(kpi_values.median(), 3),
                            'min': round(kpi_values.min(), 3),
                            'max': round(kpi_values.max(), 3),
                            'count': len(kpi_values)
                        }
                        
                        if enhanced_stats:
                            basic_stats.update({
                                'p50': round(kpi_values.quantile(0.5), 3),
                                'p90': round(kpi_values.quantile(0.9), 3),
                                'p99': round(kpi_values.quantile(0.99), 3),
                                'std': round(kpi_values.std(), 3)
                            })
                        
                        qi5_stats[int(qi5)][kpi] = basic_stats
                        
                        # Detailed stats for CSV export
                        if enhanced_stats:
                            qi5_detailed.append({
                                '5qi': int(qi5),
                                'kpi': kpi,
                                'count': len(kpi_values),
                                'mean': round(kpi_values.mean(), 3),
                                'median': round(kpi_values.median(), 3),
                                'min': round(kpi_values.min(), 3),
                                'max': round(kpi_values.max(), 3),
                                'std': round(kpi_values.std(), 3),
                                'p50': round(kpi_values.quantile(0.5), 3),
                                'p90': round(kpi_values.quantile(0.9), 3),
                                'p99': round(kpi_values.quantile(0.99), 3)
                            })
            
            results['5qi_kpi_stats'] = qi5_stats
            
            # Save detailed 5QI stats to CSV
            if enhanced_stats and qi5_detailed:
                qi5_df = pd.DataFrame(qi5_detailed)
                qi5_df.to_csv(self.output_dir / "artifacts" / "kpi_stats_by_5qi.csv", index=False)
        
        return results
    
    def create_charts(self, distributions: Dict[str, Any]):
        """Create visualization charts."""
        plt.style.use('default')
        
        # QCI Distribution Bar Chart
        if 'qci_distribution' in distributions:
            fig, ax = plt.subplots(figsize=(10, 6))
            qci_data = distributions['qci_distribution']
            
            qcis = list(qci_data['counts'].keys())
            counts = list(qci_data['counts'].values())
            
            bars = ax.bar(qcis, counts, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel('QCI Value')
            ax.set_ylabel('Count')
            ax.set_title('QCI Distribution')
            ax.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                       f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "images" / "qci_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5QI Distribution Bar Chart
        if '5qi_distribution' in distributions:
            fig, ax = plt.subplots(figsize=(10, 6))
            qi5_data = distributions['5qi_distribution']
            
            qi5s = list(qi5_data['counts'].keys())
            counts = list(qi5_data['counts'].values())
            
            bars = ax.bar(qi5s, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax.set_xlabel('5QI Value')
            ax.set_ylabel('Count')
            ax.set_title('5QI Distribution')
            ax.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                       f'{count}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "images" / "5qi_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Stacked mismatch chart if enabled
        if self.config['charts']['enable_stacked_mismatch'] and 'qci' in self.data.columns and '5qi' in self.data.columns:
            self.create_stacked_mismatch_chart()
        
        # KPI boxplots if enabled
        if self.config['charts']['enable_kpi_boxplots']:
            self.create_kpi_boxplots()
        
        # Time series charts if timestamp available
        if 'timestamp' in self.data.columns and self.data['timestamp'].notna().any():
            self.create_timeseries_charts()
            
            # Multiple time window series if enabled
            if self.config['charts']['enable_time_series']:
                self.create_multi_window_timeseries()
    
    def create_timeseries_charts(self):
        """Create time series charts for QCI/5QI counts over time."""
        # Resample by time window
        self.data_ts = self.data.set_index('timestamp')
        
        if 'qci' in self.data.columns:
            # QCI over time
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for qci in sorted(self.data['qci'].dropna().unique()):
                qci_data = self.data_ts[self.data_ts['qci'] == qci]
                qci_counts = qci_data.resample(self.window).size()
                ax.plot(qci_counts.index, qci_counts.values, marker='o', label=f'QCI {int(qci)}')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Count per Window')
            ax.set_title(f'QCI Counts Over Time (Window: {self.window})')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "images" / "qci_timeseries.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_stacked_mismatch_chart(self):
        """Create stacked bar chart for QCI vs 5QI mismatch categories."""
        try:
            both_present = self.data[self.data['qci'].notna() & self.data['5qi'].notna()]
            if len(both_present) == 0:
                return
            
            # Categorize each QCI/5QI pair
            categories = {'consistent': {}, 'mismapped': {}, 'unknown': {}}
            
            for qci in both_present['qci'].dropna().unique():
                qci_str = str(int(qci))
                qci_data = both_present[both_present['qci'] == qci]
                
                categories['consistent'][int(qci)] = 0
                categories['mismapped'][int(qci)] = 0
                categories['unknown'][int(qci)] = 0
                
                for _, row in qci_data.iterrows():
                    qi5_val = int(row['5qi'])
                    expected_5qis = self.qci_to_5qi_map.get(qci_str, [])
                    
                    if not expected_5qis:
                        categories['unknown'][int(qci)] += 1
                    elif qi5_val in expected_5qis:
                        categories['consistent'][int(qci)] += 1
                    else:
                        categories['mismapped'][int(qci)] += 1
            
            # Create stacked bar chart
            qcis = sorted(categories['consistent'].keys())
            consistent_counts = [categories['consistent'][qci] for qci in qcis]
            mismapped_counts = [categories['mismapped'][qci] for qci in qcis]
            unknown_counts = [categories['unknown'][qci] for qci in qcis]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.bar(qcis, consistent_counts, label='Consistent', color='lightgreen', alpha=0.8)
            ax.bar(qcis, mismapped_counts, bottom=consistent_counts, label='Mismapped', color='lightcoral', alpha=0.8)
            ax.bar(qcis, unknown_counts, 
                   bottom=[c+m for c,m in zip(consistent_counts, mismapped_counts)], 
                   label='Unknown', color='lightgray', alpha=0.8)
            
            ax.set_xlabel('QCI Value')
            ax.set_ylabel('Count')
            ax.set_title('QCI/5QI Mapping Consistency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "images" / "stacked_mismatch.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create stacked mismatch chart: {e}")
    
    def create_kpi_boxplots(self):
        """Create boxplot charts for KPI metrics by QCI and 5QI."""
        try:
            kpi_columns = ['throughput', 'latency', 'packet_loss']
            available_kpis = [col for col in kpi_columns if col in self.data.columns]
            
            if not available_kpis:
                return
            
            # Boxplots by QCI
            if 'qci' in self.data.columns:
                for kpi in available_kpis:
                    kpi_data = self.data[self.data[kpi].notna() & self.data['qci'].notna()]
                    if len(kpi_data) < 2:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Prepare data for boxplot
                    qci_values = sorted(kpi_data['qci'].unique())
                    boxplot_data = [kpi_data[kpi_data['qci'] == qci][kpi].values for qci in qci_values]
                    
                    # Filter out empty arrays
                    non_empty_data = []
                    non_empty_labels = []
                    for i, data in enumerate(boxplot_data):
                        if len(data) > 0:
                            non_empty_data.append(data)
                            non_empty_labels.append(f'QCI {int(qci_values[i])}')
                    
                    if non_empty_data:
                        ax.boxplot(non_empty_data, tick_labels=non_empty_labels)
                        ax.set_ylabel(kpi.replace('_', ' ').title())
                        ax.set_title(f'{kpi.replace("_", " ").title()} Distribution by QCI')
                        ax.grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(self.output_dir / "images" / f"kpi_{kpi}_by_qci_boxplot.png", 
                                   dpi=300, bbox_inches='tight')
                    plt.close()
            
            # Boxplots by 5QI
            if '5qi' in self.data.columns:
                for kpi in available_kpis:
                    kpi_data = self.data[self.data[kpi].notna() & self.data['5qi'].notna()]
                    if len(kpi_data) < 2:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Prepare data for boxplot
                    qi5_values = sorted(kpi_data['5qi'].unique())
                    boxplot_data = [kpi_data[kpi_data['5qi'] == qi5][kpi].values for qi5 in qi5_values]
                    
                    # Filter out empty arrays
                    non_empty_data = []
                    non_empty_labels = []
                    for i, data in enumerate(boxplot_data):
                        if len(data) > 0:
                            non_empty_data.append(data)
                            non_empty_labels.append(f'5QI {int(qi5_values[i])}')
                    
                    if non_empty_data:
                        ax.boxplot(non_empty_data, tick_labels=non_empty_labels)
                        ax.set_ylabel(kpi.replace('_', ' ').title())
                        ax.set_title(f'{kpi.replace("_", " ").title()} Distribution by 5QI')
                        ax.grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(self.output_dir / "images" / f"kpi_{kpi}_by_5qi_boxplot.png", 
                                   dpi=300, bbox_inches='tight')
                    plt.close()
                    
        except Exception as e:
            self.logger.warning(f"Failed to create KPI boxplots: {e}")
    
    def create_multi_window_timeseries(self):
        """Create time series charts for multiple time windows."""
        try:
            if 'qci' not in self.data.columns or 'timestamp' not in self.data.columns:
                return
            
            time_windows = self.config['time_windows']
            data_ts = self.data.set_index('timestamp')
            
            for window in time_windows:
                try:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    for qci in sorted(self.data['qci'].dropna().unique()):
                        qci_data = data_ts[data_ts['qci'] == qci]
                        if len(qci_data) == 0:
                            continue
                        qci_counts = qci_data.resample(window).size()
                        if len(qci_counts) > 0:
                            ax.plot(qci_counts.index, qci_counts.values, 
                                   marker='o', label=f'QCI {int(qci)}', alpha=0.7)
                    
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Count per Window')
                    ax.set_title(f'QCI Counts Over Time (Window: {window})')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / "images" / f"qci_timeseries_{window}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    self.logger.debug(f"Failed to create timeseries for window {window}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to create multi-window timeseries: {e}")
    
    def create_anomalies_summary(self, entity_anomalies: Dict, bearer_anomalies: Dict, kpi_outliers: Dict) -> Dict[str, Any]:
        """Create comprehensive anomalies summary."""
        anomalies_summary = []
        
        # Add entity-level mismatch anomalies
        top_n = self.config['top_n_entities']
        for score in entity_anomalies['entity_scores'][:top_n]:
            if score['severity'] in ['WARN', 'CRIT']:
                anomalies_summary.append({
                    'entity': score['entity_id'],
                    'type': 'mismatch_rate',
                    'severity': score['severity'],
                    'counts': score['mismatch_count'],
                    'window': 'overall',
                    'notes': f"Mismatch rate: {score['mismatch_rate']:.1%} ({score['mismatch_count']}/{score['total_comparisons']})"
                })
        
        # Add bearer variability anomalies
        for anomaly in bearer_anomalies['anomalies']:
            severity = 'WARN' if anomaly['row_count'] <= 5 else 'CRIT'
            anomalies_summary.append({
                'entity': anomaly['bearer_id'],
                'type': anomaly['type'],
                'severity': severity,
                'counts': anomaly['row_count'],
                'window': str(anomaly['time_window']) if 'time_window' in anomaly else 'N/A',
                'notes': f"Multiple values: {anomaly.get('unique_qcis', anomaly.get('unique_5qis', []))}"
            })
        
        # Add KPI outliers (top severity only)
        high_severity_outliers = [o for o in kpi_outliers['outliers'] if o['severity'] == 'HIGH']
        for outlier in high_severity_outliers[:top_n]:
            anomalies_summary.append({
                'entity': outlier['entity_id'],
                'type': f"kpi_outlier_{outlier['kpi']}",
                'severity': 'CRIT',
                'counts': 1,
                'window': str(outlier['time_window']) if 'time_window' in outlier else 'N/A',
                'notes': f"Z-score: {outlier['z_score']}, Value: {outlier['value']}"
            })
        
        # Sort by severity and entity
        severity_order = {'CRIT': 0, 'WARN': 1, 'OK': 2}
        anomalies_summary.sort(key=lambda x: (severity_order[x['severity']], x['entity']))
        
        # Save to artifacts
        if anomalies_summary:
            summary_df = pd.DataFrame(anomalies_summary)
            summary_df.to_csv(self.output_dir / "artifacts" / "anomalies_summary.csv", index=False)
        
        return {'summary': anomalies_summary}
    
    def generate_report(self, distributions: Dict, mismatches: Dict, anomalies: Dict, kpi_stats: Dict, 
                       entity_anomalies: Dict = None, kpi_outliers: Dict = None, anomalies_summary: Dict = None):
        """Generate the final REPORT.md file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate time span if timestamps available
        time_span = "N/A"
        if 'timestamp' in self.data.columns and self.data['timestamp'].notna().any():
            min_time = self.data['timestamp'].min()
            max_time = self.data['timestamp'].max()
            time_span = f"{min_time} to {max_time}"
        
        report = f"""# QCI/5QI Analysis Report

**Generated:** {timestamp}  
**Dataset:** {len(self.file_info)} files, {len(self.data)} total rows  
**Time Span:** {time_span}  
**Analysis Window:** {self.window}

## Dataset Summary

### Files Processed
"""
        
        for file_info in self.file_info:
            report += f"- **{file_info['filename']}**: {file_info['rows']} rows"
            if file_info['has_qci']:
                report += ", has QCI"
            if file_info['has_5qi']:
                report += ", has 5QI"
            if file_info['has_timestamp']:
                report += ", has timestamps"
            report += "\n"
        
        report += f"\n**Total Rows:** {len(self.data)}  \n"
        report += f"**Available Columns:** {', '.join(self.data.columns)}\n\n"
        
        report += """## Methods

- **Parsing Rules:** Auto-discovery of CSV/TSV/JSON files with column normalization
- **Mapping Source:** configs/qci_5qi_map.yaml
- **Time Windows:** Data aggregated using specified window size
- **Mismatch Detection:** QCI/5QI pairs validated against mapping configuration

## Results

"""
        
        # QCI Distribution
        if 'qci_distribution' in distributions:
            qci_dist = distributions['qci_distribution']
            report += "### QCI Distribution\n\n"
            report += "| QCI | Count | Percentage |\n"
            report += "|-----|-------|------------|\n"
            
            for qci in sorted(qci_dist['counts'].keys()):
                count = qci_dist['counts'][qci]
                pct = qci_dist['percentages'][qci]
                report += f"| {qci} | {count} | {pct}% |\n"
            
            report += f"\n**Total QCI Entries:** {qci_dist['total_rows']}\n\n"
            report += "![QCI Distribution](images/qci_distribution.png)\n\n"
        
        # 5QI Distribution
        if '5qi_distribution' in distributions:
            qi5_dist = distributions['5qi_distribution']
            report += "### 5QI Distribution\n\n"
            report += "| 5QI | Count | Percentage |\n"
            report += "|-----|-------|------------|\n"
            
            for qi5 in sorted(qi5_dist['counts'].keys()):
                count = qi5_dist['counts'][qi5]
                pct = qi5_dist['percentages'][qi5]
                report += f"| {qi5} | {count} | {pct}% |\n"
            
            report += f"\n**Total 5QI Entries:** {qi5_dist['total_rows']}\n\n"
            report += "![5QI Distribution](images/5qi_distribution.png)\n\n"
        
        # Mismatch Analysis
        report += "### QCI/5QI Mapping Analysis\n\n"
        report += f"**Mismatch Rate:** {mismatches['mismatch_rate']}% ({len(mismatches['mismatches'])}/{mismatches['total_comparisons']} comparisons)\n\n"
        
        if mismatches['mismatches']:
            report += "**Top Mismatches:**\n\n"
            report += "| QCI | 5QI | Expected 5QI | IMSI | Bearer ID |\n"
            report += "|-----|-----|---------------|------|----------|\n"
            
            for mismatch in mismatches['mismatches'][:10]:
                report += f"| {mismatch['qci']} | {mismatch['5qi']} | {mismatch['expected_5qi']} | {mismatch['imsi']} | {mismatch['bearer_id']} |\n"
            
            report += f"\n*See artifacts/mismatches.csv for complete list*\n\n"
        
        # Bearer Consistency
        if anomalies['anomalies']:
            report += "### Bearer/IMSI Consistency Findings\n\n"
            report += f"**Anomalies Detected:** {len(anomalies['anomalies'])}\n\n"
            
            # Group by type
            qci_anomalies = [a for a in anomalies['anomalies'] if a['type'] == 'qci_variability']
            qi5_anomalies = [a for a in anomalies['anomalies'] if a['type'] == '5qi_variability']
            
            if qci_anomalies:
                report += f"**QCI Variability:** {len(qci_anomalies)} bearers with multiple QCI values in single time window\n"
            if qi5_anomalies:
                report += f"**5QI Variability:** {len(qi5_anomalies)} bearers with multiple 5QI values in single time window\n"
            
            report += "\n*See artifacts/anomalies_by_bearer.csv for detailed analysis*\n\n"
        
        # KPI Analysis
        if kpi_stats:
            report += "### KPI Summary\n\n"
            
            if 'qci_kpi_stats' in kpi_stats:
                report += "**KPI Statistics by QCI:**\n\n"
                for qci, stats in kpi_stats['qci_kpi_stats'].items():
                    report += f"- **QCI {qci}:**\n"
                    for kpi, values in stats.items():
                        if self.config['features']['enhanced_kpi_stats'] and 'p90' in values:
                            report += f"  - {kpi}: mean={values['mean']}, median={values['median']}, p90={values['p90']}, p99={values['p99']}, range=[{values['min']}, {values['max']}] (n={values['count']})\n"
                        else:
                            report += f"  - {kpi}: mean={values['mean']}, median={values['median']}, range=[{values['min']}, {values['max']}] (n={values['count']})\n"
                    report += "\n"
            
            if 'qi5_kpi_stats' in kpi_stats and self.config['features']['enhanced_kpi_stats']:
                report += "**KPI Statistics by 5QI:**\n\n"
                for qi5, stats in kpi_stats['5qi_kpi_stats'].items():
                    report += f"- **5QI {qi5}:**\n"
                    for kpi, values in stats.items():
                        if 'p90' in values:
                            report += f"  - {kpi}: mean={values['mean']}, median={values['median']}, p90={values['p90']}, p99={values['p99']}, range=[{values['min']}, {values['max']}] (n={values['count']})\n"
                        else:
                            report += f"  - {kpi}: mean={values['mean']}, median={values['median']}, range=[{values['min']}, {values['max']}] (n={values['count']})\n"
                    report += "\n"
        
        # Anomalies Summary Section - NEW
        if anomalies_summary and anomalies_summary['summary']:
            report += "## Anomalies Summary\n\n"
            
            # Threshold configuration used
            report += "**Analysis Thresholds:**\n\n"
            report += f"- Mismatch Rate Warning: {self.config['thresholds']['mismatch_rate_warn']:.1%}\n"
            report += f"- Mismatch Rate Critical: {self.config['thresholds']['mismatch_rate_crit']:.1%}\n"
            report += f"- KPI Outlier Z-Score: {self.config['thresholds']['kpi_outlier_zscore']}\n\n"
            
            # Top entities with anomalies
            report += f"**Top {self.config['top_n_entities']} Entities with Anomalies:**\n\n"
            report += "| Entity | Type | Severity | Window | Notes |\n"
            report += "|--------|------|----------|---------|-------|\n"
            
            for summary in anomalies_summary['summary'][:self.config['top_n_entities']]:
                report += f"| {summary['entity']} | {summary['type']} | {summary['severity']} | {summary['window']} | {summary['notes']} |\n"
            
            report += f"\n*See artifacts/anomalies_summary.csv for complete list*\n\n"
            
            # Entity scoring summary if available
            if entity_anomalies:
                summary_counts = entity_anomalies['summary']
                total_entities = sum(summary_counts.values())
                report += "**Entity-Level Scoring Summary:**\n\n"
                report += f"- Total Entities: {total_entities}\n"
                report += f"- OK: {summary_counts['ok']} ({summary_counts['ok']/total_entities:.1%})\n"
                report += f"- WARNING: {summary_counts['warn']} ({summary_counts['warn']/total_entities:.1%})\n"
                report += f"- CRITICAL: {summary_counts['crit']} ({summary_counts['crit']/total_entities:.1%})\n\n"
        
        # KPI Outliers Section - NEW
        if kpi_outliers and kpi_outliers['outliers']:
            report += "### KPI Outliers Detected\n\n"
            report += f"**Total Outliers:** {len(kpi_outliers['outliers'])}\n\n"
            
            # Group by KPI type
            kpi_outlier_counts = {}
            for outlier in kpi_outliers['outliers']:
                kpi = outlier['kpi']
                kpi_outlier_counts[kpi] = kpi_outlier_counts.get(kpi, 0) + 1
            
            for kpi, count in kpi_outlier_counts.items():
                report += f"- **{kpi.replace('_', ' ').title()}**: {count} outliers\n"
            
            report += f"\n*See artifacts/kpi_outliers.csv for detailed analysis*\n\n"
        
        # Anomalies Section
        report += "## Anomalies and Recommendations\n\n"
        
        if mismatches['mismatch_rate'] > 5:
            report += "⚠️ **High Mismatch Rate:** Consider reviewing QCI/5QI mapping configuration\n\n"
        
        if anomalies['anomalies']:
            report += "⚠️ **Bearer Inconsistencies:** Multiple QCI/5QI values detected for individual bearers\n\n"
        
        report += "### Data Quality Recommendations\n\n"
        report += "- Verify mapping configuration matches network operator policy\n"
        report += "- Investigate high mismatch rates with network engineering team\n"
        report += "- Consider longer time windows if bearer variability is expected\n"
        report += "- Validate data collection processes for consistency\n\n"
        
        # Appendix
        report += "## Appendix\n\n"
        report += f"**CLI Arguments:**\n"
        report += f"- Input Directory: {self.output_dir.parent}\n"
        report += f"- Output Directory: {self.output_dir}\n"
        report += f"- Mapping File: {self.mapping_file}\n"
        report += f"- Time Window: {self.window}\n\n"
        
        report += "**Environment:**\n"
        report += f"- Python: {sys.version.split()[0]}\n"
        report += f"- Pandas: {pd.__version__}\n"
        report += f"- Analysis Tool: QCI/5QI Pipeline v1.0\n\n"
        
        report += "**Mapping Configuration Excerpt:**\n"
        report += "```yaml\n"
        # Show first few mappings
        for i, (qci, qi5s) in enumerate(list(self.qci_to_5qi_map.items())[:5]):
            report += f'  "{qci}": {qi5s}\n'
        if len(self.qci_to_5qi_map) > 5:
            report += "  ...\n"
        report += "```\n\n"
        
        report += "**Generated Artifacts:**\n"
        artifacts_dir = self.output_dir / "artifacts"
        if artifacts_dir.exists():
            for artifact in sorted(artifacts_dir.glob("*.csv")):
                report += f"- artifacts/{artifact.name}\n"
        
        images_dir = self.output_dir / "images"
        if images_dir.exists():
            for image in sorted(images_dir.glob("*.png")):
                report += f"- images/{image.name}\n"
        
        # Add notes about configuration-dependent features
        report += f"\n**Configuration Used:**\n"
        report += f"- Config File: {getattr(self, '_config_file', 'built-in defaults')}\n"
        report += f"- Enhanced KPI Stats: {self.config['features']['enhanced_kpi_stats']}\n"
        report += f"- Entity Scoring: {self.config['features']['entity_scoring']}\n"
        report += f"- Variability Analysis: {self.config['features']['variability_analysis']}\n"
        
        # Write report
        report_path = self.output_dir / "REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Generated report: {report_path}")
    
    def run_analysis(self, input_dir: str, time_col: str = 'auto', dry_run: bool = False) -> bool:
        """Run the complete analysis pipeline."""
        try:
            self.logger.info(f"Starting QCI/5QI analysis on {input_dir}")
            
            if dry_run:
                self.logger.info("DRY RUN MODE - No output files will be created")
                return True
            
            # Discover files
            files = self.discover_files(input_dir)
            if not files:
                self.logger.error("No data files found in input directory")
                return False
            
            # Load and process data
            self.data = self.load_and_process_files(files, time_col)
            if self.data.empty:
                self.logger.error("No data could be loaded from files")
                return False
            
            # Run analyses
            self.logger.info("Computing distributions...")
            distributions = self.compute_distributions()
            
            self.logger.info("Detecting mismatches...")
            mismatches = self.detect_mismatches()
            
            self.logger.info("Analyzing bearer consistency...")
            anomalies = self.analyze_bearer_consistency()
            
            self.logger.info("Computing KPI statistics...")
            kpi_stats = self.compute_kpi_stats()
            
            # New analysis methods
            self.logger.info("Classifying entity anomalies...")
            entity_anomalies = self.classify_entity_anomalies()
            
            self.logger.info("Detecting KPI outliers...")
            kpi_outliers = self.detect_kpi_outliers()
            
            # Create anomalies summary
            self.logger.info("Creating anomalies summary...")
            anomalies_summary = self.create_anomalies_summary(entity_anomalies, anomalies, kpi_outliers)
            
            # Create visualizations
            self.logger.info("Creating charts...")
            self.create_charts(distributions)
            
            # Generate report
            self.logger.info("Generating report...")
            self.generate_report(distributions, mismatches, anomalies, kpi_stats, 
                               entity_anomalies, kpi_outliers, anomalies_summary)
            
            self.logger.info("Analysis completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            if self.verbose:
                self.logger.exception("Full traceback:")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QCI/5QI Analysis Pipeline - Analyze QoS logs for consistency and anomalies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir analysis/LOG_20250731144213
  %(prog)s --input-dir analysis/LOG_20250731144213 --window 10min --verbose
  %(prog)s --input-dir samples --mapping configs/qci_5qi_map.yaml --dry-run
        """
    )
    
    parser.add_argument(
        '--input-dir',
        default='analysis/LOG_20250731144213',
        help='Input directory containing data files (default: analysis/LOG_20250731144213)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='analysis/LOG_20250731144213/output',
        help='Output directory for results (default: analysis/LOG_20250731144213/output)'
    )
    
    parser.add_argument(
        '--mapping',
        default='configs/qci_5qi_map.yaml',
        help='QCI/5QI mapping configuration file (default: configs/qci_5qi_map.yaml)'
    )
    
    parser.add_argument(
        '--window',
        default='5min',
        help='Time-based aggregation window (default: 5min)'
    )
    
    parser.add_argument(
        '--time-col',
        default='auto',
        help='Timestamp column name or "auto" to auto-detect (default: auto)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate inputs without generating outputs'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config',
        help='Analysis configuration file (optional, default: built-in defaults)'
    )
    
    args = parser.parse_args()
    
    # Validate mapping file exists
    if not Path(args.mapping).exists():
        print(f"Error: Mapping file {args.mapping} not found")
        sys.exit(1)
    
    # Create analyzer and run
    analyzer = QCIAnalyzer(
        mapping_file=args.mapping,
        output_dir=args.output_dir,
        window=args.window,
        verbose=args.verbose,
        config_file=args.config
    )
    
    success = analyzer.run_analysis(
        input_dir=args.input_dir,
        time_col=args.time_col,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()