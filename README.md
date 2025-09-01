# QCI/5QI Analysis Pipeline

A reproducible analysis pipeline for QCI/5QI (Quality of Service) log analysis. This tool detects inconsistencies and anomalies in mobile network QoS data and generates comprehensive reports with visualizations.

## Quickstart

### 1. Setup Environment

Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Analysis

Basic analysis on included sample data:
```bash
python scripts/analyze_qos.py --input-dir samples
```

Analyze the LOG_20250731144213 dataset:
```bash
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213
```

### 3. View Results

After analysis completes, check the output directory:
- `REPORT.md` - Main analysis report
- `images/` - Generated charts (QCI/5QI distributions, time series)
- `artifacts/` - Supporting CSV files with detailed results

## Features

### Analysis Capabilities
- **Auto-discovery**: Supports CSV, TSV, JSON, and JSON Lines formats
- **Data normalization**: Handles column name variations and synonyms
- **QCI/5QI mapping**: Configurable mapping validation
- **Mismatch detection**: Identifies inconsistent QCI↔5QI pairs
- **Anomaly detection**: Finds bearer consistency issues over time
- **KPI analysis**: Statistics for throughput, latency, packet loss (when available)
- **Time-series analysis**: Temporal patterns and trends

### Output Generation
- **Markdown reports**: Comprehensive analysis with tables and charts
- **Interactive charts**: Distribution plots and time series visualizations
- **CSV artifacts**: Detailed results for further analysis
- **Structured logging**: Detailed execution logs for troubleshooting

## CLI Reference

```bash
python scripts/analyze_qos.py [OPTIONS]

Options:
  --input-dir PATH      Input directory containing data files 
                        (default: analysis/LOG_20250731144213)
  --output-dir PATH     Output directory for results 
                        (default: analysis/LOG_20250731144213/output)
  --mapping PATH        QCI/5QI mapping configuration file 
                        (default: configs/qci_5qi_map.yaml)
  --window DURATION     Time-based aggregation window 
                        (default: 5min, examples: 10min, 1h)
  --time-col COLUMN     Timestamp column name or 'auto' to auto-detect 
                        (default: auto)
  --dry-run            Validate inputs without generating outputs
  --verbose            Enable detailed logging
  --help               Show help message
```

## Data Format

### Expected Columns
The pipeline supports flexible column naming (case-insensitive):

| Standard Name | Synonyms | Description |
|---------------|----------|-------------|
| `timestamp` | `time`, `datetime` | ISO format or epoch time |
| `qci` | - | QoS Class Identifier (4G/LTE) |
| `5qi` | `five_qi`, `qi5` | 5G QoS Identifier |
| `imsi` | - | International Mobile Subscriber Identity |
| `bearer_id` | `ebi`, `qfi` | Bearer/Flow identifier |
| `throughput` | `tx_bytes`, `rx_bytes` | Data rate (optional) |
| `latency` | `rtp_jitter` | Network delay (optional) |
| `packet_loss` | `loss_rate` | Packet loss rate (optional) |

### File Formats
- **CSV**: Comma-separated values
- **TSV**: Tab-separated values  
- **JSON**: Array of objects or single object
- **JSONL**: JSON Lines (one JSON object per line)

## Configuration

### QCI/5QI Mapping

Edit `configs/qci_5qi_map.yaml` to customize mappings for your network:

```yaml
qci_to_5qi:
  "1": [1]    # Conversational voice
  "2": [2]    # Conversational video
  "5": [5]    # IMS signaling
  # ... add your mappings

5qi_to_qci:
  "1": [1]
  "2": [2] 
  # ... reverse mappings
```

## Examples

### Basic Analysis
```bash
# Analyze with default settings
python scripts/analyze_qos.py --input-dir data/

# Use verbose logging
python scripts/analyze_qos.py --input-dir data/ --verbose

# Custom output location
python scripts/analyze_qos.py --input-dir data/ --output-dir results/
```

### Advanced Usage
```bash
# Custom time window and mapping
python scripts/analyze_qos.py \
  --input-dir data/ \
  --window 10min \
  --mapping custom_mapping.yaml

# Specify timestamp column explicitly
python scripts/analyze_qos.py \
  --input-dir data/ \
  --time-col event_timestamp

# Dry run to validate data
python scripts/analyze_qos.py --input-dir data/ --dry-run
```

## Repository Structure

```
├── scripts/
│   └── analyze_qos.py          # Main analysis pipeline
├── configs/
│   └── qci_5qi_map.yaml        # Default QCI/5QI mapping
├── analysis/
│   └── LOG_20250731144213/     # Analysis dataset directory
│       └── README.md           # Dataset-specific instructions
├── samples/
│   └── minimal.csv             # Test data for validation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Testing

Test the pipeline with included sample data:
```bash
python scripts/analyze_qos.py --input-dir samples --output-dir samples/output
```

This validates the installation and generates sample outputs you can review.

## Dependencies

- Python 3.10+
- pandas >= 2.0.0
- matplotlib >= 3.5.0
- pyyaml >= 6.0
- python-dateutil >= 2.8.0

## License

This analysis pipeline is auto-generated tooling for QoS analysis. Customize as needed for your specific requirements.