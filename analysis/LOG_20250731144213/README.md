# LOG_20250731144213 Analysis

This directory contains analysis results for the LOG_20250731144213 dataset.

## Running Analysis

To perform QCI/5QI analysis on this dataset:

### Prerequisites

1. Install Python 3.10+ and create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Analysis

Run the default analysis:
```bash
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213
```

This will:
- Auto-discover CSV/TSV/JSON files in the input directory
- Apply QCI/5QI mapping from `configs/qci_5qi_map.yaml`
- Generate output in `analysis/LOG_20250731144213/output/`

### Advanced Options

```bash
# Custom output directory
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213 --output-dir my_output

# Different time window for aggregation
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213 --window 10min

# Custom mapping file
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213 --mapping my_mapping.yaml

# Verbose logging
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213 --verbose

# Dry run (validate without generating output)
python scripts/analyze_qos.py --input-dir analysis/LOG_20250731144213 --dry-run
```

## Output Structure

After running analysis, you'll find:

```
analysis/LOG_20250731144213/output/
├── REPORT.md              # Main analysis report
├── run.log                # Detailed execution log
├── images/                # Generated charts
│   ├── qci_distribution.png
│   ├── 5qi_distribution.png
│   └── qci_timeseries.png (if timestamps available)
└── artifacts/             # Supporting data
    ├── qci_distribution.csv
    ├── 5qi_distribution.csv
    ├── mismatches.csv
    └── anomalies_by_bearer.csv
```

## Input Data Format

The analysis pipeline expects files in the following formats:

### CSV/TSV Files
Expected columns (case-insensitive, with common synonyms):
- `timestamp/time/datetime` - ISO format or epoch time
- `qci` - QoS Class Identifier (integer)
- `5qi/five_qi/qi5` - 5G QoS Identifier (integer)
- `imsi` - International Mobile Subscriber Identity
- `bearer_id/ebi/qfi` - Bearer/Flow identifier
- Optional KPIs: `throughput/tx_bytes/rx_bytes`, `latency/rtp_jitter`, `packet_loss/loss_rate`

### JSON Files
- Array of objects with the same fields as CSV
- JSON Lines format (.jsonl) also supported

## Configuration

Edit `configs/qci_5qi_map.yaml` to customize QCI↔5QI mappings for your network operator.

## Troubleshooting

- **No files found**: Ensure input directory contains .csv, .tsv, or .json files
- **Import errors**: Run `pip install -r requirements.txt`
- **Empty results**: Check that files contain expected column names
- **Time parsing issues**: Use `--time-col` to specify exact timestamp column name