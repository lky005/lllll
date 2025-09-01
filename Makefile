.PHONY: install test run-samples

install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest -q

run-samples:
	python scripts/analyze_qos.py --input-dir samples --output-dir samples/output --mapping configs/qci_5qi_map.yaml --window 5min --verbose