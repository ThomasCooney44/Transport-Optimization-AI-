.PHONY: setup ingest features train simulate optimize report test lint

setup:
pip install -r requirements.txt

ingest:
python cli.py ingest

features:
python cli.py features

train:
python cli.py train

simulate:
python cli.py simulate

optimize:
python cli.py optimize

report:
python cli.py report

test:
pytest -q

lint:
python -m compileall src
