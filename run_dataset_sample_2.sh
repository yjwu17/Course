export PYTHONPATH=$(find $(pwd)/src -type d | tr '\n' ':')$PYTHONPATH
python src/main/main_dataset_sample_2.py