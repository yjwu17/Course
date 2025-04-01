export PYTHONPATH=$(find $(pwd)/src -type d | tr '\n' ':')$PYTHONPATH
python src/main/main_mhnn_finetune.py --config_file $1