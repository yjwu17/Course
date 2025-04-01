export PYTHONPATH=$(find $(pwd)/src -type d | tr '\n' ':')$PYTHONPATH
python src/main/main_mhnn_input_analysis.py --config_file config/corridor_setting.ini