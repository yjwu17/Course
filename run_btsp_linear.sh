export PYTHONPATH=$(find $(pwd)/src -type d | tr '\n' ':')$PYTHONPATH
python src/main/main_mhnn_btsp_linear.py --config_file config/corridor_setting.ini