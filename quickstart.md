# Quick Start Guide

## Environment Setup

**To Prof. Wu:** A pre-configured environment is available on the server. You can activate it by running the following command:

```bash
conda activate rht_btsp
```

Manual configuration:

1. Install custom dependency packages:
    ```bash
    git clone https://github.com/RuhaoT/experiment_framework_bic.git
    cd experiment_framework_bic
    pip install .
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. [Optional] Configure Redis Server:
    - Install Redis:
        ```bash
        sudo apt-get install redis-server
        ```

    - Start Redis server:
        ```bash
        redis-server ./redis.conf
        ```

## Run BTSP-series models

### 1. Run General Fine-tuning/Testing

Select a configuration from `src/config` and run the following command:

**NOTE: This configuration is inherited from the NeuralGPR project. Many of its content could be overrided by settings(see next section)**

```bash
sh run_btsp_finetune.sh config/<configuration_file>.ini

# Example: sh run_btsp_finetune.sh config/corridor_setting.ini
# Example: sh run_btsp_finetune.sh config/forest_setting.ini
```

**To Prof. Wu:** The corridor and forest datasets are available locally at `autodl-tmp/data/`.

### 2. Custom Settings & Batch Experiment

A custom configuration can be created by coding into `src/main/main_mhnn_btsp_finetune.py`, at `__main__` function. If a setting is presented in both the configuration file and the code, the code will always override the configuration file, unless left empty in the code.

**NOTE: Selecting a unique 'experiment_name' is strongly suggested, in the current setting, an attempt to run the same experiment will overwrite the resutls of the previous one.**

The settings are defined as dataclasses, for example:

```python
@dataclasses.dataclass
class TemporalInputConfig:
    
    spike_train_sample_length: int | list[int] = 3

    # avaliable options: flatten, as_pattern, average
    temporal_dimension_handling: str | list[str] = "as_pattern"
```

Note the setting with a list type, which allows you to run multiple experiments with different settings. For example:

```python
    temporal_input_config = TemporalInputConfig(
        spike_train_sample_length=[3,4,5],
        temporal_dimension_handling=["flatten", "average"],
    )
```
This will run 6 experiments with **all combinations** of the two settings:
- spike_train_sample_length=3, temporal_dimension_handling=flatten
- spike_train_sample_length=3, temporal_dimension_handling=average
- spike_train_sample_length=4, temporal_dimension_handling=flatten
- spike_train_sample_length=4, temporal_dimension_handling=average
- spike_train_sample_length=5, temporal_dimension_handling=flatten
- spike_train_sample_length=5, temporal_dimension_handling=average

If a setting component is another setting dataclass, then the executor will recursively check all the settings and generate all combinations.

### 3. Execution Settings

#### Redis Server
For the `src/main/main_mhnn_btsp_finetune.py` file, a Redis server is used to mantain the dataset in memory to accelerate the training process.

- The server comes with a loading overhead, so it is not recommended to use with small experiment number.
- To disable using Redis, set `redis_config` to `None` in the `__main__` function.
- Make sure to flush the dataset `redis-cli flushall` before running the script in case of any changes in the previous data.

#### Distributed Execution
```python
    experiment = cuda_distributed_experiment.CudaDistributedExperiment(
        ExpMHNNBTSPFinetune(meta_params), <distribution_setting>, backend="torch"
    )
```
Where `distribution_setting` could be one of:
- `cpu`: cpu only.
- `max`: use all available GPUs.
- `list[int]`: use the specified GPUs.

### 4. Obtain Results

All results are saved in the `results` folder. Each unique experiment name will have a folder representing the results of the last run. This typically includes:

```
test    --experiment_name
 ┣ config.ini   --the configuration file used
 ┣ meta_params.json  --all meta parameters coded in the python script
 ┣ params.csv   --all parameter combinations actually passed to the model
 ┣ params_and_results.csv   all parameter combinations and their corresponding results
 ┗ results.csv   --all results

```

A special identifier 'experiment_index' is used to distinguish different runs, and `params_and_results` is the concatenation of `params` and `results` based on this index.