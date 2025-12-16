# Qwen2-VL & Qwen2.5-VL Finetune and Hyperparameter Optimization

This is a guide of flexible `Qwen2-VL` & `Qwen2.5-VL` model finetune and hyperparameter optimization with Optuna.

You can select to finetune with command lines or run optuna_tuning.py with multi-modes.

## Table of Contents

- [Features](#features)
- [Configuration Differences Between Qwen2-VL and Qwen2.5-VL](#configuration-differences-between-qwen2-vl-and-qwen25-vl)
- [Command line Finetune](#command-line-finetune)
- [Optuna Hyperparameter Optimization Finetune](#optuna-hyperparameter-optimization-finetune)
- [Configuration Parameters](#configuration-parameters)
- [Optuna features](#optuna-features)
- [Output and Results](#output-and-results)
- [Evaluation Metrics](#evaluation-metrics)
- [Calculation and Plotting of Evaluation Metrics During Fine-Tuning](#calculation-and-plotting-of-evaluation-metrics-during-fine-tuning)

##  Features
- **Finetune by command lines**: You can finetune `Qwen2-VL`/`Qwen2.5-VL` by command lines.
- **Optuna Hyperparameter Optimization Finetune**:
  Support three different modes:
  - **Optuna Mode**: Automatically search for optimal `learning rate` and `warmup steps`of best `bleu-4` result.
  - **Train+Eval Mode**: Finetune and evaluate with fixed parameters.
  - **Train-Only Mode**: Only perform model finetune without evaluation.
- **Calculation and Plotting of Evaluation Metrics During Fine-Tuning** : We provide scripts to plot evaluation metrics plots during finetune process(have to run after the whole finetune process).

## Configuration Differences Between Qwen2-VL and Qwen2.5-VL

The configuration is largely the same, but due to single-card performance limitations, it is recommended to reduce the `video_fps`,`per_device_train_batch_size` or `gradient_accumulation_steps` when finetune Qwen2.5-VL to avoid memory shortages.


## Command line Finetune & Eval

| Model            | Task          | Hardware Support       | Card numbers |
| ---------------- | ------------- | ---------------------- | ----------- |
| Qwen2-VL-7B-GPTQ | Video Q&A     | Intel B60 & NV         | Single Card |
| Qwen2.5-VL-7B    | Video Q&A     | Intel B60 & NV         | Single Card |
| Qwen2.5-VL-32B   | Video Summary | NV                     | Multi-Card  |

 **Note:** Models with smaller parameter sizes than those listed in the table are also compatible and can be run successfully.

### Qwen2-VL Finetune & Eval for Video Q&A

```bash
# Export requirments
export DATA='where you can find dataset_info.json'
export MODEL='where you put model file'
# To point which dataset llamafactory will use, have to add the datasets into dataset_info.json before finetune.
export dataset=activitynet_qa_2000_limit_20s
export eval_dataset=activitynet_qa_val_500_limit_20s

# Finetune
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $MODEL/Qwen2-VL-2B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $DATA \
    --dataset $dataset \
    --cutoff_len 2048 \
    --learning_rate 5e-06 \
    --num_train_epochs 20.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 125 \
    --warmup_steps 100 \
    --packing False \
    --report_to none \
    --output_dir saves/Qwen2-VL-2B-Instruct/lora/VQA \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --video_fps 0.1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 125 \
    --eval_dataset ${eval_dataset} \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

# Eval:
llamafactory-cli train \
    --stage sft \
    --model_name_or_path  $MODEL/Qwen2-VL-2B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $DATA \
    --eval_dataset $eval_dataset \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --max_new_tokens 128 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen2-VL-7B-Instruct/lora/VQA/eval \
    --do_predict True \
    --adapter_name_or_path saves/Qwen2-VL-7B-Instruct/lora/VQA \
    --video_fps 0.1 \
    --report_to none
```

### Qwen2.5-VL-7B Finetune & Eval for Video Q&A
```bash
# Export requirments
export DATA='where you can find dataset_info.json'
export MODEL='where you put model file'
# To point which dataset llamafactory will use, have to add the datasets into dataset_info.json before finetune.
export dataset=activitynet_qa_1000_limit_20s
export eval_dataset=activitynet_qa_val_250_limit_20s

# Finetune:
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path $MODEL/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $DATA \
    --dataset $dataset \
    --cutoff_len 2048 \
    --learning_rate 5e-06 \
    --num_train_epochs 10.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 125 \
    --warmup_steps 50 \
    --packing False \
    --report_to none \
    --output_dir  saves/Qwen2.5-VL-7B-Instruct/lora/VQA \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch \
    --video_fps 0.05 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 125 \
    --eval_dataset $eval_dataset \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

# Eval:
llamafactory-cli train \
    --stage sft \
    --model_name_or_path $MODEL/Qwen2.5-VL-7B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $DATA \
    --eval_dataset $eval_dataset \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate True \
    --max_new_tokens 64\
    --top_p 0.7 \
    --temperature 0.95 \
    --adapter_name_or_path saves/Qwen2.5-VL-7B-Instruct/lora/VQA \
    --output_dir saves/Qwen2.5-VL-7B-Instruct/lora/VQA/eval \
    --trust_remote_code True \
    --video_fps 0.05 \
    --bf16 True \
    --do_predict True

```

### Qwen2.5-VL-32B Finetune & Eval for Video Summary
We only support finetune Qwen2.5-VL-32B for video summary on NV multi-cards now.
Need to enable DeepSpeed stage:
```bash
# Export requirments
export DATA='where you can find dataset_info.json'
export MODEL='where you put model file'
# To point which dataset llamafactory will use, have to add the datasets into dataset_info.json before finetune.
export dataset=MLUS_Vsum_train
export eval_dataset=MLUS_Vsum_test

# Finetune with multi-card:
FORCE_TORCHRUN=1 llamafactory-cli train \
    --stage sft \
    --do_train true \
    --model_name_or_path $MODEL \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    --trust_remote_code true \
    --finetuning_type lora \
    --freeze_vision_tower true \
    --freeze_multi_modal_projector true \
    --freeze_language_model false \
    --lora_rank 8 \
    --lora_target all \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset_dir $DATA \
    --dataset $dataset \
    --template qwen2_vl \
    --cutoff_len 512 \
    --max_samples 1000 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir saves/qwen2_5vl-32b/lora/VSum \
    --logging_steps 10 \
    --save_steps 50 \
    --plot_loss true \
    --overwrite_output_dir true \
    --save_only_model false \
    --report_to none \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.07 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --video_fps 0.05 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 50 \
    --eval_dataset $eval_dataset

# Eval with single card:
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --model_name_or_path $MODEL/Qwen2.5-VL-32B-Instruct \
    --quantization_method bitsandbytes \
    --quantization_bit 4 \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    --trust_remote_code true \
    --finetuning_type lora \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir $data \
    --eval_dataset $eval_dataset \
    --cutoff_len 512 \
    --max_samples 100000 \
    --preprocessing_num_workers 16 \
    --output_dir saves/qwen2_5vl-32b/lora/VSum/eval \
    --overwrite_output_dir true \
    --do_predict true \
    --per_device_eval_batch_size 1 \
    --predict_with_generate true \
    --max_new_tokens 256 \
    --top_p 0.7 \
    --temperature 0.95 \
    --bf16 true \
    --video_fps 0.05
```

## Optuna Hyperparameter Optimization Finetune

We provide sample configs under `./qwen_vl_configs`, users can modify detail parameter vaules as needed.

### Optuna Mode

- **Qwen2-VL**

    ```bash
    python optuna_tuning.py --config_file ./qwen_vl_configs/config_optuna.json
    ```
- **Qwen2.5-VL**

    ```bash
  python optuna_tuning.py \
    --config_file ./qwen_vl_configs/config_optuna.json \
    --override \
      base.model_name=Qwen2.5-VL-7B-Instruct \
      optuna.study_name=qwen2_5vl_hyperparameter_tuning \
      training.gradient_accumulation_steps=4 \
      training.video_fps=0.05 \
      evaluation.video_fps=0.05
    ```
- **Visualization**:

  If you want to see visual results of Optuna:
  Run:

  ```sh
  sudo ufw allow 8084
  optuna-dashboard --host 0.0.0.0 --port 8084 sqlite:///./saves/optuna_tuning/optuna_study.db
  ```

  Open in the website:

  ```
  http://<serverIP>:8084/dashboard
  ```

### Train+Eval Mode

- **Qwen2-VL**

    ```bash
    python optuna_tuning.py --config_file ./qwen_vl_configs/config_finetune_eval.json
    ```
- **Qwen2.5-VL**

    ```bash
  python optuna_tuning.py \
    --config_file ./qwen_vl_configs/config_finetune_eval.json \
    --override \
      base.model_name=Qwen2.5-VL-7B-Instruct \
      training.gradient_accumulation_steps=4 \
      training.video_fps=0.05 \
      evaluation.video_fps=0.05
    ```


### Train-Only Mode

- **Qwen2-VL**

    ```bash
    python optuna_tuning.py --config_file ./qwen_vl_configs/config_finetune_only.json
    ```
- **Qwen2.5-VL**

    ```bash
  python optuna_tuning.py \
    --config_file ./qwen_vl_configs/config_finetune_only.json \
    --override \
      base.model_name=Qwen2.5-VL-7B-Instruct \
      training.gradient_accumulation_steps=4 \
      training.video_fps=0.05
    ```


### Configuration Parameters

#### Base Configuration (base)

| Parameter           | Description                                                           | Default Value                    | Type    |
| ------------------- | --------------------------------------------------------------------- | -------------------------------- | ------- |
| `working_directory` | Working directory for running commands                                | "./"                             | string  |
| `base_output_dir`   | Base directory for saving results                                     | "./saves/optuna_tuning"          | string  |
| `model_base_path`   | Base path for model files                                             | "./models"                       | string  |
| `model_name`        | Model name                                                            | "Qwen2-VL-7B-Instruct-GPTQ-Int8" | string  |
| `print_detail_log`  | Whether to print detailed finetune or evaluation logs of LLamaFactory | true                             | boolean |

#### Optuna Configuration (optuna) - Required only for optimization mode

| Parameter           | Description                                  | Default Value                   | Type    |
| ------------------- | -------------------------------------------- | ------------------------------- | ------- |
| `n_trials`          | Number of Optuna trials                      | 10                              | integer |
| `sampler_name`      | Optuna sampler name                          | "TPESampler"                    | string  |
| `study_name`        | Optuna study name                            | "qwen2vl_hyperparameter_tuning" | string  |
| `learning_rate_min` | Minimum value for learning rate search range | 1e-6                            | float   |
| `learning_rate_max` | Maximum value for learning rate search range | 1e-3                            | float   |
| `warmup_steps_min`  | Minimum value for warmup steps search range  | 50                              | integer |
| `warmup_steps_max`  | Maximum value for warmup steps search range  | 500                             | integer |


#### Training Configuration (training)

| Parameter                     | Description                                  | Default Value                   | Type    |
| ----------------------------- | -------------------------------------------- | ------------------------------- | ------- |
| `n_epochs`                    | Number of training epochs                    | 8                               | integer |
| `train_dataset`               | Training dataset name                        | "activitynet_qa_2000_limit_20s" | string  |
| `per_device_train_batch_size` | Training batch size per device               | 2                               | integer |
| `gradient_accumulation_steps` | Gradient accumulation steps                  | 8                               | integer |
| `lr_scheduler_type`           | Learning rate scheduler type                 | "cosine"                        | string  |
| `train_cutoff_len`            | Training sequence cutoff length              | 2048                            | integer |
| `max_grad_norm`               | Maximum gradient clipping                    | 1.0                             | float   |
| `logging_steps`               | Logging interval steps                       | 10                              | integer |
| `save_steps`                  | Checkpoint saving interval steps             | 500                             | integer |
| `optim`                       | Optimizer                                    | "adamw_torch"                   | string  |
| `preprocessing_num_workers`   | Number of preprocessing workers              | 16                              | integer |
| `bf16`                        | Whether to use bf16 precision                | true                            | boolean |
| `finetuning_type`             | Fine-tuning type                             | "lora"                          | string  |
| `lora_rank`                   | LoRA rank                                    | 8                               | integer |
| `lora_alpha`                  | LoRA alpha                                   | 16                              | integer |
| `lora_dropout`                | LoRA dropout                                 | 0.0                             | float   |
| `lora_target`                 | LoRA target modules                          | "all"                           | string  |
| `max_samples`                 | Maximum number of samples                    | 100000                          | integer |
| `video_fps`                   | Video frame rate                             | 0.1                             | float   |
| `learning_rate`               | Learning rate (required for non-Optuna mode) | null                            | float   |
| `warmup_steps`                | Warmup steps (required for non-Optuna mode)  | null                            | integer |

#### Evaluation Configuration (evaluation) - Optional

| Parameter                    | Description                              | Default Value                      | Type    |
| ---------------------------- | ---------------------------------------- | ---------------------------------- | ------- |
| `eval_dataset`               | Evaluation dataset name                  | "activitynet_qa_val_500_limit_20s" | string  |
| `per_device_eval_batch_size` | Evaluation batch size per device         | 1                                  | integer |
| `predict_with_generate`      | Whether to use generation for prediction | true                               | boolean |
| `eval_cutoff_len`            | Evaluation sequence cutoff length        | 1024                               | integer |
| `max_new_tokens`             | Maximum number of new generated tokens   | 128                                | integer |
| `top_p`                      | Top-p sampling parameter                 | 0.7                                | float   |
| `temperature`                | Temperature parameter                    | 0.95                               | float   |
| `quantization_method`        | Quantization method                      | "bitsandbytes"                     | string  |
| `max_samples`                | Maximum number of samples                | 100000                             | integer |
| `video_fps`                  | Video frame rate                         | 0.1                                | float   |

### Optuna features

#### Supported Parameters

- `learning_rate` and `warmup_steps`.

**Recommended configuration example:**

```json
"optuna": {
  "n_trials": 20,
  "sampler_name": "TPESampler",
  "study_name": "my_tuning_experiment",
  "learning_rate_min": 5e-7,
  "learning_rate_max": 5e-4,
  "warmup_steps_min": 100,
  "warmup_steps_max": 800
}
```

#### Supported Samplers

| Sampler Name          | Description                                     | Use Case                                                            |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------------------- |
| `TPESampler`          | Tree-structured Parzen Estimator                | Default recommended, suitable for continuous parameter optimization |
| `RandomSampler`       | Random sampling                                 | Quick exploration, baseline comparison                              |
| `CmaEsSampler`        | Covariance Matrix Adaptation Evolution Strategy | Continuous optimization problems                                    |
| `GPSampler`           | Gaussian Process sampler                        | Expensive objective function evaluation                             |
| `PartialFixedSampler` | Partial fixed parameter sampler                 | Scenarios with some fixed parameters                                |
| `NSGAIISampler`       | Non-dominated Sorting Genetic Algorithm II      | Multi-objective optimization                                        |
| `QMCSampler`          | Quasi-Monte Carlo sampler                       | Low discrepancy sequences, uniform exploration                      |


### Output and Results

#### Optuna Mode Output

- Training results: `{base_output_dir}/{study_name}/trial_XXX/`
- Optuna database: `{base_output_dir}/optuna_study.db`
- Best parameters: `{base_output_dir}/{study_name}/best_params.json`

#### Train+Eval/Train-Only Mode Output

- Training results: `{base_output_dir}/{model_name}/{timestamp}/`
- Evaluation results: `{base_output_dir}/{model_name}/{timestamp}/evaluation/`

### Evaluation Metrics

The script returns all evaluation metrics, including but not limited to:

- `predict_bleu-4`: BLEU-4 score
- `predict_rouge-1`: ROUGE-1 score
- `predict_rouge-2`: ROUGE-2 score
- `predict_rouge-l`: ROUGE-L score

## Calculation and Plotting of Evaluation Metrics During Fine-Tuning

After the finetune process over, if you want to plot eval metrics during finetune process:
Change `MODEL_NAME`,`EXPERIENT_NAME`,`EVAL_DATASET` as you need and run evaluation metrics calculation sctrpt:

```bash
export MODEL_DIR = where can find eval model
export MODEL_NAME= the model name you want to use(e.g.,Qwen2-VL-2B-Instruct)
export EXPERIENT_NAME="finetune_onlyplot_evalloss_5e-6"
export EVAL_DATASET=activitynet_qa_val_500_limit_20s
chmod a+x ./doc/run_eval.sh
./doc/run_eval.sh
```

Change `model_name` and `experiment_name` then run:

```bash
python plot_metrics.py --model_name your_model_name --experiment_name your_experiment_name
```
