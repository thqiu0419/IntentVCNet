# IntentVCNet: Bridging Spatio-Temporal Gaps for Intention-Oriented Controllable Video Captioning

Runner-up Solution in "IntentVC: Intention-Oriented Controllable Video Captioning"@MM'25
## ①.Easy Start For InternVL
### 1.Environmental settings

```bash
conda create -n internvl python=3.10
conda activate internvl

cd internvl_chat
pip install -r requirements/internvl_chat.txt
```

### 2.Preparing the dataset
Download the [IntentVC dataset](https://sites.google.com/view/intentvc/dataset) and place it under ‘/data/mm2025/IntentVC'.
#### 2.1. prepare npy foramt bboxes
```bash
python generate_npy_bboxes.py
```
Running this command will generate the corresponding "object_bboxes.npy" file in the .mp4 upper folder

#### 2.2. prepare dataset info
The dataset is already configured, you only need to modify the root directory in the info.
Just modify internvl_chat/shell/data/intentvc_caption.json
```
{
  "intent_data": {
    "root": "/data/mm2025/IntentVC",
    "annotation": "shell/data/intentvc_data-text.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 4900
  }
}
```
Replace "root" with the desired path
#### 2.3. download pretrained weight
[InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)

[InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B)

### 3.Start  training
```bash
sh shell/internvl3.0/2nd_finetune/roiattention_ml.sh
```
The details of the script are listed below.
```bash
set -x

GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='work_dirs/internvl_chat_v3/tp-last5'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/finetune_w_bboxes_ml.py\
  --model_name_or_path "/data3/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "./shell/data/intentvc_caption.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --min_num_frame 32 \
  --max_num_frame 48 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 16384 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --use_llm_lora 128 \
  --deepspeed "zero_stage3_config.json" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
```
Modify batch-size, number of GPUs to match the target machine

### 4.Prediction

```
python eval_const.py
```
check "checkpoint_path" and "now_str"(save_path)


## Acknowledgments
This project is based on [InternVL](https://github.com/OpenGVLab/InternVL) and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Thanks for their awesome works.

