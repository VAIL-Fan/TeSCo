# Training Code for EgoExo-SAM2

## Getting Started

1. 进入项目文件夹，配置环境（推荐使用本项目机器适配环境 **`lm-eec-apr13`**）：
    ```bash
   conda env create -f environment_lm-eec-apr13.yml
     ```
  也可显式指定名称：`conda env create -n lm-eec-apr13 -f environment_lm-eec-apr13.yml`  
  本环境使用 PyTorch CUDA 12.1（驱动 CUDA 12.x 兼容）。
2. 进入环境
    ```bash
    conda activate lm-eec-apr13
     ```
3. 下载预训练的Checkpoints文件
    ```bash
    cd checkpoints && \
    ./download_ckpts.sh && \
    cd ..
   ```
4. 修改项目中的配置
- training/train.py文件中第274行可修改gpu使用数目
- 修改 /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第16行为下载的数据集的train文件夹的地址：
   ```yaml
  img_folder: /data/seg/Ego-Exo4D/processed_xmem/train
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第5行可修改batch_size数：
   ```yaml
  train_batch_size: 8 
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第311行修改模型参数的存储路径：
   ```yaml
  checkpoint:
    save_dir: /data/seg/sam2-main/training/checkpoints_EgoExo
   ```
- /sam2/configs/sam2.1_hiera_b+_EgoExo_finetune.yaml文件中第322行修改模型预训练参数的存储路径，也就是刚才下载好的模型参数路径：
   ```yaml
      state_dict:
        _target_: training.utils.checkpoint_utils.load_checkpoint_and_apply_kernels
        checkpoint_path: /data/seg/sam2-main/checkpoints/sam2.1_hiera_base_plus.pt # PATH to SAM 2.1 checkpoint
   ```
  
5. 运行代码（当前默认配置是 **exo->ego**，文本来自 `dam_frame_captions_v2_exo`）
   ```bash
    conda run -n lm-eec-apr13 python training/train.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_EgoExo_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 4
     ```