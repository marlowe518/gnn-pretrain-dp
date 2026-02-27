# GNN Pretrain + Differential Privacy Framework

This repo implements a modular framework for:

- GNN self-supervised pretraining (DGI, GraphCL, BGRL)
- Differentially Private finetuning
- PyTorch Geometric backend
- Local development + remote GPU training

## Structure

configs/        experiment configs  
src/data/       dataset loading  
src/models/     GNN encoders  
src/pretrain/   pretrain tasks  
src/finetune/   finetune tasks  
src/trainer/    trainer logic  
src/utils/      utilities  
scripts/        remote execution scripts  
run.py          main entry  

## Usage

Dry run:

python run.py --dry_run

Train:

python run.py --mode pretrain
