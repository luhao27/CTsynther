# CTsynther
Implementation of CTsynther: Contrastive Transformer Model for Single-step Retrosynthesis Prediction

## Dependency:
Follow the below steps for dependency installation:
```
conda create -y -n CTsynther tqdm
conda activate retroformer
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
conda install -y rdkit -c conda-forge
```


## Data:
Download the raw reaction dataset from [here](https://drive.google.com/drive/folders/1tpeOx2R_sUU0KhwnaLpyIy1iFtDifAGM?usp=sharing) and put it into your data directory.

## Train:
One can specify different model and training configurations in `train.sh`. Below is a sample code that calls `train.py`. Simply run `./start.sh` for training.


```
python train.py \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 100000 \
  --batch_size_token 4096 \
  --save_per_step 2500 \
  --val_per_step 2500 \
  --report_per_step 200 \
  --device cuda \
  --known_class False \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <previous_checkpoint> 
```

## Test:
One can specify different translate configurations in `translate.sh` as the sample code below. Simply sun `./translate.sh` for inference. 

To replicate our results, download the pre-trained checkpoints from [GoogleDrive](https://drive.google.com/drive/folders/1tpeOx2R_sUU0KhwnaLpyIy1iFtDifAGM?usp=sharing).


```
python translate.py \
  --batch_size_val 8 \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir <data_folder> \
  --intermediate_dir <intermediate_folder> \
  --checkpoint_dir <checkpoint_folder> \
  --checkpoint <target_checkpoint> \
  --known_class False \
  --beam_size 10 \
  --stepwise False \
```

## Reference:

If you find our work useful, please cite by:
```
@InProceedings{pmlr-v162-wan22a,
  title = {Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer},
  author = {Wan, Yue and Hsieh, Chang-Yu and Liao, Ben and Zhang, Shengyu},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  pages = {22475--22490},
  year = {2022},
  volume = {162},
  series = {Proceedings of Machine Learning Research},
  month = {17--23 Jul},
  publisher = {PMLR},
  pdf = {https://proceedings.mlr.press/v162/wan22a/wan22a.pdf},
  url = {https://proceedings.mlr.press/v162/wan22a.html}
}
```
