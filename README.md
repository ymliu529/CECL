# CECL  
Code for paper "Evolution Rather Than Degradation: Contribution-Guided Elastic Consensus Learning for Multimodal Knowledge Graph Completion", submitted to ARR.  
## Dataset Download 
For FB15K-237 and WN18RR, images associated with entities can be found from [MKGformer](https://github.com/zjunlp/MKGformer), and descriptions associated with entities can be downloaded from [KG-BERT](https://github.com/yao8839836/kg-bert). For YAGO15K, visual embeddings can be obtained from [MMKB](https://github.com/mniepert/mmkb), and linguistic embeddings are obtained via ``src/text_encoder.py``. Also, you can use the prepared descriptions in ``src_data/DATASET_NAME/`` for these datasets.
## Dataset Processing  
For triple data, the data could be preprocessed by ``src/process_datasets.py``. We extract visual and linguistic features using the pre-trained [CLIP](https://huggingface.co/) for both FB15K-237 and WN18RR. For YAGO15K, we follow previous methods, using [BERT](https://huggingface.co/) to extract linguistic embeddings. To be specific, this can be done by running ``src/text_encoder.py`` and ``src/img_encoder.py``.  
After extracting features from fixed encoders, we save the linguistic and visual features of entities in a pickle file and save the file in ``data/DATASET_NAME/``.  
## How to Run  
We have provided an example training script to train our model on WN18RR:  
```
cd src    
CUDA_VISIBLE_DEVICES=0 python learn.py --model ComplExMDR --ckpt_dir ./ckpt --dataset WN18RR --early_stopping 10 --fusion_dscp True --fusion_img True --modality_split True --img_info data/WN18RR/img_feature_clip.pickle  --dscp_info data/WN18RR/text_feature_clip.pickle --ep 1 --rank 2000
```
Note that the displayed metrics during training are more like an upper bound of modality split KGs performance, which is not the final prediction performance.  
You can run ``src/boosting_inference.py`` to ensemble the modality split predictions to get final predictions and performance, like:  
```
CUDA_VISIBLE_DEVICES=0 python boosting_inference.py --model_path YOUR_MODEL_PATH --dataset DATASET_NAME --boosting True
```
## Acknowledgments  
We would like to thank the anonymous reviewers for their insightful comments and valuable suggestions.
