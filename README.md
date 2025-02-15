# CECL  
Code for paper "Evolution Rather Than Degradation: Contribution-Guided Elastic Consensus Learning for Multimodal Knowledge Graph Completion", submitted to ARR.  
## Dataset Preparation  
For FB15K-237 and WN18RR, images and texts associated with entities can be found from [MKGformer](https://github.com/zjunlp/MKGformer). For YAGO15K, visual embeddings can be obtained from [MMKB](https://github.com/mniepert/mmkb), and linguistic embeddings are obtained by running ``src/text_encoder_YAGO15K.py``.  
## Dataset Processing  
For triple data, the data could be preprocessed by ``src/process_datasets.py``. We extract visual and linguistic features using the pre-trained [CLIP](https://huggingface.co/) for both FB15K-237 and WN18RR. For YAGO15K, we follow previous methods, using [BERT]((https://huggingface.co/)) to extract linguistic embeddings. To be specific, this can be done by running ``src/text_encoder.py``,``src/img_encoder.py``, and ``src/text_encoder_YAGO15K.py``. After extracting features from fixed encoders, we save the linguistic and visual features of entities in a pickle file and save the file in ``data/DATASET_NAME/``.  
## How to run  
We provide a script to train model on WN18RR:  
``CUDA_VISIBLE_DEVICES=3 python learn.py --model ComplExMDR --ckpt_dir ./ckpt --dataset WN18RR --early_stopping 10 --fusion_dscp True --fusion_img True --modality_split True --img_info data/WN18RR/img_feature_clip.pickle  --dscp_info data/WN18RR/text_feature_clip.pickle --ep 1 --rank 2000``
