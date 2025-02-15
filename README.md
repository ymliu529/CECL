# CECL  
Code for paper "Evolution Rather Than Degradation: Contribution-Guided Elastic Consensus Learning for Multimodal Knowledge Graph Completion", submitted to ARR.  
## Dataset Preparation  
For FB15K-237 and WN18RR, images and texts associated with entities can be found from [MKGformer](https://github.com/zjunlp/MKGformer). For YAGO15K, visual embeddings can be obtained from [MMKB](https://github.com/mniepert/mmkb), and linguistic embeddings are obtained by running ``src/text_encoder_YAGO15K.py``.  
## Dataset Processing  
For triple data, the data could be preprocessed by ``src/process_datasets.py``. We extract visual and linguistic features using the pre-trained [CLIP](https://huggingface.co/) for both FB15K-237 and WN18RR. For YAGO15K, we follow previous methods, using [BERT]((https://huggingface.co/)) to extract linguistic embeddings.
