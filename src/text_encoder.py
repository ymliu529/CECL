import pickle
from tqdm import tqdm
from utils import read_tab_file
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, CLIPProcessor, CLIPModel
import logging
logging.basicConfig(level=logging.INFO)

def get_text_vec(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs).squeeze()
    return text_features.cpu()

def get_entity_description(imputation='name'):
    ent_dscp_path = 'src_data/WN18RR/ent_dscp.txt'
    ents, descriptions = read_tab_file(ent_dscp_path)
    ent_dscp = {}
    ent_id_path = 'data/WN18RR/ent_id'
    id_ents, ids = read_tab_file(ent_id_path)
    ent_name_path = 'src_data/WN18RR/ent_name.txt'
    name_ents, names = read_tab_file(ent_name_path)
    for ent in tqdm(id_ents):
        n_ent = ent
        try:
            if ent in ents and n_ent in name_ents:
                name = names[name_ents.index(n_ent)]
                ent_dscp[ent] = name + ': ' + descriptions[ents.index(ent)]
            elif n_ent in name_ents:
                name = names[name_ents.index(ent)]
                ent_dscp[ent] = name
            elif ent in ents:
                ent_dscp[ent] = descriptions[ents.index(ent)]
        except Exception as e:
            print(e)
    return ent_dscp

if __name__ == "__main__":
    imputation = 'name'
    ent_dscp = get_entity_description(imputation)
    model = CLIPModel.from_pretrained("path/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("path/clip-vit-large-patch14")
    ents = sorted(list(ent_dscp.keys()))
    array_filename = 'data/WN18RR/text_feature_clip.pickle'
    dscp_vec = {}
    for ent in tqdm(ents):
        dscp = ent_dscp[ent]
        if dscp == '':
            dscp_vec[ent] = np.random.normal(size=(1, 768))
        else:
            vec = get_text_vec(dscp)
            dscp_vec[ent] = vec.cpu().detach().numpy()
    dscp_array = np.array(list(dscp_vec.values()))
    with open(array_filename, 'wb') as out:
        pickle.dump(dscp_array, out)
