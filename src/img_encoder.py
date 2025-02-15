from PIL import Image
import pickle
import os
from tqdm import tqdm
import numpy as np
from transformers import CLIPModel, CLIPProcessor

class ImageEncoder():
    @staticmethod
    def get_embedding(self, filter_gate=True):
        pass
    def extract_feature(self, base_path, filter_gate=True):
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        print("start extract")
        dict = {}
        ents = os.listdir(base_path)
        pbar = tqdm(total=len(ents))
        while len(ents) > 0:
            ents_50 = []
            ents_50_ok = []
            for i in range(1):
                if len(ents) > 0:
                    ent = ents.pop()
                    try:
                        img_path = base_path + '/' + ent
                        img_name = os.listdir(img_path)
                        for item in img_name:
                            ents_50.append(img_path + '/' + item)
                        if len(img_name) > 30:
                            ents_50 = ents_50[:30]
                        elif len(img_name) < 30:
                            padding_count = 30 - len(ents_50)
                            padded_list = ents_50 + [ents_50[i % len(ents_50)] for i in range(padding_count)]
                            ents_50 = padded_list
                    except Exception as e:
                        print(e)
                        continue
            images = []
            for imgpath in ents_50:
                img = Image.open(imgpath).convert('RGB').resize((224, 224))
                images.append(img)
            img_tensor = self.processor(images=images, return_tensors="pt", padding=True)['pixel_values']
            img_tensor = img_tensor
            entityname = ents_50[0].split("/")[9]
            # FB15k-237
            # entityname = entityname.replace('.', '/')
            # entityname = "/" + entityname
            entityname = entityname[1:]
            ents_50_ok.append(entityname)
            result = self.model.get_image_features(img_tensor)
            result_npy = result.data.cpu().numpy()
            padded_tensor = np.zeros((30, 768))
            padded_tensor[:result_npy.shape[0], :] = result_npy
            for i in range(len(ents_50_ok)):
                dict[ents_50_ok[i]] = padded_tensor
            pbar.update(1)
        pbar.close()
        return dict

class VisionTransformer(ImageEncoder):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.model = CLIPModel.from_pretrained("path/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("path/clip-vit-large-patch14")

    def get_embedding(self, base_path, filter_gate=True):
        self.d = self.extract_feature(base_path, filter_gate=filter_gate)
        return self.d

if __name__ == "__main__":
    base_path = 'path/wn18-images'
    m = VisionTransformer()
    img_vec = m.get_embedding(base_path, filter_gate=True)
    f = open('data/WN18RR/ent_id', 'r')
    Lines = f.readlines()
    id2ent = {}
    img_array = []
    for l in Lines:
        ent, id = l.strip().split()
        id2ent[id] = ent
        if ent in img_vec.keys():
            print(id, ent)
            img_array.append(img_vec[ent])
            print("success")
        else:
            img_array.append(np.random.normal(size=(30,768,)))
            print("fail")
    output_file = 'data/WN18RR/img_feature_clip.pickle'
    img_array = np.array(img_array)
    with open(output_file, 'wb') as out:
        pickle.dump(img_array, out)