import torch,json
from tqdm import tqdm
from torchvision import models
import os, cv2
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np

class RetrivalImage():
    def __init__(self,model, resnet=18):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if resnet==18: self.numberFeatures = 512
        if resnet==50: self.numberFeatures = 2048
        if resnet == "google": self.numberFeatures = 1024
        self.model=model
        self.model.eval()
        self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.pre_process=transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def getVec(self, img):
        image = self.pre_process(img).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def backup(m, i, original):
          embedding.copy_(original.data)
        h = self.featureLayer.register_forward_hook(backup)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        layer=self.model._modules.get("model")._modules.get('avgpool')
        return  layer

def from_one_to_three_channel(path):
  r = cv2.imread(path,0)
  b = cv2.imread(path,0)
  g = cv2.imread(path,0)
  img = cv2.merge((r,b,g))
  img = cv2.merge((r,b,g))
  img = Image.fromarray(img, 'RGB')
  return img

def get_query_vector(model, query_path):
  extracted = 0
  query_vector = {}
  for root, _, files in os.walk(query_path):
    for file in files:
      I = Image.open(os.path.join(query_path+"/"+file))
      if I.mode != 'L': #RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
        vec = model.getVec(I)
        query_vector[file] = vec
        extracted+=1
        I.close()
      else:
        tmp_ = os.path.join(query_path+"/"+file)
        print(f"Founded GrayScale image at {tmp_}, converted into RGB")
        I =from_one_to_three_channel(os.path.join(query_path+"/"+file))
        vec = model.getVec(I)
        query_vector[file] = vec
        extracted+=1
        I.close()
    print(f"Finded {len(files)} in the query path, extracted {extracted} images")
    return query_vector

def get_gallery_vector(model, gallery_path):
  extracted= 0
  gallery_vector = {}
  for root, _, files in os.walk(gallery_path):
    for file in files:
      I = Image.open(os.path.join(gallery_path+"/"+file))
      if I.mode != 'L': #RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
        vec = model.getVec(I)
        gallery_vector[file] = vec
        extracted+=1
        I.close()
      else:
        tmp_ = os.path.join(gallery_path+"/"+file)
        print(f"Founded GrayScale image at {tmp_}, converted into RGB")
        I =from_one_to_three_channel(os.path.join(gallery_path+"/"+file))
        vec = model.getVec(I)
        gallery_vector[file] = vec
        extracted+=1
        I.close()
  print(f"Finded {len(files)} in the gallery path, extracted {(extracted)} images")
  return gallery_vector


def getSimilarityMatrix(query_vector, gallery_vector, K, similarity):
    ret = []
    gallery_keys = list(gallery_vector.keys())
    names = {0: 'name'}
    i = 1
    # print(f"We found  {len(query_vector)} query images and {len(gallery_keys)} gallery images")
    for igallery in (gallery_keys):
        names[i] = igallery
        i += 1

    for k in (query_vector):
        x1 = torch.Tensor(query_vector[k]).unsqueeze(0)
        tmp = {k: {}}
        l = [k]
        for q in gallery_keys:
            x2 = torch.Tensor(gallery_vector[q]).unsqueeze(0)
            if similarity == "cosine":
                distanza = F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
                l.append(distanza.tolist()[0])
            elif similarity == "l1":
                distanza = distance.cityblock(x1, x2)
                l.append(distanza.tolist())
            elif similarity == "eu":
                distanza = distance.euclidean(x1, x2)
                l.append(distanza.tolist())
            elif similarity == "mk":
                distanza = distance.minkowski(x1, x2, 1)
                l.append(distanza.tolist())

        df = (pd.DataFrame(l).transpose())
        df.rename(columns=names, inplace=True)
        ret.append(df)
    df = pd.concat(ret).set_index("name", drop=True)
    df_name = pd.DataFrame(index=df.index, columns=range(K))
    df_value = pd.DataFrame(index=df.index, columns=range(K))
    for j in (range(len(df))):
        kSimilar = df.iloc[j, :].sort_values(ascending=False).head(K)
        df_name.iloc[j, :] = list(kSimilar.index)
        df_value.iloc[j, :] = kSimilar.values
    return (df_name, df_value)

