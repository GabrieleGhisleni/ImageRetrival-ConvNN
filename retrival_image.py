import torch
from tqdm import tqdm
from torchvision import models
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch.nn.functional as F

class RetrivalImage():
    def __init__(self,model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.numberFeatures = 512
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
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]

    def getFeatureLayer(self):
        layer=self.model._modules.get("model")._modules.get('avgpool')
        return  layer

def get_query_vector(model, query_path):
  print("Converting query_vector images to feature vectors")
  query_vector = {}
  for root, _, files in os.walk(query_path):
    for file in files:
      I = Image.open(os.path.join(query_path+"/"+file))
      if I.mode != 'L':
        vec = model.getVec(I)
        query_vector[file] = vec
        I.close()
      else:
        I.close()
    return query_vector

def get_gallery_vector(model, gallery_path):
    print("Converting gallery_vector images to feature vectors")
    gallery_vector = {}
    for root, _, files in os.walk(gallery_path):
      for file in files:
        I = Image.open(os.path.join(gallery_path+"/"+file))
        if I.mode != 'L':
          vec = model.getVec(I)
          gallery_vector[file] = vec
          I.close()
        else:
          I.close()
    return gallery_vector

def getSimilarityMatrix(query_vector, gallery_vector, K):
  ret = []
  gallery_keys= list(gallery_vector.keys())
  names= {0:'name'}
  i=1
  for igallery in (gallery_keys):
    names[i]=igallery
    i+=1
  for k in tqdm(query_vector):
    x1 = torch.Tensor(query_vector[k]).unsqueeze(0)
    tmp = {k:{}}
    l= [k]
    for q in gallery_keys:
      x2 = torch.Tensor(gallery_vector[q]).unsqueeze(0)
      distanza= F.cosine_similarity(x1, x2, dim=1, eps=1e-8)
      l.append(distanza.tolist()[0])
    df=(pd.DataFrame(l).transpose())
    df.rename(columns=names, inplace=True)
    ret.append(df)
  df = pd.concat(ret).set_index("name", drop=True)
  df_name = pd.DataFrame(index = df.index, columns = range(K))
  df_value = pd.DataFrame(index = df.index, columns = range(K))
  for j in (range(len(df))):
      kSimilar = df.iloc[j, :].sort_values(ascending = False).head(K)
      df_name.iloc[j, :] = list(kSimilar.index)
      df_value.iloc[j, :] = kSimilar.values
  return (df_name, df_value)