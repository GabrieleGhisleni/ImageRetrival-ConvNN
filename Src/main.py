from model import *
from solver import *
from retrival_image import *
from pprint import pprint
from typing import List


def main():
  query_path = "validation/query"
  gallery_path = "validation/gallery"
  models_path = os.listdir("Models")
  results = run_all(models_path, gallery_path, query_path)
  pprint(results)
  named_properly = True #see the description on top_k_accuracy, in case
  # the name do not correspond to that description just set it into False.
  if named_properly:
    for key in results:
      print(f"Model {key}, accuracy --> {top_k_accuracy(results[key]['images'])}")


def run_all(list_path:List[str], gallery_path, query_path):
  i=0
  model_path,res = [], {}
  for path in list_path:
    if "net" in path or "next" in path: model_path.append(os.path.join("Models/", path))
  pprint(f"Number of models ---> {len(model_path)}")
  pprint(f"{model_path}")
  print(f"""\nLen of gallery {len(os.listdir(gallery_path))}, Len of query {len(os.listdir(query_path))}\n""")
  for path in (model_path):
    print(f"Working on {path}, remaining models --> {len(model_path) - i}")
    ###########################################
    if "next101" in path:
      model = load_model(path=path, resnet="resnext101_32x8d")
    elif "wide_resnet" in path:
      model = load_model(path=path, resnet="wide_resnet")
    elif "google" in path:
      model = load_model(path=path, resnet="google")
    elif "18" in path:
      model = load_model(path = path, resnet=18)
    elif "50" in path:
      model = load_model(path=path, resnet=50)
    elif "101" in path :
       model = load_model(path=path, resnet=101)
    elif "152" in path :
      model = load_model(path=path, resnet=152)
    ############################################
    if "18" in path:
      retrival = RetrivalImage(model, resnet=18)
    elif "google" in path:
      retrival = RetrivalImage(model, resnet="google")
    elif "50" in path or "152" in path or "101" in path or "32x8d" or "wide_resnet" in path:
      retrival = RetrivalImage(model, resnet=50)
    #############################################
    try:
      query_vector = get_query_vector(retrival,query_path)
      gallery_vector = get_gallery_vector(retrival,gallery_path)
      name = path.split("/")[-1]
      images,probabilities = getSimilarityMatrix(query_vector, gallery_vector, K=10, similarity="cosine")
      result = obtain_result(images, name, similarity="cosine")
      res[name]=result
      i+=1
    except Exception as e:
      print(f"Something went wrong with model {path} {e}")
  return res

if __name__ == "__main__":
  main()
