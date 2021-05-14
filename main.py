from model import ResNet, load_model
from data_loader import data_loader
from solver import ImageClassifier, obtain_result, top_k_accuracy
from retrival_image import *


if __name__ == "__main__":
  retrain = False
  if retrain:
    train_dir = "Dataset/train"
    test_dir = "Dataset/test"
    model = ResNet()
    train, test = data_loader(train_dir, test_dir)
    ic = ImageClassifier(batch_size=32, epochs=1)
    ic.train(train, test, model,path_to_save_the_model="trained_models")

  query_path = "Validation/query"
  gallery_path = "Validation/gallery"
  resnet_18_path = "trained_models/resnet_18.tth"
  try:
    model = load_model(resnet_18_path)
  except Exception:
    print(f"Load to the model not found {resnet_18_path}, did you already train the model?")
  retrival = RetrivalImage(model)
  query_vector, gallery_vector = get_query_vector(retrival,query_path), get_gallery_vector(retrival,gallery_path)
  images,probabilities = getSimilarityMatrix(query_vector, gallery_vector, K=10)

  res = obtain_result(images)
  top_k_accuracy(res)