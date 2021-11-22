import torch,json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def activate_last_ours(model):
  for name, param in model.named_parameters():
    if  "fc" in name and 'bias' not in name:
        param.requires_grad = True
        print(f'--> Finetuning {name:s} ')
    else:
        param.requires_grad = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ImageClassifier():
    def __init__(self, batch_size, epochs, ilr=0.0001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.initial_learning_rate = ilr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, training_loader, test_loader, model, path_to_save_the_model):
      total_error = 0
      supervised_loss = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate)
      scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
      model.to(self.device)

      def train_step(data, labels):
          optimizer.zero_grad()
          data = data.to(self.device)
          labels = labels.to(self.device)
          preds = model(data)
          loss = supervised_loss(preds, labels)
          loss.backward()
          optimizer.step()
          eq = torch.eq(labels, torch.max(preds, -1).indices)
          accuracy = torch.sum(eq).float()/labels.shape[0] * 100
          return loss.cpu().detach(), accuracy.cpu()

      def test_step(data, labels):
          data = data.to(self.device)
          labels = labels.to(self.device)
          # we deactivate torch autograd
          with torch.no_grad():
              preds = model(data)
          loss = supervised_loss(preds, labels)
          return loss.cpu(), preds.cpu()

      best_accuracy = 0.0
      for e in tqdm(range(self.epochs)):
          # we activate dropout, BN params
          #activate_last_ours(model)  you can activate just the last layer decommenting this line
          # and commenting the one below
          model.train()
          ## Iteration
          for i, batch in (enumerate(training_loader)):
              data, labels = batch
              try:
                batch_loss, batch_accuracy = train_step(data, labels)
              except Exception as e:
                print(f"<----ERROR TRAIN---> \nSHAPE {data.shape}, \nLABEL {labels},\n {type(data)}\n {str(e)}")
                total_error+=1
              if i == 0:
                  print('number of model parameters {}'.format(count_parameters(model)))
          # we call scheduler to decrease LR
          scheduler.step()
          # freeze BN params, deactivate dropout
          model.eval()
          # Test the whole test dataset
          test_preds = []
          test_labels = []
          total_loss = list()
          for i, batch in enumerate(test_loader):
              data, labels = batch
              try:
                batch_loss, preds = test_step(data, labels)
              except Exception:
                print(f"<----ERROR TEST---> SHAPE {data.shape}, LABEL {labels}, {type(data)}")
                total_error+=1
              batch_preds = torch.max(preds, -1).indices
              test_preds.append(batch_preds)
              test_labels.append(labels)
              total_loss.append(batch_loss)
          test_preds = torch.cat(test_preds, dim=0).view(-1)
          test_labels = torch.cat(test_labels, dim=0).view(-1)
          assert test_preds.shape[0] == test_labels.shape[0]
          loss = sum(total_loss)/len(total_loss)
          eq = torch.eq(test_labels, test_preds).sum()
          test_accuracy = (eq/test_labels.shape[0]) * 100
          if test_accuracy >= best_accuracy:
              best_accuracy = test_accuracy
              torch.save({'epoch': e,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': loss},
              path_to_save_the_model +"/"+"resnet_18.tth") #change the path according to the model that you are training!
          print('End of Epoch {0}/{1:03} -> loss: {2:0.05}, test accuracy: {3:0.03} - best accuracy: {4:0.03}'.format(
                      e + 1, self.epochs,
                      loss.numpy(),
                      test_accuracy.numpy(),
                      best_accuracy))
          print(f"Total error captured: {total_error}")

def obtain_result(dataframe,name_path, similarity):
  res = {}
  df = {}
  for i in range(len(dataframe)):
    df[dataframe.index[i]] = list(dataframe.iloc[i].values)
  res["groupname"] = "Data Pirates"
  res["images"] = df
  nm = name_path.split(".")[0]
  with open("validation/Results/result-last-"+nm+"-"+similarity+".json", 'w') as fp:
    json.dump(res, fp, indent=4)
    print(f"Dumped into the result-last-{nm}-{similarity}.json !")
  return res

def top_k_accuracy(result):
    """
    Works only if the image are called as '<nameofcategory> (1).jpg', '<nameofcategory> (2).jpg'
    and so on.
    """
    no_matched = set()
    tot, top_1, top_3, top_5, top_10,no_match = len(result),0,0,0,0,0
    for keys in result:
      name = keys.split("(")[0].strip()
      tmp = keys
      flag = True
      for value in range(len(result[keys])):
        name_v = result[keys][value].split("(")[0].strip()
        if "_q" in name: name = name.split("_q")[0]
        if flag:
          if name == name_v and value == 1:
            top_1+=1
            top_3+=1
            top_5+=1
            top_10+=1
            flag=False
          elif name == name_v and value > 1 and value <=3:
            top_3+=1
            top_5+=1
            top_10+=1
            flag=False
          elif name == name_v and value > 3 and value <=5:
            top_5+=1
            top_10+=1
            flag=False
          elif name == name_v and value > 5:
            top_10+=1
            flag=False
          if value == 9 and flag:
            no_match+=1
            no_matched.add(tmp)
    top_1 = np.round(top_1/tot,3)
    top_3 = np.round(top_3/tot,3)
    top_10 = np.round(top_10/tot,3)
    return dict(top_1= top_1, top_3 = top_3, top_10 =top_10)

