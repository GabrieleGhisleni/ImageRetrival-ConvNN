from image_augmentation_functions import *
import shutil
import random as rd
from tqdm import tqdm
import os
import numpy as np

def main():
    directory = "Dataset/train"
    test_dir = "Dataset/test"
    augment(target_dir=directory)
    check_train(target_dir=directory)
    create_test_set(ex_target_dir=directory, new_test_dir=test_dir, sample=0.2)
    check_test(ex_dir_test=test_dir)
    check_all(directory, test_dir)

def augment(target_dir:str)->None:
    """
    eg target_dir =  "drive/MyDrive/FinalDataset"
    """
    try:
        os.mkdir(target_dir)
    except:
        print("Folder already exits")
    tot_before = 0
    tot_after = 0
    for path in tqdm(os.listdir(target_dir)):
        tot_before += len(os.listdir(target_dir))
        for root, _, files in os.walk(target_dir +"/" + path + "/"):
            Folder_name = target_dir +"/" + path + "/"
            print(f"\n Working on folder: {Folder_name}")
            for file in files:
                    image_file = root + file
                    image = cv2.imread(image_file)
                    first_block(image, target_folder=Folder_name)
                    second_block(image, Target_folder=Folder_name)
                    third_block(image, Target_folder=Folder_name)
                    cv2.imwrite(Folder_name + "/original" + ".jpg", image)
            tot_after += len(os.listdir(Folder_name))
    print(f"From {tot_before} images -----> to {tot_after} images")

def check_train(target_dir:str)->None:
    total, max,min = 0, 0, 100000000000
    print(target_dir, os.listdir(target_dir))
    for path in (os.listdir(target_dir)):
        for root, _, files in os.walk(target_dir +"/" + path + "/"):
            if "check" in path:
                print(" --> WARNING THERE IS THE .CHECKPOINT FOLDER <--", path)
            print(root, f" ---> n of images: {len(files)}")
            total += len(files)
            if len(files) < min: min = len(files)
            if len(files) > max: max = len(files)
    print(f"Total image in the dataset: --> {total}")
    print(f"Min image in class: --> {min}")
    print(f"Max image in class: --> {max}")

def create_test_set(ex_target_dir:str, new_test_dir:str, sample:float=0.3)->None:
      try:
        os.mkdir(new_test_dir)
      except:
        print("folder already created")
      for path in (os.listdir(ex_target_dir)):
          for root, _, files in os.walk(ex_target_dir +"/" + path + "/"):
              try:
                  os.mkdir(new_test_dir +"/" + path)
              except:
                  pass
              l = int(len(files) * sample)
              randomlist = rd.sample(range(0, len(files)), l)
              for ifile in randomlist:
                  original = ex_target_dir +"/" + path + "/" + files[ifile]
                  target = new_test_dir +  "/"  + path + "/" + files[ifile]
                  shutil.move(original, target)

def check_test(ex_dir_test):
    for path in (os.listdir(ex_dir_test+"/")):
        for root, _, files in os.walk(ex_dir_test +"/" + path + "/"):
            print(root, f" ---> n of images: {len(files)}")

def check_all(final_dir_train:str, final_dir_test:str, compact="yes")->None:
    set_train = set()
    set_test = set()
    total_train, total_test = [], []
    for path in (os.listdir(final_dir_train)):
        for root, _, files in os.walk(final_dir_train +"/" + path + "/"):
            if compact != "yes":
                print(root, f" ---> Training number of images: {len(files)}")
            if ".ipynb" in root or ".ipynb" in path:
                try:
                    os.rmdir(root)
                    print(f"Exists path of: {os.path.exists(root)}")
                except:
                    print("mission failed")
            total_train.append(len(files))
            set_train.add(path)
    print("\n<----------------------------------------------------------------------------------------->\n")
    for path in (os.listdir(final_dir_test)):
        for root, _, files in os.walk(final_dir_test +"/" + path + "/"):
            if compact != "yes":
                print(root, f" ---> Test number of images: {len(files)}")
            if ".ipynb" in root or ".ipynb" in path:
                try:
                    os.rmdir(root)
                    print(f"Exists path of: {os.path.exists(root)}")
                except:
                    print("mission failed")
            total_test.append(len(files))
            set_test.add(path)
    print(f"Classes in test n={len(set_test)}: {set_test}")
    print(f"Classes in train n={len(set_train)}: {set_train}")
    total_train = [i for i in total_train if i != 0]  # removing the zero from -ipynb_checkpoints
    total_test = [i for i in total_test if i != 0]  # removing the zero from -ipynb_checkpoints
    print("\n<--------------------------------------- --------------------------------------->\n")
    print(
        f"TEST ---->TOTAL: {np.sum(total_test)} - Max: {np.max(total_test)} - Min: {np.min(total_test)} - Mean: {np.mean(total_test)} - SD: {np.std(total_test)}")
    print(
        f"TRAINING ----> TOTAL: {np.sum(total_train)} - Max: {np.max(total_train)} - Min: {np.min(total_train)} - Mean: {np.mean(total_train)} - SD: {np.std(total_train)}")
    print(f"TOTAL DATASET ----> {np.sum(total_train) + np.sum(total_test)}")

if __name__ == "__main__":
    main()