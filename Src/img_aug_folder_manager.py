from image_augmentation_functions import *
import shutil
import random as rd
from tqdm import tqdm
import os,time
import numpy as np



def main():
    directory = "dataset/train"
    test_dir = "dataset/test"
    tmp_transfer(new_dir=directory, old_dir= test_dir)
    augment(target_dir=directory)
    rename_all(directory)
    check_train(target_dir=test_dir)
    already_done_rename_all = False # set to true after the first run and check if all was good!
    if already_done_rename_all:
        create_test_set(ex_target_dir=directory, new_test_dir=test_dir, min_train=10)
        augment(target_dir=directory)
        check_test(ex_dir_test=test_dir)
        check_all(directory, test_dir)

def rename_all(target_dir):
    """
    Rename all the files in the folder.
    if launch multiple times create problem so be care!
    """
    i,error, names=1,0, []
    print("Renaming the files:")
    for path in tqdm(os.listdir(target_dir)):
        for root, _, files in os.walk(target_dir +"/" + path + "/"):
            for file in files:
                image_file = root + file
                index= image_file.find(".jpg")
                new_name = root+"class_"+str(path)+"_images_number_"+str(i)+".jpg"
                try:
                    os.rename(image_file, new_name)
                except Exception:
                    if "class_" not in image_file:
                        error+=1
                        names += path
                    pass
                i+=1

    print(f"total error {error}, look in classes: {names}")

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
        for root, _, files in os.walk(target_dir + "/" + path + "/"):
            i = 0
            Folder_name = target_dir + "/" + path + "/"
            print(f"\n Working on folder: {Folder_name}")
            if len(files)< 100:
                for file in files:
                    try:
                        image_file = root + file
                        image = cv2.imread(image_file)
                        if i==0:
                            first_block(image, target_folder=Folder_name)
                            i+= 1
                        elif i==1:
                            i+=1
                            second_block(image, Target_folder=Folder_name)
                        else:
                            third_block(image, Target_folder=Folder_name)
                            i=0
                        cv2.imwrite(Folder_name + "/original" + ".jpg", image)
                    except Exception as e:
                        print(f"Failure {e}\n in the class {path}, image {file}")
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

def tmp_transfer(old_dir, new_dir):
    for path in (os.listdir(new_dir)):
        for root, _, files in os.walk(new_dir + "/" + path + "/"):
            for file in files:
                original = new_dir + "/" + path + "/" + file
                target = old_dir + "/" + path + "/" + file
                try:
                    shutil.move(original, target)
                except Exception as e:
                    print(f"Original: {original}, target: {target}")
                    print(f"Error at {original, e}")

def create_test_set(ex_target_dir:str, new_test_dir:str, sample:float=None, min_train:int=10):
    """
    min_train is used to have that number in the training and all the others are moved to the test set
    sample instead is used as a float if we want to move a % of the images from trainin to test
    sample used as a int if we want to move a integer number of images from train to test
    """
    try:
        os.mkdir(new_test_dir)
    except:
        print("folder already created")
    for path in (os.listdir(ex_target_dir)):
        for root, _, files in os.walk(ex_target_dir + "/" + path + "/"):
            try:
                os.mkdir(new_test_dir + "/" + path)
            except:
                pass
            if type(sample) == int:
                randomlist = rd.sample(range(0, len(files)), sample)
            elif type(sample) == float:
                l = int(len(files) * sample)
                randomlist = rd.sample(range(0, len(files)), l)
            elif min_train != None:
                l = len(files) - 10
                randomlist = rd.sample(range(0, len(files)), l)
                print(randomlist)
            for ifile in randomlist:
                original = ex_target_dir + "/" + path + "/" + files[ifile]
                target = new_test_dir + "/" + path + "/" + files[ifile]
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
