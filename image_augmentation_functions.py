################################
# --> IMAGE AUGMENTATION FUNCTIONS <--- #
import cv2
import random,os
import numpy as np
import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
from tqdm import tqdm

def sharpen_image(image, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Target_folder+"/Sharpen-"+str(r)+ Extension, image)

def emboss_image(image, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel_emboss_1=np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1)+128
    cv2.imwrite(Target_folder + "/Emboss-" +str(r)+ Extension, image)

def edge_image(image,ksize, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.Sobel(image,cv2.CV_16U,1,0,ksize=ksize)
    cv2.imwrite(Target_folder + "/Edge-"+str(ksize)+str(r) + Extension, image)

def addeptive_gaussian_noise(image, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image=cv2.merge([h,s,v])
    cv2.imwrite(Target_folder + "/Addeptive_gaussian_noise-" +str(r)+ Extension, image)

def salt_image(image,p,a, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    cv2.imwrite(Target_folder + "/Salt-"+str(p)+"*"+str(a) +str(r)+ Extension, image)

def paper_image(image,p,a, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    noisy=image
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Target_folder + "/Paper-" + str(p) + "*" + str(a) +str(r)+ Extension, image)

def salt_and_paper_image(image,p,a, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    noisy=image
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    cv2.imwrite(Target_folder + "/Salt_And_Paper-" + str(p) + "*" + str(a)+str(r) + Extension, image)

def contrast_image(image,contrast, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row in image[:,:,2]]
    image= cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Target_folder + "/Contrast-" + str(contrast) +str(r)+ Extension, image)

def edge_detect_canny_image(image,th1,th2, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.Canny(image,th1,th2)
    cv2.imwrite(Target_folder + "/Edge Canny-" + str(th1) + "*" + str(th2)+str(r) + Extension, image)

def grayscale_image(image, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Target_folder + "/Grayscale-" +str(r)+ Extension, image)

def multiply_image(image,R,G,B, Target_folder, Extension=".jpg"):
    image=image*[R,G,B]
    r = np.random.randint(10000)
    cv2.imwrite(Target_folder+"/Multiply-"+str(R)+"*"+str(G)+"*"+str(B)+str(r)+Extension, image)

def gausian_blur(image,blur, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.GaussianBlur(image,(5,5),blur)
    cv2.imwrite(Target_folder+"/GausianBLur-"+str(blur)+str(r)+Extension, image)

def averageing_blur(image,shift, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image=cv2.blur(image,(shift,shift))
    cv2.imwrite(Target_folder + "/AverageingBLur-" + str(shift) +str(r)+ Extension, image)
def median_blur(image,shift, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image=cv2.medianBlur(image,shift)
    cv2.imwrite(Target_folder + "/MedianBLur-" + str(shift) +str(r)+ Extension, image)

def bileteralBlur(image,d,color,space, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.bilateralFilter(image, d,color,space)
    cv2.imwrite(Target_folder + "/BileteralBlur-"+str(d)+"*"+str(color)+"*"+str(space)+ str(r)+Extension, image)

def erosion_image(image,shift, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift,shift),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite(Target_folder + "/Erosion-"+"*"+str(shift) + str(r)+Extension, image)

def dilation_image(image,shift ,Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite(Target_folder + "/Dilation-" + "*" + str(shift)+ str(r)+Extension, image)

def opening_image(image,shift,Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Target_folder + "/Opening-" + "*" + str(shift)+ str(r)+Extension, image)

def closing_image(image, shift, Target_folder, Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Target_folder + "/Closing-" + "*" + str(shift) +str(r)+ Extension, image)

def morphological_gradient_image(image, shift, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Target_folder + "/Morphological_Gradient-" + "*" + str(shift) + str(r)+Extension, image)

def top_hat_image(image, shift, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Target_folder + "/Top_Hat-" + "*" + str(shift) + str(r)+Extension, image)

def black_hat_image(image, shift, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Target_folder + "/Black_Hat-" + "*" + str(shift) + str(r)+Extension, image)

def resize_image(image,w,h, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image=cv2.resize(image,(w,h))
    cv2.imwrite(Target_folder+"/Resize-"+str(w)+"*"+str(h)+ str(r)+Extension, image)

def crop_image(image,y1,y2,x1,x2, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image=image[y1:y2,x1:x2]
    cv2.imwrite(Target_folder+"/Crop-"+str(x1)+str(x2)+"*"+str(y1)+str(y2)+ str(r)+Extension, image)

def padding_image(image,Target_folder,topBorder,bottomBorder,leftBorder,rightBorder,color_of_border=[0,0,0],Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.copyMakeBorder(image,topBorder,bottomBorder,leftBorder,
        rightBorder,cv2.BORDER_CONSTANT,value=color_of_border)
    cv2.imwrite(Target_folder + "/padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(rightBorder)+str(r) + Extension, image)

def flip_image(image,dir, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.flip(image, dir)
    cv2.imwrite(Target_folder + "/flip-" + str(dir)+str(r) +Extension, image)

def superpixel_image(image,segments, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    seg=segments
    def segment_colorfulness(image, mask):
        (B, G, R) = cv2.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(B, mask=mask)
        B = np.ma.masked_array(B, mask=mask)
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
        return stdRoot + (0.3 * meanRoot)
    orig = cv2.imread(image)
    vis = np.zeros(orig.shape[:2], dtype="float")
    image = io.imread(image)
    segments = slic(img_as_float(image), n_segments=segments,
                    slic_zero=True)
    for v in np.unique(segments):
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0
        C = segment_colorfulness(orig, mask)
        vis[segments == v] = C
    vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = orig.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.imwrite(Target_folder + "/superpixels-" + str(seg) +str(r) + Extension, output)

def invert_image(image,channel, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image=(channel-image)
    cv2.imwrite(Target_folder + "/invert-"+str(channel)+str(r) +Extension, image)

def add_light(image, Target_folder, gamma=1.0,Extension=".jpg"):
    r = np.random.randint(10000)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Target_folder + "/light-"+str(gamma)+str(r) +Extension, image)
    else:
        cv2.imwrite(Target_folder + "/dark-" + str(gamma) +str(r) + Extension, image)

def add_light_color(image, color, Target_folder,gamma=1.0, Extension=".jpg"):
    r = np.random.randint(10000)
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    image=cv2.LUT(image, table)
    if gamma>=1:
        cv2.imwrite(Target_folder + "/light_color-"+str(gamma)+str(r) +Extension, image)
    else:
        cv2.imwrite(Target_folder + "/dark_color" + str(gamma) +str(r) + Extension, image)

def saturation_image(image,saturation, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Target_folder + "/saturation-" + str(saturation)+str(r)  + Extension, image)

def hue_image(image,saturation, Target_folder,Extension=".jpg"):
    r = np.random.randint(10000)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Target_folder + "/hue-" + str(saturation) +str(r) + Extension, image)


def first_block(image, target_folder):
    resize_image(image, 450, 400, Target_folder=target_folder)
    crop_image(image, 0, 300, 100, 450, Target_folder=target_folder)
    padding_image(image,topBorder=100,bottomBorder=100,
                  leftBorder=100,rightBorder=100,Target_folder=target_folder)
    flip_image(image, -1, Target_folder=target_folder)
    invert_image(image, 150, Target_folder=target_folder)
    add_light_color(image, Target_folder= target_folder,color= 200, gamma=2.0,)
    hue_image(image, 100, Target_folder=target_folder)
    saturation_image(image, 200, Target_folder=target_folder)
    saturation_image(image, 150, Target_folder=target_folder)
    flip_image(image, 0, Target_folder=target_folder)
    flip_image(image, 1, Target_folder=target_folder)
    hue_image(image, 50, Target_folder=target_folder)
    add_light(image, Target_folder=target_folder, gamma=1.5 )
    add_light(image, Target_folder=target_folder, gamma=0.3)
    add_light_color(image, Target_folder= target_folder,color= 234, gamma=1.5,)
    crop_image(image, 100, 300, 100, 350, Target_folder=target_folder)
    crop_image(image, 100, 300, 100, 350, Target_folder=target_folder)
    #crop_image(image,100,300,100,350)

def second_block(image,Target_folder):
    multiply_image(image, 1.5, 1, 1, Target_folder=Target_folder)
    gausian_blur(image, 4, Target_folder=Target_folder)
    averageing_blur(image, 6, Target_folder=Target_folder)
    median_blur(image, 5, Target_folder=Target_folder)
    bileteralBlur(image, 25, 100, 100, Target_folder=Target_folder)
    erosion_image(image, 1, Target_folder=Target_folder)
    dilation_image(image, 1, Target_folder=Target_folder)
    opening_image(image, 3, Target_folder=Target_folder)
    closing_image(image, 1, Target_folder=Target_folder)
    morphological_gradient_image(image, 10, Target_folder=Target_folder)
    top_hat_image(image, 200, Target_folder=Target_folder)
    black_hat_image(image, 300, Target_folder=Target_folder)
    gausian_blur(image, 0.25, Target_folder=Target_folder)
    gausian_blur(image, 0.50, Target_folder=Target_folder)
    opening_image(image, 5, Target_folder=Target_folder)
    dilation_image(image, 3, Target_folder=Target_folder)
    erosion_image(image, 3, Target_folder=Target_folder)
    bileteralBlur(image, 40, 75, 75, Target_folder=Target_folder)
    median_blur(image, 7, Target_folder=Target_folder)
    median_blur(image, 3, Target_folder=Target_folder)
    averageing_blur(image, 4, Target_folder=Target_folder)
    multiply_image(image, 1.25, 1.25, 1.25, Target_folder=Target_folder)
    closing_image(image, 3, Target_folder=Target_folder)
    morphological_gradient_image(image, 15, Target_folder=Target_folder)
    top_hat_image(image, 300, Target_folder=Target_folder)
    #black_hat_image(image, 500)
def third_block(image,Target_folder):
    sharpen_image(image, Target_folder=Target_folder)
    emboss_image(image, Target_folder=Target_folder)
    edge_image(image, 1, Target_folder=Target_folder)
    addeptive_gaussian_noise(image, Target_folder=Target_folder)
    salt_image(image, 0.5, 0.9, Target_folder=Target_folder)
    paper_image(image, 0.5, 0.9, Target_folder=Target_folder)
    salt_and_paper_image(image, 0.5, 0.9, Target_folder=Target_folder)
    edge_detect_canny_image(image, 100, 200, Target_folder=Target_folder)
    salt_image(image, 0.5, 0.09, Target_folder=Target_folder)
    paper_image(image, 0.5, 0.009, Target_folder=Target_folder)
    paper_image(image, 0.5, 0.09, Target_folder=Target_folder)
    salt_and_paper_image(image, 0.5, 0.009, Target_folder=Target_folder)
    salt_and_paper_image(image, 0.5, 0.09, Target_folder=Target_folder)
    edge_image(image, 3, Target_folder=Target_folder)
    edge_detect_canny_image(image, 200, 400, Target_folder=Target_folder)
    salt_image(image, 0.5, 0.009, Target_folder=Target_folder)
