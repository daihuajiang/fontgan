# -*- coding: utf-8 -*-
import os
import glob
import shutil
import _pickle as pickle
from PIL import Image, ImageTk
import numpy as np

#創建obj檔案的儲存資料夾以及模型產生圖片的資料夾
def create_dir(val_obj_path):
    global remove_state
    #創建obj檔案的儲存資料夾
    if not os.path.isdir(val_obj_path):
        os.mkdir(val_obj_path)
    #創建模型產生圖片的資料夾

#將圖片資訊打包成obj檔    
def pickle_examples(paths, val_path):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    with open(val_path, 'wb') as fv:
        for p in paths:
            try:
                label = int(os.path.basename(p).split("_")[0])
                with open(p, 'rb') as f:
                    print("img %s" % p, label)
                    img_bytes = f.read()
                    example = (label, img_bytes)
                    pickle.dump(example, fv)
            except ValueError:
                continue
                    
#將要輸入給模型的圖片打包成obj檔 
def packagetoobj(package_pic_dir, package_obj_savepath):
    dirpath=package_pic_dir
    val_path = os.path.join(package_obj_savepath, "val.obj")
    pickle_examples(sorted(glob.glob(os.path.join(dirpath, "*.jpg"))), val_path=val_path)

#圖片打包成obj檔案    
def package_and_genera_pic(jpg_path):
    #空白圖片儲存路徑
    source_dir_path = "./source/"
    #文字圖片儲存資料夾
    pic_dir = "./test/"
    #obj檔案儲存資料夾
    obj_dir = './obj/'
    
    if not os.path.isdir(source_dir_path):
        os.mkdir(source_dir_path)
    
    if not os.path.isdir(pic_dir):
        os.mkdir(pic_dir)
    
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)
    
    #圖片複製到指定資料夾
    for p in glob.glob(os.path.join(source_dir_path, "*.jpg")):
        os.remove(p)
        
    for p in glob.glob(os.path.join(pic_dir, "*.jpg")):
        os.remove(p)
    
    source_filename = source_dir_path+"0000.jpg"
    shutil.copyfile(jpg_path, source_filename)
    
    #創建符合模型格式的空白圖片
    shape = (256, 512, 3) # y, x, RGB
    img1 = Image.fromarray(np.full(shape, 255).astype(np.uint8))
    img1.save(source_dir_path+"0.jpg")
    img1 = Image.open(source_dir_path+"0.jpg")
    
    #如果寬或高大於256像素，將圖片resize
    img2 = Image.open(source_filename)
    img2_width = img2.size[0]
    img2_height = img2.size[1]
    if img2_width > 256 or img2_height > 256:
        if img2_width > img2_height:
            img2 = img2.resize((256, int(img2_height*(256/img2_width))))
        else:
            img2 = img2.resize((int(img2_width*(256/img2_height)), 256))
    
    #將resize後的圖片貼到空白圖片並儲存
    img1.paste(img2, (256, 0))
    img1.save(source_filename)
    
    #將文字圖片複製到指定資料夾
    label = 0
    for i in range(12):
        pic_name = str(label) + '_' + "0000.jpg"
        shutil.copyfile(source_filename ,pic_dir + pic_name)
        label += 1
    
    #圖片打包成obj檔案並使用模型產生圖片
    #print("package pictures to obj.")
    packagetoobj(pic_dir, obj_dir)

package_and_genera_pic('C:/Users/David01/Desktop/t/0_0000.jpg')
