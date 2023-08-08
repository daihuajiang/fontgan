# -*- coding: utf-8 -*-
import argparse
import os
import glob
import shutil
import _pickle as pickle
import numpy as np
import torch
from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import random
import time
import math
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from model.model import chk_mkdir
import cv2


writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }


parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', default='./',
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=['cuda:0'], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=1, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=161850, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./obj/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=3)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--run_all_label', action='store_true')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')
parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')

parser.add_argument('--input_dir', type=str, default='./input/', help='picture directory.')
parser.add_argument('--save_dir', type=str, default='./output/', help='directory for saving generated images.')
parser.add_argument('--input_path', type=str, default=None, help='picture directory')

args = parser.parse_args()

def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


def main():
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_txt:
        src = args.src_txt
        font = ImageFont.truetype(args.src_font, size=args.char_size)
        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)
            )
        ).unsqueeze(dim=0) for ch in src]
        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        """
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        """
        val_dataset = DatasetFromObj(os.path.join(args.obj_path),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    with open(args.type_file, 'r', encoding='utf-8') as fp:
        fonts = [s.strip() for s in fp.readlines()]
    writer_dict = {v: k for k, v in enumerate(fonts)}

    for batch in dataloader:
        if args.run_all_label:
            # global writer_dict
            writer_dict_inv = {v: k for k, v in writer_dict.items()}
            for label_idx in range(29):
                model.set_input(torch.ones_like(batch[0]) * label_idx, batch[2], batch[1])
                model.forward()
                tensor_to_plot = torch.cat([model.fake_B, model.real_B], 3)
                # img = vutils.make_grid(tensor_to_plot)
                save_image(tensor_to_plot, os.path.join(infer_dir, "infer_{}".format(writer_dict_inv[label_idx]) + "_construct.png"))
        else:
            # model.set_input(batch[0], batch[2], batch[1])
            # model.optimize_parameters()
            model.sample(batch, infer_dir)
            global_steps += 1

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))

#尋找contour列表中每一個元素的角落座標的函式
def find_contour_corner(contours):
    left = contours[0][0][0][0]
    right = contours[0][0][0][0]
    top = contours[0][0][0][1]
    bottom = contours[0][0][0][1]
    
    for i in range(len(contours)):
        for j in range(len(contours[i])):
            if contours[i][j][0][0] < left:
                left = contours[i][j][0][0]
            if contours[i][j][0][0] > right:
                right = contours[i][j][0][0]
            if contours[i][j][0][1] < top:
                top = contours[i][j][0][1]
            if contours[i][j][0][1] > bottom:
                bottom = contours[i][j][0][1]
    
    return left, right, top, bottom

#將手寫文字大小調整並儲存的函式
def resize_fontsize(filename):
    global save_crop_path
    #讀取檔案
    img = cv2.imread(filename)
    img_width , img_height = img.shape[:2]
    #轉成灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #二值化
    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    
    #找輪廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #找出文字輪廓角落
    left, right, top, bottom = find_contour_corner(contours)
    
    #重塑文字大小
    img1 = Image.open(filename)
    img1 = img1.crop((left, top, right, bottom))
    img1 = img1.resize((130, 130))
    
    #創建符合模型格式的空白圖片
    target = Image.new('RGB', (256, 256), 'white')
    #貼上重塑大小後的文字
    target.paste(img1, (15, 50))
    #儲存圖片
    target.save(filename)

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
                    #print("img %s" % p, label)
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

#圖片打包成obj檔案並使用模型產生圖片
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
    
    #調整文字尺寸
    resize_fontsize(source_filename)

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
    print("package pictures to obj.")
    packagetoobj(pic_dir, obj_dir)

def generate(input_dir, save_dir, input_path=None):
    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if input_path:
        package_and_genera_pic(input_path)
        print('generating images...')
        with torch.no_grad():
            main()

        #將生成圖片複製到指定路徑
        count = 0
        for i in range(12):
            source_dir = './infer/' + str(i) + '/'
            for p in glob.glob(os.path.join(source_dir, "*.png")):
                style_lable = i % 2
                save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                img = Image.open(p)
                img = img.crop((0, 20, 170, 190))
                img = img.resize((256, 256))
                img.save(save_file_name)
                #shutil.copyfile(p, save_file_name)
                count += 1
    
    #拼接圖片
        print('generate gif...')
        interpolate_dir = './interpolate/'
        if not os.path.isdir(interpolate_dir):
            os.mkdir(interpolate_dir)

        concate_img_list = []
        img_count = int((count + 1) / 12)
        row = 10
        colum = int(math.ceil(img_count / row))

        if img_count < 10:
            row = img_count

        for j in range(12):
            img_list = []
            for i in range(img_count):
                img_list.append(save_dir + str(j%2) + '_' + str(j + (12 * i)) + '.png')

            #創建符合模型格式的空白圖片
            blank_img = Image.new('RGB', (256 * colum, 256 * row), 'white')
            
            #將圖片拼接
            for i in range(img_count):
                concate_img = Image.open(img_list[i])
                blank_img.paste(concate_img, (256 * int(i / 10), 256 * (i % 10)))

            blank_img.save(interpolate_dir + str(j) + '.png')

        #生成gif
        label_sequence = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 ,11]
        concate_img0 = Image.open(interpolate_dir + str(label_sequence[0]) + '.png')

        for j in range(1, 12):
            concate_img = Image.open(interpolate_dir + str(label_sequence[j]) + '.png') 
            concate_img_list.append(concate_img)
        
        concate_img0.save(interpolate_dir + 'interpolate.gif', save_all=True, append_images=concate_img_list, duration=500, loop=0)

        #將gif複製到指定資料夾
        for j in range(12):
            shutil.copyfile(interpolate_dir + str(j) + '.png', save_dir + str(j) + '.png')
        
        shutil.copyfile(interpolate_dir + 'interpolate.gif', save_dir + 'interpolate.gif')

    else:
        count = 0
        for p in glob.glob(os.path.join(input_dir, "*.jpg")):
            package_and_genera_pic(p)
            print('generating images...')
            with torch.no_grad():
                main()

            #將生成圖片複製到指定路徑
            for i in range(12):
                source_dir = './infer/' + str(i) + '/'
                for p in glob.glob(os.path.join(source_dir, "*.png")):
                    style_lable = i % 2
                    save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                    img = Image.open(p)
                    img = img.crop((0, 20, 170, 190))
                    img = img.resize((256, 256))
                    img.save(save_file_name)
                    #shutil.copyfile(p, save_file_name)
                    count += 1

        for p in glob.glob(os.path.join(input_dir, "*.png")):
            package_and_genera_pic(p)
            print('generating images...')
            with torch.no_grad():
                main()

            #將生成圖片複製到指定路徑
            for i in range(12):
                source_dir = './infer/' + str(i) + '/'
                for p in glob.glob(os.path.join(source_dir, "*.png")):
                    style_lable = i % 2
                    save_file_name = save_dir + str(style_lable) + '_' + str(count) + '.png'
                    img = Image.open(p)
                    img = img.crop((0, 20, 170, 190))
                    img = img.resize((256, 256))
                    img.save(save_file_name)
                    #shutil.copyfile(p, save_file_name)
                    count += 1
        
        #拼接圖片
        print('generate gif...')
        interpolate_dir = './interpolate/'
        if not os.path.isdir(interpolate_dir):
            os.mkdir(interpolate_dir)

        concate_img_list = []
        img_count = int((count + 1) / 12)
        row = 10
        colum = int(math.ceil(img_count / row))

        if img_count < 10:
            row = img_count

        for j in range(12):
            img_list = []
            for i in range(img_count):
                img_list.append(save_dir + str(j%2) + '_' + str(j + (12 * i)) + '.png')

            #創建符合模型格式的空白圖片
            blank_img = Image.new('RGB', (256 * colum, 256 * row), 'white')
            
            #將圖片拼接
            for i in range(img_count):
                concate_img = Image.open(img_list[i])
                blank_img.paste(concate_img, (256 * int(i / 10), 256 * (i % 10)))

            blank_img.save(interpolate_dir + str(j) + '.png')

        #生成gif
        label_sequence = [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9 ,11]
        concate_img0 = Image.open(interpolate_dir + str(label_sequence[0]) + '.png')

        for j in range(1, 12):
            concate_img = Image.open(interpolate_dir + str(label_sequence[j]) + '.png') 
            concate_img_list.append(concate_img)
        
        concate_img0.save(interpolate_dir + 'interpolate.gif', save_all=True, append_images=concate_img_list, duration=500, loop=0)

        #將gif複製到指定資料夾
        for j in range(12):
            shutil.copyfile(interpolate_dir + str(j) + '.png', save_dir + str(j) + '.png')
        
        shutil.copyfile(interpolate_dir + 'interpolate.gif', save_dir + 'interpolate.gif')

generate(args.input_dir, args.save_dir, args.input_path)