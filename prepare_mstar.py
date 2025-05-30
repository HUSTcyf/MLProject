import os
import cv2
import shutil
import numpy as np
from argparse import ArgumentParser

def gen_bgnpy(bg_root, out_root, shape: tuple):
    bg_data = []
    for cname in os.listdir(bg_root):
        cdir = os.path.join(bg_root, cname)
        for fname in os.listdir(cdir):
            img = cv2.imread(os.path.join(cdir, fname))
            img = cv2.resize(img, shape)
            bg_data.append(img)
    np.save(os.path.join(out_root, 'background.npy'), np.array(bg_data))
    print(np.array(bg_data).shape)


def gen_npy(in_root, out_root, class_list, shape: tuple):
    class_nums = []
    for clsname in class_list:
        class_data = []
        for fname in os.listdir(os.path.join(in_root, clsname)):
            img = cv2.imread(os.path.join(in_root, clsname, fname))
            img = cv2.resize(img, shape)
            class_data.append(img)
        class_nums.append(len(class_data))
        np.save(os.path.join(out_root, clsname + '.npy'), np.array(class_data)) # [np.newaxis, :]
    return class_nums


def sum_npy(root, minnum):
    bgarray = None
    if os.path.exists(os.path.join(root, "background.npy")):
        bgarray = np.load(os.path.join(root, "background.npy"))
    
    save_list = [] if bgarray is None else [bgarray[np.random.choice(bgarray.shape[0], size=minnum, replace=False)]]
    for fn in os.listdir(os.path.join(root, "train")):
        if fn.endswith(".npy"):
            data = np.load(os.path.join(root, "train", fn), allow_pickle=True)
            save_list.append(data[:minnum])
    
    if bgarray is not None: save_list.append(bgarray[np.random.choice(bgarray.shape[0], size=minnum, replace=False)]) 
    for fn in os.listdir(os.path.join(root, "test")):
        if fn.endswith(".npy"):
            data = np.load(os.path.join(root, "test", fn), allow_pickle=True)
            save_list.append(data[:minnum])
    
    sum_np = np.array(save_list)
    np.save(os.path.join(root, 'Mstar.npy'), sum_np)
    return sum_np

def main(base_root, out_root):
    class_list = ['2S1', 'ZIL131', 'BTR_60', 'SN_132', 'D7', 'ZSU_23_4', 'SN_C71', 'SN_9563', 'T62', 'BRDM_2']
    resize_shape = (128, 128)
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.mkdir(out_root)
    os.mkdir(os.path.join(out_root, "train"))
    os.mkdir(os.path.join(out_root, "test"))
    train_nums = gen_npy(os.path.join(base_root, "train"), os.path.join(out_root, "train"), class_list, resize_shape)
    test_nums = gen_npy(os.path.join(base_root, "test"), os.path.join(out_root, "test"), class_list, resize_shape)
    minnum = min(train_nums + test_nums)
    gen_bgnpy(os.path.join(base_root, "background"), out_root, resize_shape)
    
    sum_np = sum_npy(out_root, minnum)
    print(sum_np.shape)

if __name__ == '__main__':
    parser = ArgumentParser(description='Mstar')
    parser.add_argument("--input", '-i', default="./datasets/Mstar")
    parser.add_argument("--output", '-o', default="./datasets/tmp")
    args = parser.parse_args()
    main(args.input, args.output)
    os.system(f"mv {args.output}/Mstar.npy ./datasets/")
