import os
import cv2
import shutil
import numpy as np


def cxcy2xyxy(bbox: list, shape: tuple):
    sx, sy = shape[0], shape[1]
    cx, cy, w, h = tuple(bbox)
    x1 = round((cx - 0.5 * w) * sx)
    x2 = round((cx + 0.5 * w) * sx)
    y1 = round((cy - 0.5 * h) * sy)
    y2 = round((cy + 0.5 * h) * sy)
    return x1, y1, x2, y2


def get_targets(img_root, lbs_root, out_root, img_shape):
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.mkdir(out_root)
    for fname in os.listdir(img_root):
        img = cv2.imread(os.path.join(img_root, fname))
        with open(os.path.join(lbs_root, fname[:-4] + '.txt')) as ftxt:
            count = 0
            for fline in ftxt:
                fdata = fline.split()
                class_dir = os.path.join(out_root, fdata[0])
                if not os.path.exists(class_dir):
                    os.mkdir(class_dir)
                bbox = [float(item) for item in fdata[1:]]
                x1, y1, x2, y2 = cxcy2xyxy(bbox, img_shape)
                xyxy = np.array([x1, y1, x2, y2])
                x1, y1, x2, y2 = tuple(np.clip(xyxy, 0, 224).tolist())
                target_img = img[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(class_dir, fname[:-4] + '_' + str(count) + '.bmp'), target_img)
                count += 1


def get_resize(out_dir, target_dir, mw, mh):
    if os.path.exists(os.path.join(out_dir, str(mw) + '*' + str(mh))):
        shutil.rmtree(os.path.join(out_dir, str(mw) + '*' + str(mh)))
    os.mkdir(os.path.join(out_dir, str(mw) + '*' + str(mh)))

    resize_list = []
    for fname in os.listdir(target_dir):
        img = cv2.imread(os.path.join(target_dir, fname))
        w, h, _ = img.shape
        if w > 0.5 * mw and h > 0.5 * mh:
            rimg = cv2.resize(img, (mw, mh))
            cv2.imwrite(os.path.join(out_dir, str(mw) + '*' + str(mh), fname), rimg)
            resize_list.append(rimg)

    resize_npy = np.array(resize_list)[np.newaxis, :]
    resize_npy = np.tile(resize_npy, (2, 1, 1, 1, 1))
    np.save(os.path.join(out_dir, str(mw) + '*' + str(mh) + '.npy'), resize_npy)


def get_background(img_root, lbs_root, out_root, resize_shape):
    bg_folder = os.path.join(out_root, "background")
    if os.path.exists(bg_folder):
        shutil.rmtree(bg_folder)
    os.mkdir(bg_folder)
    resize_list = []
    for fname in os.listdir(img_root):
        img = cv2.imread(os.path.join(img_root, fname))
        with open(os.path.join(lbs_root, fname[:-4] + '.txt')) as ftxt:
            if len(list(ftxt)) == 0:
                cv2.imwrite(os.path.join(bg_folder, fname[:-4] + '_bg' + fname[-4:]), img)
                img_resize = cv2.resize(img, resize_shape)
                resize_list.append(img_resize)
    resize_npy = np.array(resize_list)[np.newaxis, :]
    resize_npy = np.tile(resize_npy, (2, 1, 1, 1, 1))
    np.save(os.path.join(out_root, 'background.npy'), resize_npy)
    print(resize_npy.shape)


def main():
    baseroot = '/data/cyf/RS/SAR/Plane/SADD'
    img_root = os.path.join(baseroot, 'images')
    lbs_root = os.path.join(baseroot, 'labels')
    out_dir = 'SADD'
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    # os.mkdir(out_dir)
    img_shape = (224, 224, 3)
    resize_shape = (128, 128)
    # get_targets(img_root, lbs_root, os.path.join(out_dir, "targets"), img_shape)
    # get_resize(out_dir, os.path.join(out_dir, "targets/0"), mw=128, mh=128)
    get_background(img_root, lbs_root, out_dir, resize_shape)


if __name__ == "__main__":
    main()
    