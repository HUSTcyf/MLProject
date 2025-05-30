import os
import sys
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from skimage.feature import local_binary_pattern
from sklearn.metrics import silhouette_score, silhouette_samples
from pprint import pprint


def cxcy2xyxy(bbox: list, shape: tuple):
    sx, sy = shape[0], shape[1]
    cx, cy, w, h = tuple(bbox)
    x1 = round((cx - 0.5 * w) * sx)
    x2 = round((cx + 0.5 * w) * sx)
    y1 = round((cy - 0.5 * h) * sy)
    y2 = round((cy + 0.5 * h) * sy)
    return x1, y1, x2, y2


def inpaint_hist(img, mask, window_size):
    w, h, c = img.shape
    assert c == 3
    mask_org = mask.copy()
    while not mask.all():
        targetx, targety = np.where(mask == 0)
        target_xy = list(zip(targetx, targety))
        target_bg = []
        for x, y in target_xy:
            wx1, wx2 = x - window_size, x + window_size + 1
            wy1, wy2 = y - window_size, y + window_size + 1
            tx, ty = np.where(mask[wx1:wx2, wy1:wy2] > 0)
            target_bg.append(list(zip(tx, ty)))
        target_bgcount = list(map(len, target_bg))
        target_idx = target_bgcount.index(max(target_bgcount))
        fx, fy = target_xy[target_idx]
        fx1, fx2 = fx - window_size, fx + window_size + 1
        fy1, fy2 = fy - window_size, fy + window_size + 1
        img_win = img[fx1:fx2, fy1:fy2]
        pixels_bg = np.array([img_win[x, y] for x, y in target_bg[target_idx]])
        index_fg = [(x, y) for x in range(2*window_size+1) for y in range(2*window_size+1) if (x, y) not in target_bg[target_idx]]
        min_distance = sys.float_info.max
        for i in range(w - 2 * window_size):
            for j in range(h - 2 * window_size):
                # (i+window_size, j+window_size) not in target_xy
                if mask_org[i:i+2*window_size+1, j:j+2*window_size+1].all():
                    img_slide = img[i:i+2*window_size+1, j:j+2*window_size+1]
                    pixels_slide = np.array([img_slide[x, y] for x, y in target_bg[target_idx]])
                    if np.linalg.norm(pixels_bg - pixels_slide) < min_distance:
                        min_distance = np.linalg.norm(pixels_bg - pixels_slide)
                        for x, y in index_fg:
                            img[fx+x-window_size, fy+y-window_size] = img_slide[x, y]
        for x, y in index_fg:
            mask[fx+x-window_size, fy+y-window_size] = 1
            # print(img[fx+x-window_size, fy+y-window_size])
    # 直方图规定化
    # histb = cv2.calcHist([img], [0], mask, [256], [0, 255])
    # histg = cv2.calcHist([img], [1], mask, [256], [0, 255])
    # histr = cv2.calcHist([img], [2], mask, [256], [0, 255])
    # plt.plot(histb, color='b')
    # plt.plot(histg, color='g')
    # plt.plot(histr, color='r')
    # plt.savefig('output.jpg')
    # for channel in range(c):
    #     histbg = cv2.calcHist([img], [channel], mask_org, [256], [0, 255])
    #     histfg = cv2.calcHist([img], [channel], 1 - mask_org, [256], [0, 255])
    #     bg_cumsum = np.cumsum(histbg) / mask_org.sum()
    #     fg_cumsum = np.cumsum(histfg) / (1 - mask_org).sum()
    #     for g in range(256):
    #         gdiff = abs(fg_cumsum - bg_cumsum[g]).tolist()
    #         gindex = gdiff.index(min(gdiff))  # 找出累计差值最小的灰度
    #         img[:, :, channel][(img[:, :, channel] == g) & (mask_org == 0)] = gindex
    return img


def gen_yolo(img_folder, out_folder, class_index: list):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder + "origin/")
        os.makedirs(out_folder + "visual/")
        os.makedirs(out_folder + "labels/")
    for fname in os.listdir(img_folder):
        img = cv2.imread(img_folder + fname)
        img = img[14:114, 14:114, :]
        img_origin = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print("threshold:{}".format(t))
        num_obj, labels = cv2.connectedComponents(img_otsu)
        print("num objects:{}".format(num_obj))

        kernel = np.ones((3, 3))
        img_close = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        num_obj_, labels_ = cv2.connectedComponents(img_close)
        print("num objects:{}".format(num_obj_))
        while num_obj_ < num_obj:
            num_obj = num_obj_
            img_close = cv2.morphologyEx(img_close, cv2.MORPH_CLOSE, kernel)
            num_obj_, labels_ = cv2.connectedComponents(img_close)
            print("num objects:{}".format(num_obj_))

        obj_counts = [(labels_ == n).sum() for n in range(1, num_obj_)]
        obj_idx = obj_counts.index(max(obj_counts)) + 1
        ty, tx = np.nonzero(labels_ == obj_idx)
        ltx, lty = tx.min(), ty.min()  # left top
        rbx, rby = tx.max(), ty.max()  # right bottom

        w, h = img_gray.shape
        box_w, box_h = rbx-ltx, rby-lty
        cx, cy = 0.5 * (ltx + rbx), 0.5 * (lty + rby)
        bbox_yolo = cx/w, cy/h, box_w/w, box_h/h

        # 当目标长和宽小于图像大小的一半时，输出图像（先验）
        if 0.15 < bbox_yolo[2] < 0.5 and 0.15 < bbox_yolo[3] < 0.5:
            clb = class_index[int(fname[0])]
            data_list = list((clb,) + bbox_yolo)
            with open(out_folder + "labels/" + str(clb) + fname[1:-4] + ".txt", 'w') as f:
                f.write(' '.join(list(map(str, data_list))))
            cv2.rectangle(img, (ltx, lty), (rbx, rby), (0, 0, 255))
            cv2.imwrite(out_folder + "origin/" + str(clb) + fname[1:], img_origin)
            cv2.imwrite(out_folder + "visual/" + str(clb) + fname[1:-4] + "_out" + fname[-4:], img)
            

def gen_background(base_folder, out_folder, inpaint_radius, method: str):
    assert method in ['NS', 'TELEA', 'hist']
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
    os.makedirs(out_folder)
    for fname in os.listdir(base_folder + "origin/"):
        img = cv2.imread(base_folder + "origin/" + fname)
        with open(base_folder + "labels/" + fname[:-4] + ".txt", 'r') as f:
            line = f.readline()
            data_list = line.split(' ')
            cx, cy, cw, ch = tuple(map(float, data_list[1:]))
        iter_count = 5
        w, h, _ = img.shape
        mask = np.zeros((w, h))
        fgmodel = np.zeros((1, 65), dtype="float")
        bgmodel = np.zeros((1, 65), dtype="float")
        cx *= w
        cy *= h
        cw = int(round(cw * w))
        ch = int(round(ch * h))
        fx, fy = int(round(cx - 0.5 * cw)), int(round(cy - 0.5 * ch))
        mask, _, _ = cv2.grabCut(img, mask, (fx, fy, cw, ch), bgmodel, fgmodel, iter_count, cv2.GC_INIT_WITH_RECT)
        bgmask = np.where((mask == 0) | (mask == 2), 1, 0).astype(np.uint8)
        fgmask = 1 - bgmask
        print("target pixels:{}".format(fgmask.sum()))

        # img_bg = img * bgmask[:, :, np.newaxis]
        if method in ['NS', 'TELEA']:
            new_bg = cv2.inpaint(img, fgmask, inpaint_radius, eval("cv2.INPAINT_" + method))
        if method == 'hist':
            new_bg = inpaint_hist(img, bgmask, window_size=1)
        cv2.imwrite(out_folder + fname[:-4] + "_bg" + fname[-4:], new_bg)


def gen_clusters(base_folder, background_folder, noise_folder, cluster_model, model_name):
    winsize = (100, 100)
    blocksize = (20, 20)
    blockstride = (4, 4)
    cellsize = (4, 4)
    nbins = 9
    radius = 1  # LBP
    hog = cv2.HOGDescriptor(winsize, blocksize, blockstride, cellsize, nbins)
    cluster_path = os.path.join(base_folder, "clusters")
    if os.path.exists(cluster_path):
        shutil.rmtree(cluster_path)
    os.mkdir(cluster_path)
    flist = []
    nlist = []
    if noise_folder is not None:
        for noise_name in os.listdir(noise_folder):
            nlist.append(os.path.join(background_folder, *noise_name.split('-')))
    for cname in os.listdir(background_folder):
        cfolder = os.path.join(background_folder, cname)
        for fname in os.listdir(cfolder):
            append_name = os.path.join(cfolder, fname)
            if append_name not in nlist:
                flist.append(append_name)
    hog_data = np.zeros((len(flist), 99225))
    lbp_data = np.zeros((len(flist), 200))
    for n in range(len(flist)):
        fname = flist[n]
        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_array = hog.compute(img).T
        hog_data[n] = hog_array / hog_array.sum()
        lbp_array = local_binary_pattern(img_gray, 8 * radius, radius, method='ror')
        # lbp求统计直方图
        lbp_count = 0
        for i in range(int(winsize[0] / blocksize[0])):
            for j in range(int(winsize[1] / blocksize[1])):
                img_block = lbp_array[i*blocksize[0]:(i+1)*blocksize[0], j*blocksize[1]:(j+1)*blocksize[1]]
                bhist, _ = np.histogram(img_block, bins=[2 ** k - 1 for k in range(9)])
                lbp_data[n, lbp_count*8:(lbp_count+1)*8] = bhist / bhist.sum()
                lbp_count += 1
    # features = np.concatenate((hog_data, lbp_data), axis=1)
    features = hog_data
    labels = cluster_model.fit_predict(features)
    for cls in range(-1, max(labels) + 1):
        is_cls = (labels == cls)
        if cls == -1:
            print("number of noise samples: {}".format(is_cls.sum()))
        else:
            print("class {} sample number: {}".format(cls, is_cls.sum()))
        if os.path.exists(os.path.join(cluster_path, str(cls))):
            shutil.rmtree(os.path.join(cluster_path, str(cls)))
        os.makedirs(os.path.join(cluster_path, str(cls)))
        cls_names = np.array(flist)[is_cls].tolist()
        for cls_name in cls_names:
            shutil.copy(cls_name, os.path.join(cluster_path, str(cls), cls_name.split('/')[-2] + '-' + cls_name.split('/')[-1]))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    plt.scatter(pca_result[:,0], pca_result[:,1], marker='.', c=labels)
    # plt.title("{}({})".format(model_name, silhouette_score(features, labels)))
    plt.title('cluster noise')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("cluster.png")
    plt.show()


def gen_multi(base_folder, img_size = (100, 100, 3), gen_size = (2, 2), gen_num = 128):
    class_list = ['2S1', 'BRDM_2', 'BTR_60', 'D7', 'SN_132', 'SN_9563', 'SN_C71', 'T62', 'ZIL131', 'ZSU_23_4']
    generate_path = base_folder + "multi/"
    cluster_path = base_folder + "clusters/"
    if not os.path.exists(generate_path):
        os.makedirs(generate_path + "origin/")
        os.makedirs(generate_path + "visual/")
        os.makedirs(generate_path + "labels/")
    cls_numbers = list(map(int, os.listdir(cluster_path)))
    for cls_num in cls_numbers:
        if cls_num >= 0:
            bg_names = os.listdir(cluster_path + str(cls_num))
            for iter in range(gen_num):
                sample_names = random.sample(bg_names, gen_size[0] * gen_size[1])
                sample_bools = [random.getrandbits(1) for _ in range(gen_size[0] * gen_size[1])]
                synthesis_img = np.empty((gen_size[0] * img_size[0], gen_size[1] * img_size[1], img_size[2]))
                sw, sh, _ = synthesis_img.shape
                gen_name = str(cls_num) + '_' + str(iter).zfill(len(str(gen_num)))
                with open(generate_path + "labels/" + gen_name + ".txt", 'w') as mf:
                    for i in range(gen_size[0]):
                        for j in range(gen_size[1]):
                            imgidx = i * gen_size[1] + j
                            imgname, img_suffix = sample_names[imgidx][:-7], sample_names[imgidx][-4:]
                            if sample_bools[imgidx]:
                                synthesis_img[j*img_size[1]:(j+1)*img_size[1], i*img_size[0]:(i+1)*img_size[0]] = cv2.imread(base_folder + "origin/" + imgname + img_suffix)
                                with open(base_folder + "labels/" + imgname + ".txt", 'r') as sf:
                                    data = sf.readlines()[0].split(' ')
                                    bbox = list(map(float, data[1:]))
                                    bbox[0] = (bbox[0] + i) / gen_size[0]
                                    bbox[1] = (bbox[1] + j) / gen_size[1]
                                    bbox[2] /= gen_size[0]
                                    bbox[3] /= gen_size[1]
                                    data_list = [data[0]] + list(map(lambda x: round(x, 6), bbox))
                                    mf.write(' '.join(list(map(str, data_list))) + '\n')
                            else:
                                synthesis_img[j*img_size[1]:(j+1)*img_size[1], i*img_size[0]:(i+1)*img_size[0]] = cv2.imread(base_folder + "fake_background/" + sample_names[imgidx])
                cv2.imwrite(generate_path + "origin/" + gen_name + ".png", synthesis_img)
                with open(generate_path + "labels/" + gen_name + ".txt", 'r') as mf:
                    for item in mf.readlines():
                        data = item[:-1].split(' ')
                        x1, y1, x2, y2 = cxcy2xyxy(list(map(float, data[1:])), (sw, sh))
                        cv2.rectangle(synthesis_img, (x1, y1), (x2, y2), (0, 0, 255))
                        cv2.putText(synthesis_img, class_list[int(data[0])], (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 0)
                cv2.imwrite(generate_path + "visual/" + gen_name + ".png", synthesis_img)
                print(gen_name)


def main():
    class_list = ['2S1', 'BRDM_2', 'BTR_60', 'D7', 'SN_132', 'SN_9563', 'SN_C71', 'T62', 'ZIL131', 'ZSU_23_4']
    class_index_list4 = [6, 2, 5, 4]
    class_index_list6 = [3, 1, 8, 9, 0, 7]
    img_path4 = "results/fake/4/"
    img_path6 = "results/fake/6/"
    out_path = "results/"
    # left_pad, right_pad = 100, 160
    # top_pad, bottom_pad = 100, 160
    # img_with_border = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_REFLECT)
    # cv2.imshow('Image with Border', img_with_border)

    models = {
        "KMeans": KMeans(n_clusters=11, max_iter=1000),
        "DBSCAN": DBSCAN(eps=3e-3, min_samples=10),
    }

    # gen_yolo(img_path4, out_path, class_index_list4)
    # gen_yolo(img_path6, out_path, class_index_list6)
    # gen_background(out_path, out_path + "fake_background/", inpaint_radius=3, method='hist')
    gen_clusters(out_path, "/data/cyf/MYGAN/out/background", "results/noise", cluster_model=models["DBSCAN"], model_name="DBSCAN")
    gen_multi(out_path)

if __name__ == "__main__":
    main()
    