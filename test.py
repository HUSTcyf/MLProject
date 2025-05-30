import os
import argparse
import random
import shutil
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import cv2
import lpips
from PIL import Image
from trainer import Trainer
from skimage.metrics import structural_similarity
from utils import get_config, get_model_list, get_loaders, write_image, unloader


def fid(real, fake):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    command = 'python -m pytorch_fid {} {}'.format(real, fake) # command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    os.system(command)


def ssim(real, fake):
    print('Calculating SSIM...')
    reals = []
    for file in tqdm(os.listdir(real), desc='loading real data'):
        img = cv2.imread(os.path.join(real, file))
        reals.append(img)
    fakes = []
    for file in tqdm(os.listdir(fake), desc='loading fake data'):
        img = cv2.imread(os.path.join(fake, file))
        fakes.append(img)
    rt = np.stack(reals, axis=0)
    ft = np.stack(fakes, axis=0)
    result = structural_similarity(rt, ft, data_range=1.0, channel_axis=3)
    print("SSIM = {}".format(result))


def LPIPS(real, fake):
    print('Calculating LPIPS...')
    model = lpips.LPIPS(net='vgg')
    model.cuda()

    reals = []
    for file in tqdm(os.listdir(real), desc='loading real data'):
        img = lpips.im2tensor(lpips.load_image(os.path.join(real, file))) # resize (32, 32)
        reals.append(img.cuda())
    fakes = []
    for file in tqdm(os.listdir(fake), desc='loading fake data'):
        img = lpips.im2tensor(lpips.load_image(os.path.join(fake, file))) # resize (32, 32)
        fakes.append(img.cuda())
    
    count = len(reals)
    batch_size = 64
    assert len(reals) == len(fakes)
    real_tensor = torch.cat(reals, dim=0)
    fake_tensor = torch.cat(fakes, dim=0)
    if count > batch_size:
        result = 0.0
        for idx in tqdm(range(count // batch_size + 1), desc="Num Batch"):
            end = batch_size * (idx+1) if batch_size * (idx+1) < count else count
            result += model(real_tensor[batch_size*idx:end], fake_tensor[batch_size*idx:end]).sum()
        result /= count
    else:
        result = model(real_tensor, fake_tensor)
    print("LPIPS = {}".format(result.mean().item()))

    # classes = set([file.split('_')[0] for file in files])
    # res = []
    # for cls in tqdm(classes):
    #     temp = []
    #     files_cls = [file for file in files if file.startswith(cls + '_')]
    #     for i in range(0, len(files_cls) - 1, 1):
    #         for j in range(i + 1, len(files_cls), 1):
    #             img1 = data[cls][i].cuda()
    #             img2 = data[cls][j].cuda()
    #             d = model(img1, img2, normalize=True)
    #             temp.append(d.detach().cpu().numpy())
    #     res.append(np.mean(temp))
    # print(np.mean(res))


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default="")
parser.add_argument('--conf', type=str, default="")
parser.add_argument('--name', type=str, default="")
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

conf_file = os.path.join(args.name, args.conf)
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

_, test_dataloader = get_loaders(config)

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    real_dir = "results/real/" + args.dir
    fake_dir = "results/fake/" + args.dir
    if os.path.exists(real_dir):
        shutil.rmtree(real_dir)
    os.makedirs(real_dir)
    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir)

    trainer = Trainer(config)
    last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen_00100000")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()

    # out_dir = os.path.join(args.name, 'test')
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    # os.makedirs(out_dir, exist_ok=True)

    # with torch.no_grad():
    #     for it, (imgs, _) in tqdm(enumerate(test_dataloader)):
    #         imgs = imgs.cuda()
    #         fake_xs = []
    #         for i in range(config['num_generate']):
    #             fake_xs.append(trainer.generate(imgs).unsqueeze(1))
    #         fake_xs = torch.cat(fake_xs, dim=1)
    #         write_image(it, out_dir, imgs, fake_xs, format='png')

    num = 0
    if args.name.endswith('Mstar'):
        num = 11 # 10+1
    if args.name.endswith('SADD'):
        num = 1
    data = np.load(config['data_root'])
    data_for_gen = data[:num, :, :, :, :]
    data_for_fid = data[num:, :, :, :, :]
    print(data_for_gen.shape)
    print(data_for_fid.shape)
    gen_num = data.shape[1]

    for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
        os.mkdir(os.path.join(real_dir, str(cls)))
        for idx in range(data_for_fid.shape[1]):
            real_img = data_for_fid[cls, idx, :, :, :]
            real_img = Image.fromarray(np.uint8(real_img))
            real_img.save(os.path.join(real_dir, str(cls), '{}.png'.format(str(idx).zfill(len(str(gen_num))))), 'png')

    for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
        os.mkdir(os.path.join(fake_dir, str(cls)))
        for i in range(gen_num):
            idx = np.random.choice(data_for_gen.shape[1], config['n_sample_test'])
            # idx = np.array([i for _ in range(config['n_sample_test'])])
            imgs = data_for_gen[cls, idx, :, :, :]
            imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
            fake_x = trainer.generate(imgs)
            output = unloader(fake_x[0].cpu())
            output.save(os.path.join(fake_dir, str(cls), '{}.png'.format(str(i).zfill(len(str(gen_num))))), 'png')
    
    for cls in range(data_for_gen.shape[0]):
        print(f"Metrics class: {cls}")
        fid(os.path.join(real_dir, str(cls)), os.path.join(fake_dir, str(cls)))
        ssim(os.path.join(real_dir, str(cls)), os.path.join(fake_dir, str(cls)))
        LPIPS(os.path.join(real_dir, str(cls)), os.path.join(fake_dir, str(cls)))
