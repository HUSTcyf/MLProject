import random
import math
import cv2
import lpips
import numpy as np
import torch_dct as dct
from PIL import Image
from tqdm import tqdm
from torch import autograd
from torch import nn
from torchvision.utils import save_image, make_grid
from networks.blocks import *
from networks.loss import *
from utils import batched_index_select, batched_scatter
from utils import unloader


def get_wav_two(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d
    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH

class WavePool2(nn.Module):
    def __init__(self, in_channels):
        super(WavePool2, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav_two(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class GaussPool(nn.Module):
    def __init__(self, kernel_size, sigma, channels) -> None:
        super(GaussPool, self).__init__()
        self.ksize = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.gaussian = self.get_gaussian_kernel()
    
    def forward(self, x):
        g = self.gaussian(x)
        return g, g-x

    def get_gaussian_kernel(self):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(self.ksize)
        x_grid = x_coord.repeat(self.ksize).view(self.ksize, self.ksize)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (self.ksize - 1) / 2.0
        variance = self.sigma ** 2.0

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel /= torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.ksize, self.ksize)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=self.ksize, groups=self.channels, bias=False, stride=1, padding=self.ksize//2)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter


class DCTPool(nn.Module):
    def __init__(self, ratio=2) -> None:
        super(DCTPool, self).__init__()
        assert isinstance(ratio, int)
        self.ratio = ratio
    
    def forward(self, x, eps=1e-8):
        assert len(x.shape) == 4
        _, _, h, w = x.shape
        h_, w_ = h//2, w//2
        x_dct = dct.dct_2d(x)
        x_dct_low = torch.zeros(x.shape).to(x.device)
        x_dct_low[:, :, :h_, :w_] = x_dct[:, :, :h_, :w_]
        x_dct_high = x_dct - x_dct_low
        XL, XH = dct.idct_2d(x_dct_low), dct.idct_2d(x_dct_high)
        amp_dct = torch.log(x_dct ** 2 + eps) # 防止log0出现
        return amp_dct, XL, XH


class DCTGAN(nn.Module):
    def __init__(self, config):
        super(DCTGAN, self).__init__()
        self.use_dct = True
        self.gen = Generator(config['gen'])
        self.dis = Discriminator(config['dis'])

        self.w_adv_g = config['w_adv_g']
        self.w_adv_d = config['w_adv_d']
        self.w_recon = config['w_recon']
        self.w_cls = config['w_cls']
        self.w_gp = config['w_gp']
        self.n_sample = config['n_sample_train']
        self.Pool = DCTPool().cuda()
        self.WPool = WavePool2(3).cuda()
        self.L1_loss = torch.nn.L1Loss()


    def forward(self, xs, y, mode):
        if mode == 'gen_update':
            # visualize the frequency component of fake and real images
            # fake_dir = "./vggface_Frequency"
            # fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)
            # xs_index = xs[:,base_index,:,:,:]
            # for i in range(8):
            #     img = xs_index[i,:,:,:]
            #     os.makedirs(fake_dir, exist_ok=True)
            #     output = unloader(img.cpu())
            #     if os.path.exists(fake_dir):
            #         output.save(os.path.join(fake_dir, 'xs_{}_Frequency.png'.format(i)), 'png')
            
            fake_x, similarity, indices_feat, indices_ref, base_index = self.gen(xs)
            xs_index = xs[:,base_index,:,:,:]
            loss_recon = local_recon_criterion(xs, fake_x, similarity, indices_feat, indices_ref, base_index, s=8)

            if self.use_dct:
                dct_real, _, _ = self.Pool(xs_index)
                dct_fake, _, _ = self.Pool(fake_x)
                L1_loss = self.L1_loss(dct_real, dct_fake)
            else:
                LL_real, LH_real, HL_real ,HH_real = self.WPool(xs_index)
                LL_fake, LH_fake, HL_fake ,HH_fake = self.WPool(fake_x)
                L1_loss = self.L1_loss(LL_real, LL_fake) + self.L1_loss(LH_real, LH_fake) + self.L1_loss(HL_real, HL_fake) + self.L1_loss(HH_real, HH_fake)

            feat_real, _, _ = self.dis(xs)
            feat_fake, logit_adv_fake, logit_c_fake = self.dis(fake_x)
            loss_adv_gen = torch.mean(-logit_adv_fake)
            loss_cls_gen = F.cross_entropy(logit_c_fake, y.squeeze())

            loss_recon = loss_recon * self.w_recon
            loss_adv_gen = loss_adv_gen * self.w_adv_g
            loss_cls_gen = loss_cls_gen * self.w_cls

            loss_total = loss_recon + loss_adv_gen + loss_cls_gen + L1_loss
            loss_total.backward()

            return {'loss_total': loss_total,
                    'loss_recon': loss_recon,
                    'loss_adv_gen': loss_adv_gen,
                    'loss_cls_gen': loss_cls_gen}

        elif mode == 'dis_update':
            xs.requires_grad_()

            _, logit_adv_real, logit_c_real = self.dis(xs)
            loss_adv_dis_real = torch.nn.ReLU()(1.0 - logit_adv_real).mean()
            loss_adv_dis_real = loss_adv_dis_real * self.w_adv_d
            loss_adv_dis_real.backward(retain_graph=True)

            y_extend = y.repeat(1, self.n_sample).view(-1).long()
            index = torch.LongTensor(range(y_extend.size(0))).cuda()
            # logit_c_real_forgp = logit_c_real[index, y_extend].unsqueeze(1)
            # loss_reg_dis = self.calc_grad2(logit_c_real_forgp, xs)
            #
            # loss_reg_dis = loss_reg_dis * self.w_gp
            # loss_reg_dis.backward(retain_graph=True)

            loss_cls_dis = F.cross_entropy(logit_c_real, y_extend)
            loss_cls_dis = loss_cls_dis * self.w_cls
            loss_cls_dis.backward()

            with torch.no_grad():
                fake_x = self.gen(xs)[0]

            _, logit_adv_fake, _ = self.dis(fake_x.detach())
            loss_adv_dis_fake = torch.nn.ReLU()(1.0 + logit_adv_fake).mean()
            loss_adv_dis_fake = loss_adv_dis_fake * self.w_adv_d
            loss_adv_dis_fake.backward()

            loss_total = loss_adv_dis_real + loss_adv_dis_fake + loss_cls_dis
            return {'loss_total': loss_total,
                    'loss_adv_dis': loss_adv_dis_fake + loss_adv_dis_real,
                    'loss_adv_dis_real': loss_adv_dis_real,
                    'loss_adv_dis_fake': loss_adv_dis_fake,
                    'loss_cls_dis': loss_cls_dis,
                    'loss_reg': loss_cls_dis}

        else:
            assert 0, 'Not support operation'

    def generate(self, xs):
        fake_x = self.gen(xs)[0]
        return fake_x

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()
        reg /= batch_size
        return reg


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.soft_label = False
        nf = config['nf']
        n_class = config['num_classes']
        n_res_blks = config['n_res_blks']

        cnn_f = [Conv2dBlock(3, nf, 5, 1, 2,
                             pad_type='reflect',
                             norm='sn',
                             activation='none')]
        for i in range(n_res_blks):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])

        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf_out, fhid=None, activation='lrelu', norm='sn')]
        cnn_adv = [nn.AdaptiveAvgPool2d(1),
                   Conv2dBlock(nf_out, 1, 1, 1,
                               norm='none',
                               activation='none',
                               activation_first=False)]
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(nf_out, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_adv = nn.Sequential(*cnn_adv)
        self.cnn_c = nn.Sequential(*cnn_c)

    def forward(self, x):
        if len(x.size()) == 5:
            B, K, C, H, W = x.size()
            x = x.view(B * K, C, H, W)
        else:
            B, C, H, W = x.size()
            K = 1
        feat = self.cnn_f(x)
        logit_adv = self.cnn_adv(feat).view(B * K, -1)
        logit_c = self.cnn_c(feat).view(B * K, -1)
        return feat, logit_adv, logit_c


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fusion = LocalFusionModule(inplanes=128, rate=config['rate'])

    def forward(self, xs):
        b, k, C, H, W = xs.size()
        xs = xs.view(-1, C, H, W)

        querys, skips = self.encoder(xs)
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)

        c1, s1 = skips['conv1_1'], skips['skip1']
        c2, s2 = skips['conv2_1'], skips['skip2']
        c3, s3 = skips['conv3_1'], skips['skip3']
        c4, s4 = skips['conv4_1'], skips['skip4']
        
        # cg = make_grid(torch.mean(c1, dim=1), normalize=True)
        # save_image(cg, "conv1.png")
        # sg = make_grid(torch.mean(s1, dim=1), normalize=True)
        # save_image(sg, "skip1.png")
        # cg = make_grid(torch.mean(c2, dim=1), normalize=True)
        # save_image(cg, "conv2.png")
        # sg = make_grid(torch.mean(s2, dim=1), normalize=True)
        # save_image(sg, "skip2.png")
        # cg = make_grid(torch.mean(c3, dim=1), normalize=True)
        # save_image(cg, "conv3.png")
        # sg = make_grid(torch.mean(s3, dim=1), normalize=True)
        # save_image(sg, "skip3.png")
        # cg = make_grid(torch.mean(c4, dim=1), normalize=True)
        # save_image(cg, "conv4.png")
        # sg = make_grid(torch.mean(s4, dim=1), normalize=True)
        # save_image(sg, "skip4.png")

        similarity_total = torch.cat([torch.rand(b, 1) for _ in range(k)], dim=1).cuda()  # b*k
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)  # b*k
        similarity = similarity_total / similarity_sum  # b*k

        base_index = random.choice(range(k))
        base_feat = querys[:, base_index, :, :, :]

        feat_gen, indices_feat, indices_ref = self.fusion(base_feat, querys, base_index, similarity)
        fake_x = self.decoder(feat_gen, skips, base_index)

        return fake_x, similarity, indices_feat, indices_ref, base_index


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = Conv2dBlock(3, 32, 5, 1, 2,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.gauss1 = GaussPool(kernel_size=3, sigma=0.5, channels=32).cuda()
        self.dct1 = DCTPool().cuda()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv2dBlock(32, 64, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.gauss2 = GaussPool(kernel_size=3, sigma=0.5, channels=64).cuda()
        self.dct2 = DCTPool().cuda()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv2dBlock(64, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.gauss3 = GaussPool(kernel_size=3, sigma=0.5, channels=128).cuda()
        self.dct3 = DCTPool().cuda()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.gauss4 = GaussPool(kernel_size=3, sigma=0.5, channels=128).cuda()
        self.dct4 = DCTPool().cuda()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv2dBlock(128, 128, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')

    def forward(self, x):
        # (24,3,128,128)
        skips = {}
        x = self.conv1(x)
        # (24,32,128,128)
        skips['conv1_1'] = x
        _, l1, h1 = self.dct1(x)
        # x = self.pool1(x + g1)
        x = (x + l1) * 0.5
        # (24,32,128,128)
        skips['skip1'] = h1
        x = self.conv2(x)
        #（24,128,64,64）
        skips['conv2_1'] = x
        _, l2, h2 = self.dct2(x)
        # x = self.pool2(x + g2)
        x = (x + l2) * 0.5
        #（24,128,64,64）
        skips['skip2'] = h2
        x = self.conv3(x)
        # (24,128,32,32)
        skips['conv3_1'] = x
        _, l3, h3 = self.dct3(x)
        # x = self.pool3(x + g3)
        x = (x + l3) * 0.5
        # (24,128,32,32)
        skips['skip3'] = h3
        x = self.conv4(x)
        # (24,128,16,16)
        skips['conv4_1'] = x
        _, l4, h4 = self.dct4(x)
        # x = self.pool4(x + g4)
        x = (x + l4) * 0.5
        # (24,128,16,16)
        skips['skip4'] = h4
        x = self.conv5(x)
        # (24,128,8,8)
        return x, skips


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.Upsample = nn.Upsample(scale_factor=2)
        self.Conv1 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.recon_block1 = nn.Upsample(scale_factor=2).cuda()
        self.Conv2 = Conv2dBlock(128, 128, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.recon_block2 = nn.Upsample(scale_factor=2).cuda()
        self.Conv3 = Conv2dBlock(128, 64, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.recon_block3 = nn.Upsample(scale_factor=2).cuda()
        self.Conv4 = Conv2dBlock(64, 32, 3, 1, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='reflect')
        # self.recon_block4 = nn.Upsample(scale_factor=2).cuda()
        self.Conv5 = Conv2dBlock(32, 3, 5, 1, 2,
                             norm='none',
                             activation='tanh',
                             pad_type='reflect')

    def forward(self, x, skips, base_index):
        x1 = self.Upsample(x)
        x2 = self.Conv1(x1)
        d1 = skips['skip4']
        c, h, w = d1.size()[-3:]
        
        # Base_index
        if self.training:
            d1 = d1.view(8, 3, c, h, w)
            d1 = d1[:,base_index,:,:,:]
        original1 = skips['conv4_1']
        x_deconv1 = (x2 + d1) * 0.5

        x3 = self.Upsample(x_deconv1)
        x4 = self.Conv2(x3)
        d2 = skips['skip3']
        original2 = skips['conv3_1']
        c, h, w = d2.size()[-3:]
        
        # Base_index
        if self.training:
            d2 = d2.view(8, 3, c, h, w)
            d2 = d2[:, base_index, :, :, :]
        x_deconv2 = (x4 + d2) * 0.5

        x5 = self.Upsample(x_deconv2)
        x6 = self.Conv3(x5)
        d3 = skips['skip2']
        original3 = skips['conv2_1']
        c, h, w = d3.size()[-3:]
        
        # Base_index
        if self.training:
            d3 = d3.view(8, 3, c, h, w)
            d3 = d3[:, base_index, :, :, :]
        x_deconv3 = (x6 + d3) * 0.5
        
        x7 = self.Upsample(x_deconv3)
        x8 = self.Conv4(x7)
        d4 = skips['skip1']
        original4 = skips['conv1_1']
        c, h, w = d4.size()[-3:]
        
        # Base_index
        if self.training:
            d4 = d4.view(8, 3, c, h, w)
            d4 = d4[:, base_index, :, :, :]
        x_deconv4 = (x8 + d4) * 0.5

        x9 = self.Conv5(x_deconv4)
        return x9


class LocalFusionModule(nn.Module):
    def __init__(self, inplanes, rate):
        super(LocalFusionModule, self).__init__()

        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate

    def forward(self, feat, refs, index, similarity):
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        # local selection
        rate = self.rate
        num = int(rate * h * w)
        feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
                                 dim=0).cuda()  # B*num

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=feat_indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=feat_indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=feat_indices, src=feat_fused)
        feat = feat.view(b, c, h, w)  # (32*128*8*8)

        return feat, feat_indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)
