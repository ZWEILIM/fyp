
import os
import json
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class ChicagoFSWild(Dataset):

    def __init__(self, split, img_dir, fcsv, vocab_map, transform,
                 img_size=224, map_size=14, lambda_x=None, scale_x=None):

        assert split == 'test', 'Unknown split: %s'.format(split)
        self.split = split
        self.img_dir = img_dir
        self.fcsv = fcsv
        self.vocab_map = vocab_map
        self.transform = transform
        self.img_size = img_size
        self.map_size = map_size
        with open(lambda_x, 'r') as f:
            self.lambda_x = json.load(f)
        assert scale_x in ['1', '2', '3', '4'], 'Invalid value for `scale_x` parameter: %d' % scale_x
        self.scale_x = scale_x

        self._parse()

    def _parse(self):
        with open(self.fcsv, 'r') as fo:
            lns = fo.readlines()
        print('%d %s samples' % (len(lns), self.split))
        self.imdirs, self.labels, self.n_frames = [], [], []
        for i in range(len(lns)):
            imdir, label, nframes = lns[i].strip().split(',')
            self.imdirs.append(imdir)
            self.labels.append(label)
            self.n_frames.append(int(nframes))

    def __len__(self):
        return len(self.imdirs)

    def __getitem__(self, idx):
        subdir = self.imdirs[idx]
        label = list(map(lambda x: self.vocab_map[x], self.labels[idx]))
        fnames = [str(i).zfill(4) + '.jpg' for i in range(1, self.n_frames[idx]+1)]
        
        pad = self.lambda_x[subdir]['pad']
        l_pad, u_pad, r_pad, d_pad = pad['l'], pad['u'], pad['r'], pad['d']

        # boxes are stored in polar-like coordinates
        x0, y0, x1, y1 = self.to_cartesian_coord(self.lambda_x[subdir][self.scale_x])

        imgs, grays = [], []
        for fname in fnames:
            rgb = cv.imread(os.path.join(self.img_dir, subdir, fname))
            rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            expand_rgb = cv.copyMakeBorder(rgb, u_pad, d_pad, l_pad, r_pad,
                                            cv.BORDER_CONSTANT, value=(0, 0, 0))
            patch_rgb = expand_rgb[y0 + u_pad: y1 + u_pad, x0 + l_pad: x1 + l_pad]
            patch_rgb = cv.resize(patch_rgb, (self.img_size, self.img_size))
            patch_gray = cv.cvtColor(patch_rgb, cv.COLOR_RGB2GRAY)
            imgs.append(patch_rgb)
            grays.append(patch_gray)

        imgs, gray = np.stack(imgs), np.stack(grays)[..., np.newaxis]
        sample = {'imgs': imgs, 'gray': gray, 'label': label}

        return self.transform(sample)

    def to_cartesian_coord(self, polar_coord):
        cx, cy, r = polar_coord['cx'], polar_coord['cy'], polar_coord['r']
        return [cx - r, cy - r, cx + r, cy + r]


class ToTensor(object):
    def __init__(self):
        return

    def __call__(self, sample):
        # swap color axis: DxHxWxC => DxCxHxW
        imgs = torch.from_numpy(sample['imgs'])
        imgs = imgs.transpose(2, 3).transpose(1, 2)
        sample['imgs'] = imgs
        if 'gray' in sample.keys():
            sample['gray'] = torch.from_numpy(sample['gray'])
        if 'maps' in sample.keys():
            sample['maps'] = torch.from_numpy(sample['maps'])
        if 'label' in sample.keys():
            sample['label'] = torch.IntTensor(sample['label'])
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, sample):
        sample['imgs'] = (sample['imgs'] / 255.0 - self.mean) / self.std
        return sample


class PriorToMap(object):
    def __init__(self, map_size):
        self.map_size = map_size
        return

    def __call__(self, sample):
        priors = sample['priors']
        maps = [cv.resize(prior, (self.map_size, self.map_size)) for prior in priors]
        sample['maps'] = np.stack(maps, axis=0)
        return sample


class Batchify(object):
    def __init__(self):
        return

    def __call__(self, sample):
        sample['imgs'] = sample['imgs'].unsqueeze(dim=0)
        sample['maps'] = sample['maps'].unsqueeze(dim=0)
        return sample
