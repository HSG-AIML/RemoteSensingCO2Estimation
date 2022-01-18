import os
import json
import numpy as np
import rasterio as rio
import torch
from rasterio.features import rasterize
from shapely.geometry import Polygon
from torchvision import transforms
import cv2

from custom_augmentations import Flip, Mirror, Rotate
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    """Smoke plumes subset dataset."""
    def __init__(self, channels, size=None, reg_data=None,
                 datadir=None, seglabeldir=None, mult=1,
                 transform=None):
        """
        Args:
            datadir (string): Path to the folder of the images.
        """
        fuel_type_dict = {
            'Fossil Brown coal/Lignite': 0,
            'Fossil Hard coal': 1,
            'Fossil Gas': 2,
            'Fossil Peat': 3,
            'Fossil Coal-derived gas': 2,
            'Fossil Oil': 3}

        self.datadir = datadir
        self.seglabeldir = seglabeldir
        self.reg_data = reg_data
        self.transform = transform
        self.channels = np.array(channels)

        self.size = size

        # list of image files, labels (positive or negative), segmentation
        # label vector edge coordinates
        self.imgfiles = []
        self.labels = []
        self.weather = []
        self.gen_outputs = []
        self.seglabels = []
        self.fossil_type = []

        # list of indices of positive and negative images
        self.positive_indices = []
        self.negative_indices = []

        # read in segmentation label files
        seglabels = []
        segfile_lookup = {}

        for i, seglabelfile in enumerate(os.listdir(self.seglabeldir)):
            segdata = json.load(open(os.path.join(self.seglabeldir,
                                                  seglabelfile), 'r'))
            seglabels.append(segdata)
            segfile_lookup[
                "-".join(segdata['data']['image'].split('-')[1:]).replace(
                    '.png', '.tif')] = i

        # read in image file names for positive images
        idx = 0
        for root, _, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if filename not in segfile_lookup.keys():
                    continue
                polygons = []
                for completions in seglabels[segfile_lookup[filename]]['completions']:
                    for result in completions['result']:
                        polygons.append(
                            np.array(
                                result['value']['points'] + [result['value']['points'][0]]) * self.size / 100)
                        # factor necessary to scale edge coordinates
                        # appropriately
                if 'positive' in root and polygons != []:
                    if len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
                        self.positive_indices.append(idx)
                        self.labels.append(True)
                        self.imgfiles.append(os.path.join(root, filename))
                        self.seglabels.append(polygons)
                        self.fossil_type.append(fuel_type_dict[self.reg_data[self.reg_data['filename'] == filename]['fuel_type'].unique()[0]])
                        self.gen_outputs.append(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].values[0])
                        self.weather.append(self.reg_data[self.reg_data['filename'] == filename][
                            ['temp', 'humidity', 'wind-u', 'wind-v']].to_numpy())
                        idx += 1
        # add as many negative example images
        for root, _, files in os.walk(self.datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    continue
                if idx >= len(self.positive_indices) * 2:
                    break
                if 'negative' in root:
                    if len(self.reg_data[self.reg_data['filename'] == filename]) != 0 and sum(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].isna()) == 0:
                        self.negative_indices.append(idx)
                        self.labels.append(False)
                        self.imgfiles.append(os.path.join(root, filename))
                        self.seglabels.append([])
                        self.gen_outputs.append(self.reg_data[self.reg_data['filename'] == filename]['gen_output'].values[0])
                        self.fossil_type.append(fuel_type_dict[self.reg_data[self.reg_data['filename'] == filename]['fuel_type'].unique()[0]])
                        self.weather.append(self.reg_data[self.reg_data['filename'] == filename][
                                                         ['temp', 'humidity', 'wind-u', 'wind-v']].to_numpy())
                        idx += 1
        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.gen_outputs = np.array(self.gen_outputs)
        self.labels = np.array(self.labels)
        self.fossil_type = np.array(self.fossil_type)
        self.weather = np.array(self.weather)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.weather = np.array([*self.weather] * mult)
            self.gen_outputs = np.array([*self.gen_outputs] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)
            self.seglabels = self.seglabels * mult
            self.fossil_type = np.array([*self.fossil_type] * mult)

    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)

    def __getitem__(self, idx):
        """Read in image data, preprocess, build segmentation mask, and apply
        transformations."""

        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13]])
        # keep only selected channels
        imgdata = imgdata[self.channels]

        size = imgdata.shape[1]
        # force image shape to be square
        if imgdata.shape[1] != size:
            newimgdata = np.empty((len(self.channels), size, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:,
                                                          :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:,
                                                          imgdata.shape[1] - 1:, :]
            imgdata = newimgdata
        if imgdata.shape[2] != size:
            newimgdata = np.empty((len(self.channels), size, size))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:,
                                                          :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:,
                                                          :, imgdata.shape[2] - 1:]
            imgdata = newimgdata

        # rasterize segmentation polygons
        fptdata = np.zeros(imgdata.shape[1:], dtype=np.uint8)
        polygons = self.seglabels[idx].copy()
        shapes = []

        if len(polygons) > 0:
            for pol in polygons:
                try:
                    pol = Polygon(pol)
                    shapes.append(pol)
                except ValueError:
                    continue
            fptdata = rasterize(((g, 1) for g in shapes),
                                out_shape=fptdata.shape,
                                all_touched=True)
        list_polygons = [pol.tolist() for pol in polygons]

        if size == 300:
            fptcropped = fptdata[int((fptdata.shape[0] - 120) / 2):int((fptdata.shape[0] + 120) / 2),
                                 int((fptdata.shape[1] - 120) / 2):int((fptdata.shape[1] + 120) / 2)]
            if np.sum(fptcropped) == np.sum(fptdata):
                fptdata = fptcropped
                imgdata = imgdata[:, int((imgdata.shape[1] - 120) / 2):int((imgdata.shape[1] + 120) / 2),
                                  int((imgdata.shape[2] - 120) / 2):int((imgdata.shape[2] + 120) / 2)]
            else:
                imgdata = cv2.resize(np.transpose(imgdata, (1, 2, 0)).astype('float32'), (120, 120),
                                     interpolation=cv2.INTER_CUBIC)
                imgdata = np.transpose(imgdata, (2, 0, 1))
                fptdata = cv2.resize(fptdata, (120, 120), interpolation=cv2.INTER_CUBIC)

        sample = {
            'idx': idx,
            'lbl': self.labels[idx],
            'img': imgdata,
            'fpt': fptdata,
            'type': self.fossil_type[idx],
            'gen_output': self.gen_outputs[idx],
            'weather': self.weather[idx],
            'polygons': list_polygons,
            'imgfile': self.imgfiles[idx]
        }

        # apply transformations
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'lbl': sample['lbl'],
               'type': sample['type'],
               'img': torch.from_numpy(sample['img'].copy()),
               'fpt': torch.from_numpy(sample['fpt'].copy()),
               'polygons': sample['polygons'],
               'gen_output': sample['gen_output'],
               'weather': sample['weather'],
               'imgfile': sample['imgfile']}

        return out


class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']
        fptdata = sample['fpt']

        # mirror horizontally
        func = Mirror()
        imgdata, fptdata = func(imgdata, fptdata)
        # flip vertically
        func = Flip()
        imgdata, fptdata = func(imgdata, fptdata)
        # rotate by [0,1,2,3]*90 deg
        func = Rotate()
        imgdata, fptdata = func(imgdata, fptdata)

        return {'idx': sample['idx'],
                'lbl': sample['lbl'],
                'type': sample['type'],
                'img': imgdata.copy(),
                'fpt': fptdata.copy(),
                'polygons': sample['polygons'],
                'gen_output': sample['gen_output'],
                'weather': sample['weather'],
                'imgfile': sample['imgfile']}


class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self, channels):
        self.channels_means = np.array(
            [960.97437, 1110.9012, 1250.0942, 1259.5178, 1500.98,
             1989.6344, 2155.846, 2251.6265, 2272.9438, 2442.6206,
             1914.3, 1512.0585])
        self.channels_stds = np.array(
            [1302.0157, 1418.4988, 1381.5366, 1406.7112, 1387.4155, 1438.8479,
             1497.8815, 1604.1998, 1516.532, 1827.3025, 1303.83, 1189.9052])

        self.channel_means = self.channels_means[channels]
        self.channel_stds = self.channels_stds[channels]


    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """
        sample['img'] = (sample['img'] - self.channel_means.reshape(
            sample['img'].shape[0], 1, 1)) / self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)
        return sample


def create_dataset(*args, apply_transforms=True, train=False, size=120,
                   channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], **kwargs):
    """Create a dataset; uses same input parameters as PowerPlantDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        if train:
            data_transforms = transforms.Compose([
                Normalize(np.array(channels)),
                Randomize(),
                ToTensor()])
        else:
            data_transforms = transforms.Compose([
                Normalize(np.array(channels)),
                ToTensor()])

    data = MultiTaskDataset(channels=channels, size=size, *args, **kwargs,
                            transform=data_transforms)
    return data
