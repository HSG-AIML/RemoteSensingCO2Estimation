import numpy as np
import random


class Mirror(object):
    """Mirror image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata, fptdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            imgdata = np.flip(imgdata, 2)
            fptdata = np.flip(fptdata, 1)
        return imgdata, fptdata


class Flip(object):
    """Flip image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata, fptdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            imgdata = np.flip(imgdata, 1)
            fptdata = np.flip(fptdata, 0)
        return imgdata, fptdata


class Rotate(object):
    """Rotate image."""
    def __init__(self, p=0.5, always_apply=False):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, imgdata, fptdata):
        if self.always_apply:
            self.p = 0
        if self.p < random.random():
            rot = np.random.randint(0, 4)
            imgdata = np.rot90(imgdata, rot, axes=(1, 2))
            fptdata = np.rot90(fptdata, rot, axes=(0, 1))
        return imgdata, fptdata
