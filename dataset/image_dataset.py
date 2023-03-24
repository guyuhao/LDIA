# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

"""
define image dataset, only support two parties, used for CIFAR and CINIC
"""


class ImageDataset(VisionDataset):

    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        super(ImageDataset, self).__init__(root="", transform=transform,
                                           target_transform=target_transform)
        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        if type(img) is np.str_:
            img = Image.open(img)
        else:
            img = np.array(img) * 255
            if img.shape[2] == 3:
                img = Image.fromarray(img.astype('uint8')).convert('RGB')
            elif img.shape[2] == 1:
                img = Image.fromarray(img.astype('uint8').squeeze())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
