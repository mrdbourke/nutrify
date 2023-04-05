import pandas as pd
import os
from pathlib import Path
import timm

from timm.data import ImageDataset
from timm.data.readers.reader import Reader

class FoodVisionReader(Reader):
    def __init__(
        self,
        image_root,
        label_root,
        class_to_idx,
        quick_experiment,
        split="train",
    ):
        super().__init__()
        self.image_root = Path(image_root)

        # Get a mapping of classes to indexes
        self.class_to_idx = class_to_idx

        # Get a list of the samples to be used
        # TODO: could create the class_to_idx here? after loading the labels?
        # TODO: this would save opening the labels with pandas more than once...
        self.label_root = pd.read_csv(label_root)

        # Filter samples into "train" and "test"
        # TODO: add an index so I can select X amount of samples to use (e.g. for quick exerpimentation)
        # TODO: e.g. if args.quick_experiment == True: self.samples = self.samples[:100]
        if split == "train":
            self.samples = self.label_root[self.label_root["split"] == "train"][
                "image_name"
            ].to_list()
        elif split == "test":
            self.samples = self.label_root[self.label_root["split"] == "test"][
                "image_name"
            ].to_list()

        # Perform a quick training experiment on a small subset of the data
        if quick_experiment:
            self.samples = self.samples[:100]

    def __get_label(self, sample_name):
        return self.label_root.loc[self.label_root["image_name"] == sample_name][
            "label"
        ].values[0]

    def __getitem__(self, index):
        sample_name = self.samples[index]
        # print(sample_name)
        path = self.image_root / sample_name
        # print(path)
        target = self.__get_label(sample_name)
        # print(target, type(target))
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = Path(self.samples[index])
        if basename:
            filename = filename.parts[-1]
        elif not absolute:
            filename = self.image_root / filename
        return filename