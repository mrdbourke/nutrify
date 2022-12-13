# Make a custom dataset Parser for Timm to read images from a directory
from pathlib import Path
import pandas as pd

from timm.data import ImageDataset
from timm.data.readers.reader import Reader


class FoodVisionReader(Reader):
    def __init__(self, image_root, label_root, class_to_idx, split="train"):
        super().__init__()
        self.image_root = Path(image_root)

        # Get a mapping of classes to indexes
        self.class_to_idx = class_to_idx

        # Get a list of the samples to be used
        self.label_root = pd.read_csv(label_root)

        # Filter samples into "train" and "test"
        if split == "train":
            self.samples = self.label_root[self.label_root["split"] == "train"][
                "image_name"
            ].to_list()
        elif split == "test":
            self.samples = self.label_root[self.label_root["split"] == "test"][
                "image_name"
            ].to_list()

    def __get_label(self, sample_name):
        return self.label_root.loc[self.label_root["image_name"] == sample_name][
            "label"
        ].values[0]

    def __getitem__(self, index):
        sample_name = self.samples[index]
        path = self.image_root / sample_name
        target = self.__get_label(sample_name)
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
