import os
import shutil
import tarfile
import pickle
import re
from pyunpack import Archive
import zipfile
from PIL import Image
import numpy as np


class DatasetFormatter:
    """
    Global pipeline/functions for dataset formatting
    """

    def __init__(self, dataset_name, level, extra_name="", set_names=["train", "valid", "test"]):
        self.dataset_name = dataset_name
        self.level = level
        self.set_names = set_names
        self.source_fold_path = os.path.join("../raw", dataset_name)
        self.target_fold_path = os.path.join("../formatted", "{}_{}{}".format(dataset_name, level, extra_name))
        self.map_datasets_files = dict()
        self.temp_fold = None
        self.extract_with_dirname = False

    def format(self):
        self.temp_fold = os.path.join(self.target_fold_path, "temp")
        self.init_format()
        self.map_datasets_files[self.dataset_name][self.level]["format_function"]()
        self.end_format()

    def init_format(self):
        """
        Load and extracts needed files
        """
        os.makedirs(self.target_fold_path, exist_ok=True)
        os.makedirs(self.temp_fold, exist_ok=True)
        for filename in self.map_datasets_files[self.dataset_name][self.level]["needed_files"]:
            filepath = os.path.join(self.source_fold_path, filename)
            if not os.path.exists(filepath):
                print("error - {} not found".format(filepath))
                exit(-1)
        for filename in self.map_datasets_files[self.dataset_name][self.level]["arx_files"]:
            filepath = os.path.join(self.source_fold_path, filename)
            if not os.path.exists(filepath):
                print("error - {} not found".format(filepath))
                exit(-1)
            if self.extract_with_dirname:
                self.copy_or_extract(filepath, fold=os.path.dirname(filename))
            else:
                self.copy_or_extract(filepath)

        for set_name in self.set_names:
            os.makedirs(os.path.join(self.target_fold_path, set_name), exist_ok=True)

    def end_format(self):
        """
        Remove temporary folders
        """
        shutil.rmtree(self.temp_fold)

    def copy_or_extract(self, filepath, fold=None):
        """
        Extract archive files
        """
        ext = filepath.split(".")[-1]
        extract_fold = os.path.join(self.temp_fold, fold) if fold is not None else self.temp_fold
        if ext in ["tgz", "gz", "tar", "tbz2"]:
            tar = tarfile.open(filepath)
            tar.extractall(extract_fold)
            tar.close()
        elif ext == "rar":
            rar = Archive(filepath)
            rar.extractall(extract_fold)
        elif ext == "zip":
            zip = zipfile.ZipFile(filepath)
            zip.extractall(extract_fold)
            zip.close()


class OCRDatasetFormatter(DatasetFormatter):
    """
    Specific pipeline/functions for OCR/HTR dataset formatting
    """

    def __init__(self, source_dataset, level, extra_name="", set_names=["train", "valid", "test"]):
        super(OCRDatasetFormatter, self).__init__(source_dataset, level, extra_name, set_names)
        self.charset = set()
        self.gt = dict()
        for set_name in set_names:
            self.gt[set_name] = dict()

    def format_text_label(self, label):
        """
        Remove extra space or line break characters
        """
        temp = re.sub("(\n)+", '\n', label)
        return re.sub("( )+", ' ', temp).strip(" \n")

    def load_resize_save(self, source_path, target_path, source_dpi, target_dpi):
        """
        Load image, apply resolution modification and save it
        """
        if source_dpi != target_dpi:
            img = Image.open(source_path)
            img = self.resize(img, source_dpi, target_dpi)
            img = Image.fromarray(img)
            img.save(target_path)
        else:
            shutil.move(source_path, target_path)

    def resize(self, img, source_dpi, target_dpi):
        """
        Apply resolution modification to image
        """
        if source_dpi == target_dpi:
            return img
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            img = Image.fromarray(img)
        else:
            w, h = img.size
        ratio = target_dpi / source_dpi
        img = img.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
        return np.array(img)

    def adjust_coord_ratio(self, sample, ratio):
        """
        Adjust bounding box coordinates given ratio
        """
        if ratio == 1:
            return sample
        try:
            for side in ["left", "right", "bottom", "top"]:
                sample[side] = int(np.floor(sample[side] * ratio))
        except:
            print("ko")
        return sample

    def end_format(self):
        """
        Save label file
        """
        super().end_format()
        with open(os.path.join(self.target_fold_path, "labels.pkl"), "wb") as f:
            pickle.dump({
                "ground_truth": self.gt,
                "charset": sorted(list(self.charset)),
            }, f)
