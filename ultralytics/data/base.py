# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference
    in object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        cv2_flag (int): OpenCV flag for reading images.
        im_files (list[str]): List of image file paths.
        labels (list[dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (list[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    """
    Base dataset class for loading and processing image data.

    Added ROI support:
      - global ROI: pass roi=(x1,y1,x2,y2) to __init__
      - per-image ROI: provide label["roi"] = (x1,y1,x2,y2) in labels loaded by get_labels()
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
        roi: tuple[int, int, int, int] | None = None,  # NEW: global roi (x1,y1,x2,y2) or None
        roi_policy: Any | None = None, 
        output_shape: tuple[int,int] | None = None,
        roi_name: str | None = None  # <<<< 新增 roi_name 参数
    ):
        """
        Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | list[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (list[int], optional): List of included classes.
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR

        self.roi = roi  # store global ROI
        self.roi_policy = roi_policy
        self.output_shape = output_shape
        self.roi_name = roi_name if roi_name else "default" # <<<< 新增：存储 roi_name

        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # # Cache images (options are cache = True, False, None, "ram", "disk")
        # self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        # self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        # self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        # if self.cache == "ram" and self.check_cache_ram():
        #     if hyp.deterministic:
        #         LOGGER.warning(
        #             "cache='ram' may produce non-deterministic training results. "
        #             "Consider cache='disk' as a deterministic alternative if your disk space allows."
        #         )
        #     self.cache_images()
        # elif self.cache == "disk" and self.check_cache_disk():
        #     self.cache_images()

        # Cache images (options are cache = True, False, None, "ram", "ram_full", "disk")
        self.ims, self.ims_full = [None] * self.ni, [None] * self.ni
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None

        # If using dynamic ROI policy, avoid 'ram' (would lock first crop); prefer 'ram_full' or 'disk'
        if self.roi_policy is not None and self.cache == "ram":
            LOGGER.info(
                f"{self.prefix}Dynamic ROI policy detected with cache='ram'. Switching to cache='ram_full' to keep randomness."
            )
            self.cache = "ram_full"

        if self.cache == "ram" and self.check_cache_ram():
            if getattr(hyp, "deterministic", False):
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "ram_full" and self.check_cache_ram():
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """
        Read image files from the specified path.

        Args:
            img_path (str | list[str]): Path or list of paths to image directories or files.

        Returns:
            (list[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """
        Update labels to include only specified classes.

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]

                labels_3d = self.labels[i].get("labels_3d", None)
                faces_3d = self.labels[i].get("faces_3d", None)
                has_3d_mask = self.labels[i].get("has_3d_mask", None)
                vehicle_mask = self.labels[i].get("vehicle_mask", None)
                face_vis_mask = self.labels[i].get("face_vis_mask", None)
                face_weight = self.labels[i].get("face_weight", None)

                self.labels[i]["labels_3d"] = labels_3d[j] if labels_3d is not None else None
                self.labels[i]["faces_3d"] = faces_3d[j] if faces_3d is not None else None
                self.labels[i]["has_3d_mask"] = has_3d_mask[j] if has_3d_mask is not None else None
                self.labels[i]["vehicle_mask"] = vehicle_mask[j] if vehicle_mask is not None else None
                self.labels[i]["face_vis_mask"] = face_vis_mask[j] if face_vis_mask is not None else None
                self.labels[i]["face_weight"] = face_weight[j] if face_weight is not None else None

                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def _load_full_image(self, i: int) -> np.ndarray:
        """
        Load full image (no crop/resize), using RAM cache if available or disk .npy cache.
        """
        im_full, f, fn = self.ims_full[i], self.im_files[i], self.npy_files[i]
        if im_full is not None:
            return im_full
        if fn.exists():  # load npy-cached full image
            try:
                im_full = np.load(fn)
            except Exception as e:
                LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                Path(fn).unlink(missing_ok=True)
                im_full = imread(f, flags=self.cv2_flag)
        else:
            im_full = imread(f, flags=self.cv2_flag)
        if im_full is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        if im_full.ndim == 2:
            im_full = im_full[..., None]
        return im_full

    def load_image(self, i: int, rect_mode: bool = True) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def load_image_with_ROI(
        self, i: int, rect_mode: bool = True, roi: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index 'i' with optional ROI cropping and resizing.

        Returns:
            (im_resized, (h0_full, w0_full), (h0_crop, w0_crop), (h_resized, w_resized))
        """
        # If caching original full images, start from full
        if self.cache == "ram_full":
            im_full = self._load_full_image(i)
        else:
            # If cropped+resized image is cached in RAM and valid, return the cached triplet extended to 4-tuple
            im_cached = self.ims[i]
            if im_cached is not None:
                # self.im_hw0[i] holds the 'crop original' size; self.im_hw[i] is resized size
                return im_cached, self.im_hw0[i], self.im_hw[i], self.im_hw[i]

            # Otherwise load full from disk (or .npy)
            im_full = self._load_full_image(i)

        h0_full, w0_full = im_full.shape[:2]

        # Apply ROI crop if provided
        if roi is None:
            im_crop = im_full
            h0_crop, w0_crop = h0_full, w0_full
            x1 = y1 = 0  # not used further but define to keep static analyzers happy
        else:
            x1, y1, x2, y2 = (int(round(v)) for v in roi)
            # clip to image bounds
            x1 = max(0, min(x1, w0_full - 1))
            y1 = max(0, min(y1, h0_full - 1))
            x2 = max(x1 + 1, min(x2, w0_full))
            y2 = max(y1 + 1, min(y2, h0_full))
            im_crop = im_full[y1:y2, x1:x2]
            h0_crop, w0_crop = im_crop.shape[:2]

        # Resize
        if self.output_shape is not None:
            W, H = int(self.output_shape[0]), int(self.output_shape[1])
            im_resized = cv2.resize(im_crop, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0_crop, w0_crop)
                if r != 1:
                    w, h = (min(math.ceil(w0_crop * r), self.imgsz), min(math.ceil(h0_crop * r), self.imgsz))
                    im_resized = cv2.resize(im_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    im_resized = im_crop
            elif not (h0_crop == w0_crop == self.imgsz):  # stretch to square imgsz
                im_resized = cv2.resize(im_crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            else:
                im_resized = im_crop

        if im_resized.ndim == 2:
            im_resized = im_resized[..., None]

        # Cache cropped+resized only in 'ram' mode (not in 'ram_full', to keep randomness)
        if self.augment and self.cache == "ram":
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im_resized, (h0_crop, w0_crop), im_resized.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im_resized, (h0_full, w0_full), (h0_crop, w0_crop), im_resized.shape[:2]

    def load_image_with_ROI_v1(self, i: int, rect_mode: bool = True, roi: tuple[int, int, int, int] | None = None) -> tuple:
        """
        Load an image from dataset index 'i'.

        Now supports ROI cropping. Returns:
            (im_resized, (h0_full, w0_full), (h0_crop, w0_crop), (h_resized, w_resized))

        Previously it returned (im, (h0, w0), im.shape[:2]).
        To stay compatible with callers expecting 3-tuple, callers are updated to accept a 4-tuple.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im_full = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im_full = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im_full = imread(f, flags=self.cv2_flag)  # BGR
            if im_full is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0_full, w0_full = im_full.shape[:2]  # full original hw

            # Crop ROI if provided (roi in pixel coordinates relative to full image)
            if roi is None:
                roi_pixel = None
                im_crop = im_full
                h0_crop, w0_crop = h0_full, w0_full
            else:
                x1, y1, x2, y2 = (int(round(v)) for v in roi)
                # x1 = max(0, min(x1, w0_full - 1))
                # y1 = max(0, min(y1, h0_full - 1))
                # x2 = max(x1 + 1, min(x2, w0_full))
                # y2 = max(y1 + 1, min(y2, h0_full))
                im_crop = im_full[y1:y2, x1:x2]
                h0_crop, w0_crop = im_crop.shape[:2]

            # Resize / rect logic applied to the cropped image
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0_crop, w0_crop)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0_crop * r), self.imgsz), min(math.ceil(h0_crop * r), self.imgsz))
                    im_resized = cv2.resize(im_crop, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    im_resized = im_crop
            elif not (h0_crop == w0_crop == self.imgsz):  # resize by stretching image to square imgsz
                im_resized = cv2.resize(im_crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            else:
                im_resized = im_crop

            if im_resized.ndim == 2:
                im_resized = im_resized[..., None]

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im_resized, (h0_crop, w0_crop), im_resized.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            # Return resized image and shapes
            return im_resized, (h0_full, w0_full), (h0_crop, w0_crop), im_resized.shape[:2]

        # if cached in RAM, we previously stored im as the resized image; we don't have crop info stored -> return existing shapes
        return self.ims[i], self.im_hw0[i], self.im_hw[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes

        if self.cache == "disk":
            fcn, storage = (self.cache_images_to_disk, "Disk")
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, _ in pbar:
                    b += self.npy_files[i].stat().st_size if self.npy_files[i].exists() else 0
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
                pbar.close()
            return

        if self.cache == "ram_full":
            fcn, storage = (self._load_full_image, "RAM(full)")
            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(fcn, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, im_full in pbar:
                    self.ims_full[i] = im_full
                    b += self.ims_full[i].nbytes
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
                pbar.close()
            return

        if self.cache == "ram":
            storage = "RAM"
            # Build per-index ROI if global ROI is given; avoid dynamic roi_policy here
            def _load_with_roi(idx: int):
                roi_i = None
                # If per-label roi exists use it, else global self.roi; ignore dynamic roi_policy at caching time
                if isinstance(self.labels[idx], dict):
                    roi_i = self.labels[idx].get("roi", None)
                roi_i = self.roi if roi_i is None else roi_i
                return self.load_image_with_ROI(idx, rect_mode=True, roi=roi_i)

            with ThreadPool(NUM_THREADS) as pool:
                results = pool.imap(_load_with_roi, range(self.ni))
                pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
                for i, x in pbar:
                    im_resized, hw_full, hw_crop, hw_resized = x
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = im_resized, hw_crop, hw_resized
                    b += self.ims[i].nbytes
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
                pbar.close()

    def cache_images_v1(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image_with_ROI, "RAM") #(self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """
        Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """
        Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    # def set_rectangle(self) -> None:
    #     """Set the shape of bounding boxes for YOLO detections as rectangles."""
    #     bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
    #     nb = bi[-1] + 1  # number of batches

    #     s = np.array([x.pop("shape") for x in self.labels])  # hw
    #     ar = s[:, 0] / s[:, 1]  # aspect ratio
    #     irect = ar.argsort()
    #     self.im_files = [self.im_files[i] for i in irect]
    #     self.labels = [self.labels[i] for i in irect]
    #     ar = ar[irect]

    #     # Set training image shapes
    #     shapes = [[1, 1]] * nb
    #     for i in range(nb):
    #         ari = ar[bi == i]
    #         mini, maxi = ari.min(), ari.max()
    #         if maxi < 1:
    #             shapes[i] = [maxi, 1]
    #         elif mini > 1:
    #             shapes[i] = [1, 1 / mini]

    #     self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
    #     self.batch = bi  # batch index of image
    def set_rectangle(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        # Decide effective shape per image:
        # Priority for shape estimation:
        # - If dynamic roi_policy and output_shape provided -> use output_shape (H,W) for all
        # - Else if per-label roi/global roi -> use ROI crop shape
        # - Else fall back to label["shape"] (full image)
        s_list = []
        if self.roi_policy is not None and self.output_shape is not None:
            # Use unified output shape
            H, W = int(self.output_shape[1]), int(self.output_shape[0])  # output_shape is (W,H)
            s_list = [(H, W) for _ in range(self.ni)]
            for i in range(self.ni):
                self.labels[i]["shape"] = (H, W)
        else:
            for i in range(len(self.labels)):
                label = self.labels[i]
                orig_shape = label.get("shape", None)  # (h, w) of full image
                if orig_shape is None:
                    full_h, full_w = self.imgsz, self.imgsz
                else:
                    full_h, full_w = int(orig_shape[0]), int(orig_shape[1])

                # determine roi: per-image overrides global
                roi = label.get("roi", None) if isinstance(label, dict) else None
                if roi is None:
                    roi = getattr(self, "roi", None)

                if roi is None:
                    crop_h, crop_w = full_h, full_w
                else:
                    x1, y1, x2, y2 = (int(round(v)) for v in roi)
                    x1 = max(0, min(x1, full_w - 1))
                    y1 = max(0, min(y1, full_h - 1))
                    x2 = max(x1 + 1, min(x2, full_w))
                    y2 = max(y1 + 1, min(y2, full_h))
                    crop_w = max(1, x2 - x1)
                    crop_h = max(1, y2 - y1)

                # update label["shape"] to be the crop shape so later logic sees cropped sizes
                self.labels[i]["shape"] = (int(crop_h), int(crop_w))
                s_list.append((int(crop_h), int(crop_w)))

        s = np.array(s_list)  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def set_rectangle_v1(self) -> None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        # --- ROI-aware shape extraction: for each label compute crop shape if roi present ---
        s_list = []
        for i in range(len(self.labels)):
            label = self.labels[i]
            # original shape stored in label["shape"] is (h, w) for full image
            orig_shape = label.get("shape", None)
            if orig_shape is None:
                # fallback: if missing, assume (imgsz, imgsz) to avoid zero division
                full_h, full_w = self.imgsz, self.imgsz
            else:
                full_h, full_w = int(orig_shape[0]), int(orig_shape[1])

            # determine roi: per-image overrides global
            roi = label.get("roi", None) if isinstance(label, dict) else None
            if roi is None:
                roi = getattr(self, "roi", None)

            if roi is None:
                crop_h, crop_w = full_h, full_w
            else:
                # roi expected as (x1, y1, x2, y2) in pixel coords relative to full image
                x1, y1, x2, y2 = (int(round(v)) for v in roi)
                # clip to image bounds
                x1 = max(0, min(x1, full_w - 1))
                y1 = max(0, min(y1, full_h - 1))
                x2 = max(x1 + 1, min(x2, full_w))
                y2 = max(y1 + 1, min(y2, full_h))
                crop_w = max(1, x2 - x1)
                crop_h = max(1, y2 - y1)

            # update label["shape"] to be the crop shape so later logic sees cropped sizes
            self.labels[i]["shape"] = (int(crop_h), int(crop_w))
            s_list.append((int(crop_h), int(crop_w)))

        s = np.array(s_list)  # hw

        # --- rest of original logic unchanged ---
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def _adapt_label_for_roi(
        self,
        label: dict[str, Any],
        roi: tuple[int, int, int, int] | None,
        full_hw: tuple[int, int],
        crop_hw: tuple[int, int],
    ) -> dict[str, Any]:
        """
        If roi is not None, adapt label coordinates (which are expected to be normalized relative to full image)
        to the cropped image coordinate system and re-normalize to the cropped image.

        This function mutates and returns label (deepcopy already done by caller).
        It handles:
          - label["bboxes"]: xywh normalized OR absolute (we detect by value ranges)
          - label["segments"]: list of np.array (N,2) normalized OR absolute
          - label["keypoints"]: numpy array [..., 2] or normalized
          - 3D fields: best-effort transform of x,y components (first two dims).
        Also filters out objects that fall fully outside the crop.
        """
        if roi is None:
            return label

        x1, y1, x2, y2 = roi
        full_h, full_w = full_hw
        crop_h, crop_w = crop_hw

        # Helper: convert normalized xywh -> absolute cx,cy,w,h (pixels)
        bboxes = label.get("bboxes")
        if bboxes is None or bboxes.size == 0:
            label["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            # segments/keypoints/3d keep original shapes or empty
            return label

        b = np.array(bboxes, dtype=np.float32).copy()  # (n,4)
        normalized_flag = label.get("normalized", True)

        if normalized_flag:
            # xywh normalized w.r.t full image
            cx_abs = b[:, 0] * full_w
            cy_abs = b[:, 1] * full_h
            w_abs = b[:, 2] * full_w
            h_abs = b[:, 3] * full_h
        else:
            cx_abs = b[:, 0]
            cy_abs = b[:, 1]
            w_abs = b[:, 2]
            h_abs = b[:, 3]

        # Convert center-based -> xyxy absolute (full image coords)
        x1_box = cx_abs - w_abs / 2.0
        y1_box = cy_abs - h_abs / 2.0
        x2_box = cx_abs + w_abs / 2.0
        y2_box = cy_abs + h_abs / 2.0

        # Shift into crop coordinates (origin at roi x1,y1)
        x1_box_crop = x1_box - x1
        y1_box_crop = y1_box - y1
        x2_box_crop = x2_box - x1
        y2_box_crop = y2_box - y1

        # Intersection with crop bounds
        clip_x1 = np.clip(x1_box_crop, 0, crop_w)
        clip_y1 = np.clip(y1_box_crop, 0, crop_h)
        clip_x2 = np.clip(x2_box_crop, 0, crop_w)
        clip_y2 = np.clip(y2_box_crop, 0, crop_h)

        # Compute areas for filtering
        orig_area = np.maximum(0.0, (x2_box_crop - x1_box_crop)) * np.maximum(0.0, (y2_box_crop - y1_box_crop))
        inter_w = np.maximum(0.0, clip_x2 - clip_x1)
        inter_h = np.maximum(0.0, clip_y2 - clip_y1)
        inter_area = inter_w * inter_h

        # Decide keep mask:
        # - keep if intersection area is > 0 and either center is inside crop,
        #   or intersection area is at least min_area_ratio of original area (to avoid tiny fragments).
        min_area_ratio = 0.3  # keep if >= 5% of original area (tunable)
        center_x_crop = (cx_abs - x1)
        center_y_crop = (cy_abs - y1)
        center_inside = (center_x_crop > 0) & (center_x_crop < crop_w) & (center_y_crop > 0) & (center_y_crop < crop_h)

        # handle orig_area == 0 (degenerate) by comparing against small absolute threshold
        orig_area_safe = np.where(orig_area <= 0.0, 1.0, orig_area)
        keep_by_area = inter_area / orig_area_safe >= min_area_ratio
        keep_mask = (inter_area > 0) & (center_inside | keep_by_area)

        # # Convert center-based absolute -> crop-relative absolute
        # cx_crop = cx_abs - x1
        # cy_crop = cy_abs - y1
        # # widths/heights unchanged in pixels
        # w_crop = w_abs
        # h_crop = h_abs

        # # Re-normalize relative to crop size
        # cx_norm = cx_crop / float(crop_w)
        # cy_norm = cy_crop / float(crop_h)
        # w_norm = w_crop / float(crop_w)
        # h_norm = h_crop / float(crop_h)

        # # Build new bboxes (xywh normalized to cropped image)
        # new_bboxes = np.stack([cx_norm, cy_norm, w_norm, h_norm], axis=1).astype(np.float32)

        # # Filter objects that are fully outside the crop or degenerate sizes
        # # We consider an object valid if its center lies inside [0,1] and width/height > eps
        # mask_keep = (cx_norm > 0) & (cx_norm < 1) & (cy_norm > 0) & (cy_norm < 1) & (w_norm > 1e-6) & (h_norm > 1e-6)
        if keep_mask.sum() == 0:
            # return empty label structures consistent with others
            label["cls"] = label["cls"][:0]
            label["bboxes"] = np.zeros((0, 4), dtype=np.float32)
            label["segments"] = []
            if "keypoints" in label and label["keypoints"] is not None:
                label["keypoints"] = np.zeros((0, label["keypoints"].shape[1]), dtype=label["keypoints"].dtype)
            # 3d zeros/empties if present
            for k in ("labels_3d", "faces_3d", "has_3d_mask", "vehicle_mask", "face_vis_mask", "face_weight"):
                if k in label:
                    v = label[k]
                    if isinstance(v, np.ndarray):
                        label[k] = v[:0]
                    else:
                        label[k] = np.array([], dtype=type(v[0]) if len(v) else v.dtype)
            return label
        
        # Build new clipped boxes and convert back to normalized xywh (relative to crop)
        new_x1 = clip_x1[keep_mask]
        new_y1 = clip_y1[keep_mask]
        new_x2 = clip_x2[keep_mask]
        new_y2 = clip_y2[keep_mask]

        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        new_cx = new_x1 + new_w / 2.0
        new_cy = new_y1 + new_h / 2.0

        # Normalize to crop size
        new_bboxes = np.stack(
            [new_cx / float(crop_w), new_cy / float(crop_h), new_w / float(crop_w), new_h / float(crop_h)], axis=1
        ).astype(np.float32)

        # Apply mask to cls and bboxes
        label["cls"] = label["cls"][keep_mask]
        label["bboxes"] = new_bboxes #[mask_keep]
        # segments: list of arrays, each array is (M,2) normalized or absolute depending on label["normalized"]
        segs = label.get("segments", [])
        if segs:
            new_segs = []
            for seg, keep in zip(segs, keep_mask):
                if not keep:
                    continue
                seg_arr = np.array(seg, dtype=np.float32).reshape(-1, 2).copy()
                if normalized_flag:
                    # normalized to full image -> absolute -> crop -> normalize to crop
                    seg_arr[:, 0] = seg_arr[:, 0] * full_w - x1
                    seg_arr[:, 1] = seg_arr[:, 1] * full_h - y1
                    seg_arr[:, 0] = seg_arr[:, 0] / float(crop_w)
                    seg_arr[:, 1] = seg_arr[:, 1] / float(crop_h)
                else:
                    seg_arr[:, 0] = seg_arr[:, 0] - x1
                    seg_arr[:, 1] = seg_arr[:, 1] - y1
                    seg_arr[:, 0] = seg_arr[:, 0] / float(crop_w)
                    seg_arr[:, 1] = seg_arr[:, 1] / float(crop_h)
                new_segs.append(seg_arr)
            label["segments"] = new_segs
        else:
            label["segments"] = []

        # keypoints: numpy array with last dim >=2 (x,y,...)
        if "keypoints" in label and label["keypoints"] is not None:
            kp = np.array(label["keypoints"], dtype=np.float32)
            if kp.size:
                # kp shape: (n_kps, k, 2/3) or (n, nk*dim) depending on format; handle common case n x (nk*dim) stored as array
                # Here we assume keypoints stored per-instance (n_instances, nk*dim) OR (n_instances, nk, dim)
                kps = kp.copy()
                if kps.ndim == 2 and (kps.shape[1] % 2 == 0 and kps.shape[1] > 2):  # flat format
                    # reshape to (n_instances, nk, 2)
                    nk = kps.shape[1] // 2
                    kps = kps.reshape(kps.shape[0], nk, 2)
                    flat_back = True
                else:
                    flat_back = False
                # transform each point
                for i in range(kps.shape[0]):
                    for j in range(kps.shape[1]):
                        x = kps[i, j, 0]
                        y = kps[i, j, 1]
                        if normalized_flag:
                            x = x * full_w - x1
                            y = y * full_h - y1
                            x = x / float(crop_w)
                            y = y / float(crop_h)
                        else:
                            x = x - x1
                            y = y - y1
                            x = x / float(crop_w)
                            y = y / float(crop_h)
                        kps[i, j, 0] = x
                        kps[i, j, 1] = y
                if flat_back:
                    kps = kps.reshape(kps.shape[0], -1)
                label["keypoints"] = kps[keep_mask]
            else:
                label["keypoints"] = np.zeros((0,), dtype=np.float32)

        # 3D-handling: best-effort transform of first two elements (assumed x,y image coords normalized or absolute)
        # labels_3d: (n_instances, 9) -> we attempt to transform first two entries as (cx, cy) normalized -> adjust them.
        if "labels_3d" in label and isinstance(label["labels_3d"], np.ndarray) and label["labels_3d"].size:
            l3d = label["labels_3d"].astype(np.float32).copy()
            # If normalized_flag, we assume last two columns are normalized cx,cy
            if l3d.shape[1] >= 2:
                if normalized_flag:
                    l3d[:, -2] = (l3d[:, -2] * full_w - x1) / float(crop_w)
                    l3d[:, -1] = (l3d[:, -1] * full_h - y1) / float(crop_h)
                else:
                    l3d[:, -2] = (l3d[:, -2] - x1) / float(crop_w)
                    l3d[:, -1] = (l3d[:, -1] - y1) / float(crop_h)
            label["labels_3d"] = l3d[keep_mask]

        # faces_3d: (n, 4, 7) -> try to transform index (3,4) values in last dim for each face vertex
        if "faces_3d" in label and isinstance(label["faces_3d"], np.ndarray) and label["faces_3d"].size:
            f3d = label["faces_3d"].astype(np.float32).copy()
            # transform last-dim first two elements for all faces and vertices
            if f3d.ndim == 3 and f3d.shape[-1] >= 2:
                for idx in range(f3d.shape[0]):
                    for v in range(f3d.shape[1]):
                        x = f3d[idx, v, 3]
                        y = f3d[idx, v, 4]
                        if normalized_flag:
                            x = (x * full_w - x1) / float(crop_w)
                            y = (y * full_h - y1) / float(crop_h)
                        else:
                            x = (x - x1) / float(crop_w)
                            y = (y - y1) / float(crop_h)
                        f3d[idx, v, 3] = x
                        f3d[idx, v, 4] = y
            label["faces_3d"] = f3d[keep_mask]

        # Masks / weights keep selection
        for k in ("has_3d_mask", "vehicle_mask", "face_vis_mask", "face_weight"):
            if k in label:
                v = label[k]
                if isinstance(v, np.ndarray):
                    label[k] = v[keep_mask]

        # Finally ensure label["normalized"] remains True and bbox_format kept as-is
        label["normalized"] = True
        return label

    def _get_full_hw(self, index: int) -> tuple[int, int]:
        """
        Helper to fetch full original image size (h,w) for index, using caches if available.
        """
        if self.cache == "ram_full" and self.ims_full[index] is not None:
            im = self.ims_full[index]
            return im.shape[0], im.shape[1]
        # Try .npy cache to avoid decoding if present
        fn = self.npy_files[index]
        if fn.exists():
            try:
                im = np.load(fn, mmap_mode="r")
                return im.shape[0], im.shape[1]
            except Exception:
                pass
        # Fallback to imread
        im = imread(self.im_files[index], flags=self.cv2_flag)
        if im is None:
            raise FileNotFoundError(f"Image Not Found {self.im_files[index]}")
        return im.shape[0], im.shape[1]

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy()
        label.pop("shape", None)  # shape is for rect, remove it

        # Determine ROI with priority: per-image 'roi' > roi_policy > global self.roi
        roi = label.pop("roi", None)
        if roi is None:
            if self.roi_policy is not None:
                h_full, w_full = self._get_full_hw(index)
                # Policy expects (W,H)
                try:
                    roi = self.roi_policy.get_roi((w_full, h_full), meta=label)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}roi_policy.get_roi failed at index {index}: {e}. Falling back to global roi.")
                    roi = self.roi
            else:
                roi = self.roi

        # Load with ROI; returns resized image and shapes
        im_resized, hw_full, hw_crop, hw_resized = self.load_image_with_ROI(index, rect_mode=True, roi=roi)

        # Populate label dict
        label["img"], label["ori_shape"], label["resized_shape"] = im_resized, hw_crop, hw_resized
        label["full_ori_shape"] = hw_full
        label["crop_ori_shape"] = hw_crop

        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]

        # Adapt label coordinates to crop (and filter out instances fully outside crop)
        label = self._adapt_label_for_roi(label, roi, hw_full, hw_crop)

        # <<<< 新增：在此处注入 roi_name >>>>
        label["roi_name"] = self.roi_name

        return self.update_labels_info(label)

    def get_image_and_label_v1(self, index: int) -> dict[str, Any]:
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        # label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)

        # Check for per-image roi key
        roi = label.pop("roi", None)
        if roi is None:
            roi = self.roi

        # load_image now returns 4-tuple: resized_im, full_hw, crop_hw, resized_hw
        im_resized, hw_full, hw_crop, hw_resized = self.load_image_with_ROI(index, rect_mode=True, roi=roi)
        label["img"], label["ori_shape"], label["resized_shape"] = im_resized, hw_crop, hw_resized
        # keep both full image original shape and crop shape if downstream code needs full
        label["full_ori_shape"] = hw_full
        label["crop_ori_shape"] = hw_crop

        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]

        # Adapt label coordinates to crop (and filter out instances fully outside crop)
        label = self._adapt_label_for_roi(label, roi, hw_full, hw_crop)

        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """
        Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """
        Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
