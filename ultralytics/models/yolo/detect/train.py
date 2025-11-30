# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

from torch.utils.data import ConcatDataset
from pathlib import Path
import sys
import importlib.util # 新增导入

class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models
    for object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        Initialize a DetectionTrainer object for training YOLO object detection model training.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        # original
        # if getattr(dataset, "rect", False) and shuffle:
        #     LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        #     shuffle = True # False

        # --- 新的、健ateur的 rect 检查逻辑 ---
        is_rect = False
        if isinstance(dataset, ConcatDataset):
            # 如果任一子数据集是 rect, 则整体视为 rect
            is_rect = any(getattr(subset, "rect", False) for subset in dataset.datasets)
        else:
            is_rect = getattr(dataset, "rect", False)

        if is_rect and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = True # 默认为 False

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )
    
    def get_dataloader_for_qat(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            # dataset = self.build_dataset(dataset_path, mode, batch_size)
            dataset = build_yolo_dataset(self.args, dataset_path, batch_size, self.data, mode=mode, rect=True, stride=32)
        shuffle = mode == "train"
        # original
        # if getattr(dataset, "rect", False) and shuffle:
        #     LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        #     shuffle = True # False

        # --- 新的、健ateur的 rect 检查逻辑 ---
        is_rect = False
        if isinstance(dataset, ConcatDataset):
            # 如果任一子数据集是 rect, 则整体视为 rect
            is_rect = any(getattr(subset, "rect", False) for subset in dataset.datasets)
        else:
            is_rect = getattr(dataset, "rect", False)

        if is_rect and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = True # 默认为 False

        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        # 1. cfg 是 Python 文件路径：使用 Python 构造模型
        if isinstance(cfg, (str, Path)) and str(cfg).endswith(".py"):
            path = Path(cfg)
            if not path.is_file():
                raise FileNotFoundError(f"Python model config file not found: {path}")

            LOGGER.info(colorstr("model: ") + f"Loading Python model from '{path}'")

            module_name = path.stem  # 例如 yolo11
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            # 优先使用 build_model(args, data) 工厂函数
            if hasattr(module, "build_model"):
                build_fn = getattr(module, "build_model")
                model = build_fn(self.args, self.data)
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        f"build_model(...) in {path} must return an nn.Module, got {type(model)} instead."
                    )
                LOGGER.info(colorstr("model: ") + f"Instantiated model via build_model() from '{path}'.")
                return model

            # 备选：尝试寻找 YOLO11DetectionModel 类
            if hasattr(module, "YOLO11DetectionModel"):
                ModelClass = getattr(module, "YOLO11DetectionModel")
                model = ModelClass(args=self.args, nc=self.data["nc"], ch=self.data["channels"], scale="n")
                LOGGER.info(colorstr("model: ") + f"Instantiated YOLO11DetectionModel from '{path}'.")
                return model

            raise ImportError(
                f"Failed to construct model from Python file '{path}'. "
                f"Expected a 'build_model(args, data)' function or 'YOLO11DetectionModel' class."
            )
        
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        # self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        # self.loss_names = ["box", "cls", "dfl", "base3d", "faces3d", "cutcls"]
        self.loss_names = ["box", "cls", "dfl", "xyz", "proj", "size", "angle_cls", "angle_reg", "xyz_face", "proj_face", "size_face", "vis_face", "cutcls"]
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """

        # self.loss_names = ["box", "cls", "dfl", "base3d", "faces3d", "cutcls"]
        self.loss_names = ["box", "cls", "dfl", "xyz", "proj", "size", "angle_cls", "angle_reg", "xyz_face", "proj_face", "size_face", "vis_face", "cutcls"]

        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):

        # original
        # """Create a labeled training plot of the YOLO model."""
        # boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        # cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        # plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

        # modified for ConcatDataset
        if isinstance(self.train_loader.dataset, ConcatDataset):
            # 如果是，则遍历所有子数据集并收集它们的 labels
            all_labels = []
            for subset in self.train_loader.dataset.datasets:
                if hasattr(subset, 'labels'):
                    all_labels.extend(subset.labels)
                else:
                    LOGGER.warning(f"A subset of the training dataset ({type(subset).__name__}) does not have a 'labels' attribute. Skipping for plotting.")
            
            if not all_labels:
                LOGGER.warning("No labels found across all training dataset subsets. Cannot plot training labels.")
                return
            
            labels_to_plot = all_labels
        
        elif hasattr(self.train_loader.dataset, 'labels'):
            # 如果是单个数据集，则按原方式处理
            labels_to_plot = self.train_loader.dataset.labels
        
        else:
            # 数据集既不是 ConcatDataset，也没有 labels 属性
            LOGGER.warning(f"Training dataset ({type(self.train_loader.dataset).__name__}) is not a ConcatDataset and has no 'labels' attribute. Cannot plot training labels.")
            return

        # 使用收集到的标签进行后续处理
        try:
            boxes = np.concatenate([lb["bboxes"] for lb in labels_to_plot], 0)
            cls = np.concatenate([lb["cls"] for lb in labels_to_plot], 0)
        except (ValueError, TypeError) as e:
            LOGGER.error(f"Error concatenating labels for plotting: {e}. Check label format.")
            return
            
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """
        Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        # max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        # del train_dataset  # free memory
        # return super().auto_batch(max_num_obj)
            
            # --- 新的、健壮的标签收集逻辑 ---
        if isinstance(train_dataset, ConcatDataset):
            all_labels = [label for subset in train_dataset.datasets if hasattr(subset, 'labels') for label in subset.labels]
        elif hasattr(train_dataset, 'labels'):
            all_labels = train_dataset.labels
        else:
            LOGGER.warning("auto_batch: Dataset has no 'labels' attribute. Using a default value for max_num_obj.")
            all_labels = [{"cls": []}] # 提供一个默认值以避免崩溃
            
        if not all_labels:
            max_num_obj = 0
        else:
            max_num_obj = max(len(label["cls"]) for label in all_labels) * 4  # 4 for mosaic augmentation
        
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)
