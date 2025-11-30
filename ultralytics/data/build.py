# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataloader, distributed

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data.dataset import GroundingDataset, YOLODataset, YOLOMultiModalDataset
from ultralytics.data.loaders import (
    LOADERS,
    LoadImagesAndVideos,
    LoadPilAndNumpy,
    LoadScreenshots,
    LoadStreams,
    LoadTensor,
    SourceTypes,
    autocast_list,
)
from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.utils import RANK, colorstr, LOGGER
from ultralytics.utils.checks import check_file
from ultralytics.utils.torch_utils import TORCH_2_0

# NEW: ROI policies factory (build from YAML config if provided here)
try:
    from ultralytics.data.roi_policies import build_roi_policy_from_config
except Exception:
    build_roi_policy_from_config = None  # optional import; will warn if used without module


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers for infinite iteration.

    This dataloader extends the PyTorch DataLoader to provide infinite recycling of workers, which improves efficiency
    for training loops that need to iterate through the dataset multiple times without recreating workers.

    Attributes:
        batch_sampler (_RepeatSampler): A sampler that repeats indefinitely.
        iterator (Iterator): The iterator from the parent DataLoader.

    Methods:
        __len__: Return the length of the batch sampler's sampler.
        __iter__: Create a sampler that repeats indefinitely.
        __del__: Ensure workers are properly terminated.
        reset: Reset the iterator, useful when modifying dataset settings during training.

    Examples:
        Create an infinite dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = InfiniteDataLoader(dataset, batch_size=16, shuffle=True)
        >>> for batch in dataloader:  # Infinite iteration
        >>>     train_step(batch)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the InfiniteDataLoader with the same arguments as DataLoader."""
        if not TORCH_2_0:
            kwargs.pop("prefetch_factor", None)  # not supported by earlier versions
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        """Return the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self) -> Iterator:
        """Create an iterator that yields indefinitely from the underlying iterator."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def __del__(self):
        """Ensure that workers are properly terminated when the dataloader is deleted."""
        try:
            if not hasattr(self.iterator, "_workers"):
                return
            for w in self.iterator._workers:  # force terminate
                if w.is_alive():
                    w.terminate()
            self.iterator._shutdown_workers()  # cleanup
        except Exception:
            pass

    def reset(self):
        """Reset the iterator to allow modifications to the dataset during training."""
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever for infinite iteration.

    This sampler wraps another sampler and yields its contents indefinitely, allowing for infinite iteration
    over a dataset without recreating the sampler.

    Attributes:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler: Any):
        """Initialize the _RepeatSampler with a sampler to repeat indefinitely."""
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        """Iterate over the sampler indefinitely, yielding its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id: int):  # noqa
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_yolo_dataset_old(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset

    # --- NEW: parse ROI-related configs from data (YAML) ---
    roi = data.get("roi", None)  # global fixed ROI [x1,y1,x2,y2]
    output_shape = data.get("output_shape", None)
    if isinstance(output_shape, (list, tuple)) and len(output_shape) == 2:
        output_shape = (int(output_shape[0]), int(output_shape[1]))
    elif output_shape is not None:
        LOGGER.warning(f"{colorstr('yellow','bold')}Invalid output_shape in data YAML: {output_shape}, expect [W,H]. Ignoring.")
        output_shape = None

    # If an object already built upstream (e.g., in BaseTrainer.get_dataset), reuse it
    roi_policy_obj = data.get("roi_policy_obj", None)

    # Else, if YAML provided a raw roi_policy dict and factory available, build it here
    if roi_policy_obj is None and isinstance(data.get("roi_policy", None), dict):
        if build_roi_policy_from_config is None:
            LOGGER.warning(
                f"{colorstr('yellow','bold')}roi_policy config found but roi_policies module missing. "
                f"Install/add ultralytics.data.roi_policies to enable policy building here."
            )
        else:
            try:
                roi_policy_obj = build_roi_policy_from_config(data["roi_policy"])
            except Exception as e:
                LOGGER.warning(f"{colorstr('yellow','bold')}Failed to build roi_policy from YAML: {e}. Using 'roi' if provided.")
                roi_policy_obj = None

    return dataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.0, # else 0.5
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,

        # NEW: pass-through ROI params (YOLODataset/BaseDataset should accept these)
        roi=roi,
        roi_policy=roi_policy_obj,
        output_shape=output_shape,
    )

def build_yolo_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
):
    """Build and return a YOLO dataset based on configuration parameters."""
    
    # 动态确定 dataset_class
    dataset_class = YOLOMultiModalDataset if multi_modal else YOLODataset

    # --- 新的双ROI & 多分辨率构建逻辑 ---
    roi_configs = data.get("roi_configs")
    # 仅在训练模式且 roi_configs 是一个列表时生效
    if mode == "train" and isinstance(roi_configs, list) and len(roi_configs) > 1:
        LOGGER.info(f"{colorstr('green', 'bold', 'Multi-ROI Training Enabled:')} Found {len(roi_configs)} ROI configurations.")
        datasets = []
        for i, roi_conf in enumerate(roi_configs):
            roi_name = roi_conf.get("name", f"ROI-{i+1}")
            LOGGER.info(f"Building dataset for {colorstr('blue', roi_name)}...")
            
            # --- 从每个 ROI 配置中解析参数 ---
            # 您的代码已经支持 roi_policy 是一个字典
            roi_policy_dict = roi_conf.get("roi_policy")

            try:
                roi_policy_obj = build_roi_policy_from_config(roi_conf["roi_policy"])
            except Exception as e:
                LOGGER.warning(f"{colorstr('yellow','bold')}Failed to build roi_policy from YAML: {e}. Using 'roi' if provided.")
                roi_policy_obj = None

            output_shape = roi_conf.get("output_shape")

            # 注意：不再需要解析单一的 `roi` 字段，因为所有逻辑都在 `roi_policy` 中
            
            ds = dataset_class(
                img_path=img_path,
                imgsz=cfg.imgsz,
                batch_size=batch,
                augment=True,
                hyp=cfg,
                rect=cfg.rect or rect,
                cache=cfg.cache or None,
                single_cls=cfg.single_cls or False,
                stride=stride,
                pad=0.0,
                prefix=colorstr(f"{mode} ({roi_name}): "),
                task=cfg.task,
                classes=cfg.classes,
                data=data,
                fraction=cfg.fraction,
                
                # --- 为每个数据集实例传入特定的 ROI 参数 ---
                # `roi` 参数可以设为 None，因为所有逻辑都由 roi_policy 处理
                roi=None, 
                roi_policy=roi_policy_obj, 
                output_shape=output_shape,
                roi_name=roi_name,
            )
            datasets.append(ds)
        
        # 使用 YOLOConcatDataset (如果存在) 或标准 ConcatDataset 拼接
        try:
            from ultralytics.data.dataset import YOLOConcatDataset
            return YOLOConcatDataset(datasets)
        except ImportError:
            from torch.utils.data import ConcatDataset
            return ConcatDataset(datasets)

    # --- 验证/测试或单ROI模式的逻辑 ---
    # 如果不是训练模式，或 roi_configs 不存在/为空，则使用列表的第一个配置或旧的单一配置
    roi_policy_dict = None
    output_shape_val = None
    roi_val = None
    prefix_mode = mode

    if isinstance(roi_configs, list) and len(roi_configs) > 0:
        roi_conf = roi_configs[0]
        roi_name = roi_conf.get("name", f"ROI-1")
        LOGGER.info(f"Building dataset for {colorstr('blue', roi_name)}...")
        roi_policy_dict = roi_conf.get("roi_policy")
        output_shape_val = roi_conf.get("output_shape")
        prefix_mode = f"{mode} ({roi_conf.get('name', 'Default_ROI')})"
    else:
        # Fallback to legacy single roi/roi_policy fields from the root of the data dict
        roi_val = data.get("roi")
        roi_policy_dict = data.get("roi_policy")
        output_shape_val = data.get("output_shape")
    
    # 您的 `build_roi_policy_from_config` 逻辑可以放在这里，作用于 roi_policy_dict
    try:
        roi_policy_obj = build_roi_policy_from_config(roi_conf["roi_policy"])
    except Exception as e:
        LOGGER.warning(f"{colorstr('yellow','bold')}Failed to build roi_policy from YAML: {e}. Using 'roi' if provided.")
        roi_policy_obj = None

    return dataset_class(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.0,
        prefix=colorstr(f"{prefix_mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        roi=roi_val,
        roi_policy=roi_policy_obj,
        output_shape=output_shape_val,
        roi_name=roi_name,
    )

def build_grounding(
    cfg: IterableSimpleNamespace,
    img_path: str,
    json_file: str,
    batch: int,
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    max_samples: int = 80,
):
    """Build and return a GroundingDataset based on configuration parameters."""
    return GroundingDataset(
        img_path=img_path,
        json_file=json_file,
        max_samples=max_samples,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )


def build_dataloader(dataset, batch: int, workers: int, shuffle: bool = True, rank: int = -1, drop_last: bool = False):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool, optional): Whether to shuffle the dataset.
        rank (int, optional): Process rank in distributed training. -1 for single-GPU training.
        drop_last (bool, optional): Whether to drop the last incomplete batch.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.

    Examples:
        Create a dataloader for training
        >>> dataset = YOLODataset(...)
        >>> dataloader = build_dataloader(dataset, batch=16, workers=4, shuffle=True)
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        prefetch_factor=4 if nw > 0 else None,  # increase over default 2
        pin_memory=nd > 0,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def check_source(source):
    """
    Check the type of input source and return corresponding flag values.

    Args:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The input source to check.

    Returns:
        source (str | int | Path | list | tuple | np.ndarray | PIL.Image | torch.Tensor): The processed source.
        webcam (bool): Whether the source is a webcam.
        screenshot (bool): Whether the source is a screenshot.
        from_img (bool): Whether the source is an image or list of images.
        in_memory (bool): Whether the source is an in-memory object.
        tensor (bool): Whether the source is a torch.Tensor.

    Examples:
        Check a file path source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source("image.jpg")

        Check a webcam source
        >>> source, webcam, screenshot, from_img, in_memory, tensor = check_source(0)
    """
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        source_lower = source.lower()
        is_file = source_lower.rpartition(".")[-1] in (IMG_FORMATS | VID_FORMATS)
        is_url = source_lower.startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source_lower == "screen"
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, LOADERS):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError("Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict")

    return source, webcam, screenshot, from_img, in_memory, tensor


def load_inference_source(source=None, batch: int = 1, vid_stride: int = 1, buffer: bool = False, channels: int = 3):
    """
    Load an inference source for object detection and apply necessary transformations.

    Args:
        source (str | Path | torch.Tensor | PIL.Image | np.ndarray, optional): The input source for inference.
        batch (int, optional): Batch size for dataloaders.
        vid_stride (int, optional): The frame interval for video sources.
        buffer (bool, optional): Whether stream frames will be buffered.
        channels (int, optional): The number of input channels for the model.

    Returns:
        (Dataset): A dataset object for the specified input source with attached source_type attribute.

    Examples:
        Load an image source for inference
        >>> dataset = load_inference_source("image.jpg", batch=1)

        Load a video stream source
        >>> dataset = load_inference_source("rtsp://example.com/stream", vid_stride=2)
    """
    source, stream, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(stream, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif stream:
        dataset = LoadStreams(source, vid_stride=vid_stride, buffer=buffer, channels=channels)
    elif screenshot:
        dataset = LoadScreenshots(source, channels=channels)
    elif from_img:
        dataset = LoadPilAndNumpy(source, channels=channels)
    else:
        dataset = LoadImagesAndVideos(source, batch=batch, vid_stride=vid_stride, channels=channels)

    # Attach source types to the dataset
    setattr(dataset, "source_type", source_type)

    return dataset
