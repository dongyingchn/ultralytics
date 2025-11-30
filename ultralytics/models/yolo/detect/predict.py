# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops
from ultralytics.utils.tal import make_anchors

import torch
import numpy as np

class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo11n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """
        Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo11n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None

        # Determine whether the model output includes an auxiliary 3D tensor
        extra3d_cat = None
        feats_for_anchors = None
        # Extract preds_for_nms (the detection tensor(s)) depending on model return
        if isinstance(preds, (list, tuple)):
            # common patterns:
            # ((y_cat, extra3d_cat), (raw_levels, extra3d_levels))
            preds = preds[0]
            
            if isinstance(preds, (list, tuple)) and len(preds) == 2:
                preds_2d = preds[0]
                extra3d_cat = preds[1]
            elif torch.is_tensor(preds) and preds.ndim == 3:
                preds_2d = preds

        # If extra3d present or we want saved feats, request indices from NMS
        need_idxs = save_feats or (extra3d_cat is not None)

        preds_nms_out = nms.non_max_suppression(
            preds_2d,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=need_idxs,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # Unpack NMS result
        if need_idxs:
            preds_kept, idxs = preds_nms_out
        else:
            preds_kept = preds_nms_out
            idxs = None

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, idxs)
        else:
            obj_feats = None

        # Align extra3d to kept detections if present
        obj_3d_raw = None
        obj_3d_decoded = None
        anchor_points = None
        stride_tensor = None

        head_module = None
        try:
            head_module = self.model.model.model[-1] if hasattr(self.model.model, "model") else None
        except Exception:
            head_module = None

        if extra3d_cat is not None and idxs is not None:
            # If we have feats (multi-level) use them to compute anchors/strides for decode
            feats = feats_for_anchors if feats_for_anchors is not None else None
            try:
                if feats is not None:
                    ap, st = make_anchors(feats, getattr(head_module, "stride", None), 0.5)
                    anchor_points = ap
                    stride_tensor = st
            except Exception:
                anchor_points = None
                stride_tensor = None

            # extract raw per-detection 3d vectors aligned to kept indices
            # obj_3d_raw = self.get_obj_3d(extra3d_cat, idxs)

            # Try to decode using head.decode_extra3d if available
            if head_module is not None and hasattr(head_module, "decode_extra3d"):
                try:
                    # decoder may return:
                    #  - {'raw': Tensor(B,N,n3d), 'base_decoded': {...}, 'faces_decoded': {...}}
                    #  - or only {'base_decoded': {...}, 'faces_decoded': {...}}
                    #  - or just {'raw': Tensor(...)}
                    decoded_whole = head_module.decode_extra3d(
                        extra3d_cat, anchor_points=anchor_points, stride_tensor=stride_tensor, as_numpy=False
                    )
                except Exception:
                    decoded_whole = None

                if decoded_whole:
                    # Always attempt to align any decoded fields into a consistent structure.
                    # Align raw if provided
                    decoded_raw = decoded_whole.get("raw", None)
                    base_decoded_dict = decoded_whole.get("base_decoded", None)
                    faces_decoded_dict = decoded_whole.get("faces_decoded", None)

                    aligned = {}
                    # Align raw into list-of-per-image tensors if present
                    if decoded_raw is not None:
                        aligned_raw = self.get_obj_3d(decoded_raw, idxs)  # list per image
                        aligned["raw"] = aligned_raw
                    # Align per-field decoded dicts (each field a (B,N,...) tensor or array)
                    if base_decoded_dict or faces_decoded_dict:
                        aligned_fields = self._align_decoded_dicts(base_decoded_dict, faces_decoded_dict, idxs)
                        # _align_decoded_dicts returns {'base_decoded': {k: [per_image tensors]}, 'faces_decoded': {k: [...]}}
                        aligned.update(aligned_fields)

                    # Use aligned dict as obj_3d_decoded; downstream code will handle dict vs list
                    obj_3d_decoded = aligned if aligned else None

        # Build Results objects
        results = self.construct_results(preds_kept, img, orig_imgs, **kwargs)

        # Attach feats
        if obj_feats is not None:
            for r, f in zip(results, obj_feats):
                r.feats = f

        # Attach raw 3D vectors (detached, CPU numpy)
        if obj_3d_decoded is not None and 'raw' in obj_3d_decoded:
            obj_3d_raw = obj_3d_decoded['raw']
        if obj_3d_raw is not None:
            for r, arr in zip(results, obj_3d_raw):
                if isinstance(arr, torch.Tensor):
                    if arr.numel() == 0:
                        r.extra3d_raw = np.zeros((0,))
                    else:
                        r.extra3d_raw = arr.detach().cpu().numpy()
                elif isinstance(arr, (list, tuple, np.ndarray)):
                    r.extra3d_raw = np.asarray(arr)
                else:
                    r.extra3d_raw = None

                # attempt to split raw into faces/base based on head attributes if possible
                if isinstance(r.extra3d_raw, np.ndarray) and r.extra3d_raw.ndim == 2:
                    n3d = r.extra3d_raw.shape[1]
                    # Get head params if available
                    num_faces = getattr(head_module, "num_faces", None) if head_module is not None else None
                    face_dims_out = getattr(head_module, "face_dims_out", None) if head_module is not None else None
                    base_dims = getattr(head_module, "base_dims", None) if head_module is not None else None
                    if num_faces is None:
                        num_faces = 4
                    if face_dims_out is None:
                        face_dims_out = 6
                    if base_dims is None:
                        base_dims = n3d - num_faces * face_dims_out
                    if n3d == num_faces * face_dims_out + base_dims and base_dims > 0:
                        try:
                            faces = r.extra3d_raw[:, : num_faces * face_dims_out].reshape(-1, num_faces, face_dims_out)
                            base = r.extra3d_raw[:, num_faces * face_dims_out :]
                            r.faces3d_raw = faces
                            r.base3d_raw = base
                        except Exception:
                            r.faces3d_raw = None
                            r.base3d_raw = None
                    else:
                        r.faces3d_raw = None
                        r.base3d_raw = None
                else:
                    r.faces3d_raw = None
                    r.base3d_raw = None

        # Attach decoded 3D values (if decoder produced any aligned fields)
        if obj_3d_decoded is not None:
            # obj_3d_decoded is a dict possibly containing:
            #  - 'raw': list of per-image tensors
            #  - 'base_decoded': {field: list_of_per_image_tensors}
            #  - 'faces_decoded': {field: list_of_per_image_tensors}
            # Handle 'raw' first (attach as extra3d)
            raw_aligned = obj_3d_decoded.get("raw", None)
            base_fields = obj_3d_decoded.get("base_decoded", {}) or {}
            face_fields = obj_3d_decoded.get("faces_decoded", {}) or {}

            # Attach raw decoded if present
            if raw_aligned is not None:
                for r, arr in zip(results, raw_aligned):
                    if isinstance(arr, torch.Tensor):
                        r.extra3d = arr.detach().cpu().numpy() if arr.numel() else np.zeros((0,))
                    elif isinstance(arr, (list, tuple, np.ndarray)):
                        r.extra3d = np.asarray(arr)
                    else:
                        r.extra3d = None

            # Attach per-field base_decoded/face_decoded if present
            if base_fields or face_fields:
                # base_fields : field -> [per-image tensors]
                # face_fields : field -> [per-image tensors]
                # attach per result
                B = len(results)
                # Build per-image dicts for base and faces
                # base_per_image = {k: self._align_and_extract_tensor(v, idxs) for k, v in base_fields.items()}
                # face_per_image = {k: self._align_and_extract_tensor(v, idxs) for k, v in face_fields.items()}

                base_per_image = base_fields
                face_per_image = face_fields

                for i, r in enumerate(results):
                    # r.base_decoded : dict of field -> numpy array per det (or empty arrays)
                    r.base_decoded = {}
                    r.faces_decoded = {}
                    for k, arr_list in base_per_image.items():
                        val = arr_list[i]
                        if isinstance(val, torch.Tensor):
                            r.base_decoded[k] = val.detach().cpu().numpy() if val.numel() else np.zeros((0,))
                        else:
                            r.base_decoded[k] = np.asarray(val)
                    for k, arr_list in face_per_image.items():
                        val = arr_list[i]
                        if isinstance(val, torch.Tensor):
                            r.faces_decoded[k] = val.detach().cpu().numpy() if val.numel() else np.zeros((0,))
                        else:
                            r.faces_decoded[k] = np.asarray(val)

        return results

    def _align_and_extract_tensor(self, tensor_BNx, idxs):
        """
        Align a tensor with shape (B, N, ...) to per-image kept indices.
        Returns list length B where each element is tensor (num_kept, ...)
        Accepts torch.Tensor or numpy array.
        """
        # Convert numpy to tensor for consistent indexing
        is_numpy = False
        if isinstance(tensor_BNx, np.ndarray):
            is_numpy = True
            tensor_BNx = torch.from_numpy(tensor_BNx)
        if not torch.is_tensor(tensor_BNx):
            # unsupported type -> return list of empty lists
            return [[] for _ in idxs]

        out = []
        for per_img_vals, keep in zip(tensor_BNx.unbind(0), idxs):
            if keep.shape[0]:
                try:
                    sel = per_img_vals[keep].contiguous()
                    out.append(sel)
                except Exception:
                    # fallback for different layout
                    try:
                        sel = per_img_vals[keep.cpu().numpy()]
                        out.append(sel)
                    except Exception:
                        out.append(torch.zeros((0, *per_img_vals.shape[1:]), dtype=per_img_vals.dtype, device=per_img_vals.device))
            else:
                out.append(torch.zeros((0, *per_img_vals.shape[1:]), dtype=per_img_vals.dtype, device=per_img_vals.device))
        return out

    def _align_decoded_dicts(self, base_decoded_dict, faces_decoded_dict, idxs):
        """
        Given base_decoded and faces_decoded dicts (field -> tensor shaped (B,N,...)),
        produce a combined dict { 'base_decoded': {field: [per-image aligned arrays]...},
                               'faces_decoded': {field: [per-image aligned arrays]...} }
        Each per-image entry is a tensor (num_dets, ...) on same device as input.
        """
        out = {"base_decoded": {}, "faces_decoded": {}}
        if base_decoded_dict:
            for k, v in base_decoded_dict.items():
                out["base_decoded"][k] = self._align_and_extract_tensor(v, idxs)
        if faces_decoded_dict:
            for k, v in faces_decoded_dict.items():
                out["faces_decoded"][k] = self._align_and_extract_tensor(v, idxs)
        return out

    def get_obj_feats(self, feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def get_obj_3d(self, extra3d_cat, idxs):
        """
        Extract per-detection 3D vectors aligned with kept indices after NMS.

        Args:
            extra3d_cat (torch.Tensor): (bs, n3d, N) 3D outputs aligned with detection grid/anchors before NMS.
            idxs (list[torch.Tensor]): Per-image LongTensor indices of kept detections from NMS.

        Returns:
            list[torch.Tensor | list]: For each image, a tensor of shape (num_dets, n3d) on the same device as extra3d_cat.
        """
        out = []
        # if input is list of per-level tensors rather than a single concatenated tensor, try to concat first
        if isinstance(extra3d_cat, (list, tuple)):
            # assume list of (B, n3d, H, W) per level -> flatten to (B, n3d, N)
            try:
                bs = extra3d_cat[0].shape[0]
                n3d = extra3d_cat[0].shape[1]
                tensors = [e.reshape(bs, n3d, -1) for e in extra3d_cat]
                extra3d_cat = torch.cat(tensors, dim=2)
            except Exception:
                # fallback: not concatenable -> return empty
                return [ [] for _ in idxs ]

        for per_img_3d, keep in zip(extra3d_cat, idxs):
            if keep.shape[0]:
                # per_img_3d: (n3d, N) -> select columns
                # If per_img_3d is (n3d, N), we prefer (n3d, N) indexing; but usually it's (n3d, N) or (n3d, total_anchors)
                # Ensure shape is (n3d, N)
                if per_img_3d.ndim == 3:  # unlikely, but guard
                    per_img_3d = per_img_3d.squeeze(0)
                # select and transpose to (num_dets, n3d)
                try:
                    sel = per_img_3d[keep].contiguous()
                    out.append(sel)
                except Exception:
                    out.append([])
            else:
                out.append([])
        return out

    def construct_results(self, preds, img, orig_imgs):
        """
        Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        # return [
        #     self.construct_result(pred, img, orig_img, img_path)
        #     for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        # ]
    
        results = []
        batch_paths = self.batch[0] if hasattr(self, "batch") and len(self.batch) > 0 else [None] * len(orig_imgs)
        for idx, (pred, orig_img, img_path) in enumerate(zip(preds, orig_imgs, batch_paths)):
            results.append(self.construct_result(pred, img, orig_img, img_path, idx))
        return results

    def construct_result(self, pred, img, orig_img, img_path, batch_index: int = 0):
        """
        Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        # pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
    
        # If there are no detections
        if pred is None or pred.shape[0] == 0:
            return Results(orig_img, path=img_path, names=self.model.names, boxes=np.zeros((0, 6)))

        # Determine ROI metadata for this batch index
        batch_rois = getattr(self, "_batch_rois", None)
        batch_full_shapes = getattr(self, "_batch_full_shapes", None)

        roi = None
        full_shape = None
        if batch_rois is not None and batch_index < len(batch_rois):
            roi = batch_rois[batch_index]
        if batch_full_shapes is not None and batch_index < len(batch_full_shapes):
            full_shape = batch_full_shapes[batch_index]

        # If ROI exists, target for scaling is the crop shape; otherwise target is orig_img.shape
        if roi is not None:
            x1, y1, x2, y2 = (int(v) for v in roi)
            crop_h = y2 - y1
            crop_w = x2 - x1
            # scale from model input shape -> cropped image (pixels)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], (crop_h, crop_w))
            # shift to full-image coordinates by adding roi offsets
            pred[:, 0] += x1
            pred[:, 2] += x1
            pred[:, 1] += y1
            pred[:, 3] += y1
            # orig_for_plot: prefer to use original full image for plotting if available via self.dataset
            full_img_for_plot = None
            # If predictor dataset provided full images for this batch (loader can expose them), use them
            dataset_full_images = getattr(self, "dataset", None)
            if dataset_full_images is not None and getattr(dataset_full_images, "last_full_images_for_batch", None):
                try:
                    full_img_for_plot = dataset_full_images.last_full_images_for_batch[batch_index]
                except Exception:
                    full_img_for_plot = None
            if full_img_for_plot is None:
                # fallback to provided orig_img (which is the original full image passed into postprocess)
                full_img_for_plot = orig_img
            return Results(full_img_for_plot, path=img_path, names=self.model.names, boxes=pred[:, :6])

        else:
            # No ROI: keep original behavior (scale directly to orig_img.shape)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
