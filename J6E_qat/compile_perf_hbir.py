# Copyright (c) Horizon Robotics. All rights reserved.
# This script requires user define key "hbir_compiler" in config file, whose
# type must be "HbirCompiler".
# If "hbir_compiler" is not defined, we will try to gather required info
# from "compile_cfg".
# All arguments of HbirCompiler.__init__ can be modified with command line
# arguments.

import argparse
import logging
import os
import typing
from collections import defaultdict
from distutils.version import LooseVersion

from hbdk4.compiler import compile, convert, hbm_perf, load, save  # noqa: E402
from horizon_plugin_pytorch import March
from horizon_plugin_pytorch.quantization.hbdk4 import (  # noqa: E402
    get_hbir_input_flattener,
)

from hat.registry import OBJECT_REGISTRY, RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
from hat.utils.setup_env import setup_args_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="./config/fs_model_resnet18.py",
        required=False,
        help="Config file path",
    )
    parser.add_argument(
        "--only-export",
        action="store_true",
        help="Whether only export pre compile hbir",
    )

    init_args = typing.get_type_hints(HbirCompiler.__init__)
    init_args.pop("return")
    for name, type in init_args.items():
        parser.add_argument("--" + name, type=type, required=False)

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


@OBJECT_REGISTRY.register
class HbirCompiler:
    """Compile mlir.Module and estimate performance.

    Args:
        model_path: Quantized hbir module file path.
        out_path: A filesystem path to save hbm. Must ends with ".hbm".
        march: BPU march, options are "bayes", "b25", "b25e", "b30", "b30g".
        opt: Optimization level. Defaults to 2.
        jobs: Number of threads launched during compiler optimization.
            Defaults to 4.
        max_time_per_fc:
            Set maximum time constraint (unit:us) for per funccall.
        debug: Set Whether to contain debug info in HBM.
        input_source:
            A structure of strings has the same structure with model input.
            Supported options are "ddr" and "pyramid".
            Defaults to None.
        transpose_dim:
            For transpose inputs or outputs node.
        split_dim:
            For split pyramid inputs.
        only_export:
            Whether only export compile hbir.
        layer_details:
            Whether to generate layer details.
        hbdk3_compatible_mode:
            Set whether to compile in hbdk3 compatible mode.
    """

    logger = None

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    def __init__(
        self,
        model_path: str,
        out_path: str,
        march: str,
        opt: int,
        jobs: int,
        max_time_per_fc: float,
        debug: bool,
        input_source=None,
        transpose_dim: dict = None,
        split_dim: dict = None,
        only_export: bool = False,
        layer_details: bool = False,
        hbdk3_compatible_mode: bool = False,
    ) -> None:
        self.model = load(model_path)
        self.out_path = out_path
        self.march = march
        self.opt = opt
        self.jobs = jobs
        self.debug = debug
        self.max_time_per_fc = max_time_per_fc

        if input_source is None:
            self.input_source = input_source
        else:
            if isinstance(input_source, str):
                input_source = input_source.split(",")
            self.input_source = get_hbir_input_flattener(self.model)(
                input_source
            )
            self.input_source = [
                source.strip() for source in self.input_source
            ]
        if transpose_dim is None:
            self.transpose_dim = defaultdict(dict)
        else:
            self.transpose_dim = transpose_dim
        if split_dim is None:
            self.split_dim = defaultdict(dict)
        else:
            self.split_dim = split_dim

        self.only_export = only_export
        self.layer_details = layer_details
        self.hbdk3_compatible_mode = hbdk3_compatible_mode

    def set_input_source(
        self, input_node, source, transpose_dim=None, split_dim=None
    ):
        if source == "ddr":
            self.logger.info(
                f"Input node : {input_node.name}, compiled with ddr!!"
                f"transpose_dim: {transpose_dim}"
            )
            # self.logger.info(f"{source}: {transpose_dim}")
            if transpose_dim is not None:
                input_node = input_node.insert_transpose(transpose_dim)
            else:
                pass
        elif source == "pyramid":
            self.logger.info(
                f"Input node : {input_node.name}, compiled with pyramid!!"
            )
            input_node = input_node.insert_transpose([0, 3, 1, 2])
            input_node = input_node.insert_image_preprocess(
                mode=None, divisor=1, mean=[0, 0, 0], std=[256, 256, 256]
            )
            input_node = input_node.insert_image_convert("nv12")
        else:
            msg = "Unsupported input source {}".format(source)
            self.logger.error(msg)
            raise ValueError(msg)

    def set_output_layout(self, output_node, transpose_dim):
        self.logger.info(
            f"Output node : {output_node.name}, transpose_dim: {transpose_dim}"
        )
        output_node = output_node.insert_transpose(transpose_dim)

    def set_output_split(self, output_node, split_dim):
        self.logger.info(f"output split_dim: {split_dim}")
        output_node.insert_split(split_dim)

    def remove_quant_dequant(self, model):
        from hbdk4.compiler import version

        if LooseVersion(version.VERSION) >= LooseVersion("4.1.3"):
            model.functions[0].remove_io_op(["Quantize", "Dequantize"])
        elif LooseVersion(version.VERSION) > LooseVersion("4.0.13"):
            self.remove_cpu_qcast_dcast(model[0].inputs)
            self.remove_cpu_qcast_dcast(model[0].outputs)
        else:
            from hbdk4.compiler.toolbox import remove_function_io  # noqa: E402

            remove_function_io(model, "qnt.quantize")
            remove_function_io(model, "qnt.dequantize")

    def remove_cpu_qcast_dcast(self, args):
        for arg in args:
            removable, diagnostic = arg.is_removable
            if removable:
                attached_op = arg.get_attached_op[0]
                if attached_op.type == "hbtl.call":
                    schema = attached_op.schema
                    if schema.namespace == "quant" and schema.signature in [
                        "qcast",
                        "dcast",
                    ]:
                        removed, diagnostic = arg.remove_attached_op()
                        if removed is False:
                            raise RuntimeError(
                                "Remove quant/dequant failed, reason is:"
                                "\n{}".format(diagnostic)
                            )

    def __call__(self):
        func = self.model.functions[0]
        if self.input_source is not None:
            inputs_split_dim = self.split_dim.get("inputs", defaultdict(dict))

            # split pyramid inputs
            for idx, value in inputs_split_dim.items():
                node_tmp = func.inputs[int(idx)]
                node_tmp.insert_split(value[0])
                for _ in range(value[1] - 1):
                    self.input_source.insert(int(idx), "pyramid")
            # transpose dim and insert pyramid for inputs.
            offset = 0
            inputs_transpose_dim = self.transpose_dim.get(
                "inputs", defaultdict(dict)
            )
            global_transpose_dim = inputs_transpose_dim.get("global", None)
            num_inputs = len(func.inputs)

            diff_num = num_inputs - len(self.input_source)

            if diff_num > 0:
                self.input_source = self.input_source + ["ddr"] * diff_num

            for idx in range(num_inputs):
                node = func.inputs[idx + offset]
                source = self.input_source[idx]

                if global_transpose_dim is not None:
                    node_transpose_dim = global_transpose_dim
                else:
                    node_transpose_dim = inputs_transpose_dim.get(
                        str(idx - offset + 1), None
                    )
                self.set_input_source(node, source, node_transpose_dim)
                if source == "pyramid":
                    offset = offset + 1

        # transpose outputs dim
        outputs_transpose_dim = self.transpose_dim.get(
            "outputs", defaultdict(dict)
        )

        if len(outputs_transpose_dim):
            global_transpose_dim = outputs_transpose_dim.get("global", None)
            for idx, node in enumerate(func.outputs):
                if global_transpose_dim is not None:
                    node_transpose_dim = global_transpose_dim
                else:
                    node_transpose_dim = outputs_transpose_dim.get(
                        str(idx), None
                    )

                if node_transpose_dim is not None:
                    self.set_output_layout(node, node_transpose_dim)

        dst_dir = os.path.dirname(self.out_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        self.logger.info(
            format_msg(
                "Start convert hbir",
                MSGColor.GREEN,
            )
        )
        self.model = convert(self.model, self.march)
        self.logger.info(
            format_msg(
                "Convert Succeed",
                MSGColor.GREEN,
            )
        )
        self.remove_quant_dequant(self.model)
        func = self.model.functions[0]
        save(self.model, os.path.join(dst_dir, "compile_hbir.bc"))

        self.logger.info(
            format_msg(
                "Compiled hbir saved: %s"
                % os.path.join(dst_dir, "compile_hbir.bc"),
                MSGColor.GREEN,
            )
        )

        if not self.only_export:
            compile(
                self.model,
                path=self.out_path,
                march=self.march,
                opt=self.opt,
                jobs=self.jobs,
                max_time_per_fc=self.max_time_per_fc,
                debug=(self.debug or self.layer_details),
                hbdk3_compatible_mode=self.hbdk3_compatible_mode,
            )

            self.logger.info(
                format_msg(
                    "Compiled model: %s" % self.out_path, MSGColor.GREEN
                )
            )
            if self.layer_details:
                hbm_perf(self.out_path, dst_dir)


def compile_perf_hbir(
    cfg_file,
    compile_args,
):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger.info("=" * 50 + "BEGIN HBLR COMPILE" + "=" * 50)

    OPTMAPPING = {
        "O0": 0,
        "O1": 1,
        "O2": 2,
    }
    default_config = dict(  # noqa: C408
        type="HbirCompiler",
        march=March.NASH_E,
        opt=2,
        jobs=32,
        debug=False,
        max_time_per_fc=0.0,
        only_export=args.only_export,
        out_path="./model.hbm",
    )

    if cfg_file is not None:
        cfg = Config.fromfile(cfg_file)
        if "hbir_compiler" in cfg:
            hbir_compiler = cfg.hbir_compiler
        else:
            logger.warning(
                format_msg(
                    "Do not find hbir_compiler in config file, "
                    "trying to collect need info automaticlly.",
                    MSGColor.RED,
                )
            )

            if "compile_cfg" in cfg:
                hbir_compiler = dict(  # noqa: C408
                    march=cfg.compile_cfg["march"],
                    out_path=cfg.compile_cfg["hbm"],
                    input_source=cfg.compile_cfg["input_source"],
                    transpose_dim=cfg.compile_cfg.get(
                        "transpose_dim", defaultdict(dict)
                    ),
                    split_dim=cfg.compile_cfg.get(
                        "split_dim", defaultdict(dict)
                    ),
                    opt=OPTMAPPING[cfg.compile_cfg.get("opt", "O2")],
                    layer_details=cfg.compile_cfg.get("layer_details", False),
                    hbdk3_compatible_mode=cfg.compile_cfg.get(
                        "hbdk3_compatible_mode", False
                    ),
                )
                if "hbir_save_dir" in cfg:
                    model_path = os.path.join(cfg.hbir_save_dir, "qat.bc")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(cfg.ckpt_dir, "qat.mlir")
                    if not os.path.exists(model_path):
                        msg = "Model file {} do not exist.".format(
                            os.path.join(cfg.ckpt_dir, "qat.bc/mlir")
                        )

                    hbir_compiler["model_path"] = model_path
                elif "ckpt_dir" in cfg:
                    model_path = os.path.join(cfg.ckpt_dir, "qat.bc")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(cfg.ckpt_dir, "qat.mlir")
                    if not os.path.exists(model_path):
                        msg = "Model file {} do not exist.".format(
                            os.path.join(cfg.ckpt_dir, "qat.bc/mlir")
                        )

                    hbir_compiler["model_path"] = model_path
                else:
                    msg = "Do not find ckpt_dir in config file."
                    logger.error(format_msg(msg, MSGColor.RED))
                    raise ValueError(msg)
            else:
                msg = "Do not find compile_cfg in config file."
                logger.error(format_msg(msg, MSGColor.RED))
                raise ValueError(msg)

        default_config.update(hbir_compiler)

    compile_kwargs = vars(compile_args)
    if cfg_file is not None:
        compile_kwargs.pop("config")
    for k, v in compile_kwargs.items():
        if v is not None:
            default_config[k] = v

    logger.info("Compile config:")
    logger.info(default_config)

    HbirCompiler.set_logger(logger)
    with RegistryContext():
        hbir_compiler: HbirCompiler = build_from_registry(default_config)
        hbir_compiler()

    logger.info("=" * 50 + "END HBLR COMPILE" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    compile_perf_hbir(
        cfg_file=args.config,
        compile_args=args,
    )