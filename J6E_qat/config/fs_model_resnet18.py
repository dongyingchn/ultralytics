import os
from horizon_plugin_pytorch.march import March

ckpt_dir = "./compile"
march = March.NASH_E
task_name = "fs_resnet18"
compile_dir = ckpt_dir
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt="O2",
)