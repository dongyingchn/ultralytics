import os
from horizon_plugin_pytorch.march import March

ckpt_dir = "./J6E_qat/compile"
march = March.NASH_E
task_name = "mono3d_resnetj6e"
compile_dir = ckpt_dir
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid"],
    opt="O2",
)