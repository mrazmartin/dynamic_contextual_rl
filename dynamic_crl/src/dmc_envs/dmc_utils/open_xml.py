import mujoco
import mujoco.viewer
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / ".." / "env_assets" / "unitree_go2"
_XML = _ASSETS_DIR / "scene_weight.xml"

model = mujoco.MjModel.from_xml_path(str(_XML))
data  = mujoco.MjData(model)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
