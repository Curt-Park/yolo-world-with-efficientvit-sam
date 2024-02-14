import os
from functools import lru_cache

import gradio as gr
from PIL import ImageFile
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS


YOLO_CONFIG = os.path.join("configs", "yolov8_l_syncbn_fast_8xb16-500e_coco.py")
YOLO_WEIGHT = "yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"


@lru_cache
def load_yolo_world_runner(config: str, checkpoint: str) -> Runner:
    cfg = Config.fromfile(config)
    cfg.load_from = checkpoint
    cfg.work_dir = os.path.join("work_dirs", os.path.splitext(os.path.basename(config))[0])
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    return runner


def segment(image_path: str, query: str) -> ImageFile.ImageFile:
    yolo_runner = load_yolo_world_runner(YOLO_CONFIG, YOLO_WEIGHT)
    return None


app = gr.Interface(
    fn=segment,
    inputs=[gr.Image(type="filepath", label="input image"), "text"],
    outputs="image",
    allow_flagging="never",
    title="Fast Text to Segmentation with Yolo-World + Efficient-Vit SAM",
    examples=[
        [
            os.path.join(os.path.dirname(__file__), "examples/dog.jpg"),
            "dog",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/city.jpg"),
            "building",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/food.jpg"),
            "strawberry",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/horse.jpg"),
            "horse",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/bears.jpg"),
            "bear",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/cats.jpg"),
            "cat",
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/fish.jpg"),
            "fish",
        ],
    ],
)


if __name__ == "__main__":
    os.system("make model")
    app.launch()
