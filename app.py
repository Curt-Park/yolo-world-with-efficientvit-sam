import os
from functools import lru_cache

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from mmdet.visualization import DetLocalVisualizer
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from PIL import Image
from torchvision.ops import nms

YOLO_CONFIG = os.path.join(
    "configs",
    "yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
)
YOLO_WEIGHT = "yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth"


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


@lru_cache
def load_yolo_world_runner(config: str, checkpoint: str) -> Runner:
    cfg = Config.fromfile(config)
    cfg.load_from = checkpoint
    cfg.work_dir = os.path.join(
        "work_dirs", os.path.splitext(os.path.basename(config))[0]
    )
    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    if not torch.cuda.is_available():
        runner.model.cpu()
    runner.model.eval()
    return runner


def segment(
    image: Image.Image,
    query: str,
    max_num_boxes: int,
    score_thr: float,
    nms_thr: float,
    image_path: str = os.path.join("work_dirs", "demo.png"),
) -> Image.Image:
    yolo_runner = load_yolo_world_runner(YOLO_CONFIG, YOLO_WEIGHT)
    texts = [[t.strip()] for t in query.split(",")] + [[" "]]
    print("texts: ", texts)

    image.save(image_path)
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = yolo_runner.pipeline(data_info)
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = yolo_runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta["classes"] = [t[0] for t in texts]
    visualizer.add_datasample(
        "image",
        np.array(image),
        output,
        draw_gt=False,
        out_file=image_path,
        pred_score_thr=score_thr,
    )
    return Image.open(image_path)


app = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="pil", label="input image"),
        gr.Text(info="you can input multiple words with comma (,)"),
        gr.Slider(
            minimum=1,
            maximum=300,
            value=100,
            step=1,
            interactive=True,
            label="Maximum Number Boxes",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.05,
            step=0.001,
            interactive=True,
            label="Score Threshold",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.001,
            interactive=True,
            label="NMS Threshold",
        ),
    ],
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
            "strawberry,banana",
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
