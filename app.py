"""Fast text to segmentation with yolo-world and efficient-vit sam."""
import os

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from inference.models import YOLOWorld

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model

# Load models.
yolo_world = YOLOWorld(model_id="yolo_world/l")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = EfficientViTSamPredictor(
    create_sam_model(name="xl1", weight_url="xl1.pt").to(device).eval()
)

# Load annotators.
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()


def detect(
    image: np.ndarray,
    query: str,
    confidence_threshold: float,
    nms_threshold: float,
) -> np.ndarray:
    # Preparation.
    categories = [category.strip() for category in query.split(",")]
    yolo_world.set_classes(categories)
    print("categories:", categories)

    # Object detection.
    results = yolo_world.infer(image, confidence=confidence_threshold)
    detections = sv.Detections.from_inference(results).with_nms(
        class_agnostic=True, threshold=nms_threshold
    )
    print("detected:", detections)

    # Segmentation.
    sam.set_image(image, image_format="RGB")
    masks = []
    for xyxy in detections.xyxy:
        mask, _, _ = sam.predict(box=xyxy, multimask_output=False)
        masks.append(mask.squeeze())
    detections.mask = np.array(masks)
    print("masks shaped as", detections.mask.shape)

    # Annotation.
    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    labels = [
        f"{categories[class_id]}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)


app = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="input image"),
        gr.Text(info="you can input multiple words with comma (,)"),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.3,
            step=0.01,
            interactive=True,
            label="Confidence Threshold",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.01,
            interactive=True,
            label="NMS Threshold",
        ),
    ],
    outputs=gr.Image(type="pil", label="output image"),
    allow_flagging="never",
    title="Fast Text to Segmentation with YOLO-World + EfficientViT SAM",
    description="""
    ## Core components
    ### YOLO-World
    [YOLO-World](https://github.com/AILab-CVC/YOLO-World) is an open-vocabulary object detection model with high efficiency.
    On the challenging LVIS dataset, YOLO-World achieves 35.4 AP with 52.0 FPS on V100,
    which outperforms many state-of-the-art methods in terms of both accuracy and speed.

    ### EfficientViT SAM
    [EfficientViT SAM](https://github.com/mit-han-lab/efficientvit) is a new family of accelerated segment anything models.
    Thanks to the lightweight and hardware-efficient core building block,
    it delivers 48.9× measured TensorRT speedup on A100 GPU over SAM-ViT-H without sacrificing performance.

    ## Demo especially powered by
    Roboflow's [inference](https://github.com/roboflow/inference) and [supervision](https://github.com/roboflow/supervision).

    ## Example images came from
    [Segment Anything Demo](https://segment-anything.com/demo) and [Unsplash](https://unsplash.com/).
    """,
    examples=[
        [
            os.path.join(os.path.dirname(__file__), "examples/livingroom.jpg"),
            "table, lamp, dog, sofa, plant, clock, carpet, frame on the wall",
            0.05,
            0.5
        ],
        [
            os.path.join(os.path.dirname(__file__), "examples/cat_and_dogs.jpg"),
            "cat, dog",
            0.2,
            0.5
        ],
    ],
)


if __name__ == "__main__":
    os.system("make model")
    app.launch(server_name="0.0.0.0")
