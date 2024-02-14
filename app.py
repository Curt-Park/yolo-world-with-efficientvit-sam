import os
import subprocess

import gradio as gr
from PIL import ImageFile


def segment(image_path: str, query: str) -> ImageFile.ImageFile:
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
    subprocess.run(["make", "model"])
    app.launch()
