#!/usr/bin/env python
# -*- coding: ascii -*-

import argparse
import numpy as np
import os
from PIL import Image, ImageFilter
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
from typing import Any

# camera feed dimensions in the stitched image
CAMERA_HEIGHT: int = 96
CAMERA_WIDTH: int = 96

# two-pass upscaling: 4x real-world then 2x classical (96 -> 384 -> 768)
MODEL_4X: str = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
MODEL_2X: str = "caidas/swin2SR-classical-sr-x2-64"

# standard VLM input size
STANDARD_SIZE: tuple[int, int] = (512, 512)


def _run_sr_pass(
    image: Image.Image,
    model: Swin2SRForImageSuperResolution,
    processor: AutoImageProcessor,
) -> Image.Image:
    # run super-resolution inference and convert output back to PIL
    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt").to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        outputs = model(**inputs)
    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=2)
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output, "RGB")


def load_model(
    model_id: str,
    device: str | None = None,
) -> tuple[Swin2SRForImageSuperResolution, AutoImageProcessor]:
    # auto-detect device, load processor and model, set to eval mode
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def main(args: dict[str, Any] = dict()) -> None:
    upscaled = upscale_stitched_image(
        stitched_path=args.get(
            "input",
            os.path.join(
                "results", "completed localisation task", "debug_stitched_image.png"
            ),
        ),
        output_dir=args.get("output_dir", "results"),
        device=args.get("device"),
    )
    print(
        f"upscaled {len(upscaled)} camera images to"
        f" {STANDARD_SIZE[0]}x{STANDARD_SIZE[1]}"
    )


def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Upscale debug stitched camera images using AI super-resolution."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            "results", "completed localisation task", "debug_stitched_image.png"
        ),
        help="path to the stitched debug image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="directory to save upscaled images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to run the model on (default: auto-detect)",
    )
    return vars(parser.parse_args())


def split_stitched_image(
    stitched: Image.Image,
    camera_height: int = CAMERA_HEIGHT,
    camera_width: int = CAMERA_WIDTH,
) -> list[Image.Image]:
    width, height = stitched.size
    if width != camera_width:
        raise ValueError(
            f"stitched image width {width} does not match camera width {camera_width}"
        )
    if height % camera_height != 0:
        raise ValueError(
            f"stitched image height {height} is not divisible by camera height"
            f" {camera_height}"
        )

    # crop each camera tile from the vertically stacked image
    num_cameras = height // camera_height
    images = []
    for i in range(num_cameras):
        top = i * camera_height
        bottom = (i + 1) * camera_height
        images.append(stitched.crop((0, top, camera_width, bottom)))
    return images


def upscale_image(
    image: Image.Image,
    model_4x: Swin2SRForImageSuperResolution,
    processor_4x: AutoImageProcessor,
    model_2x: Swin2SRForImageSuperResolution,
    processor_2x: AutoImageProcessor,
    target_size: tuple[int, int] = STANDARD_SIZE,
) -> Image.Image:
    # pass 1: 4x real-world SR (e.g. 96 -> 384)
    upscaled = _run_sr_pass(image, model_4x, processor_4x)
    # pass 2: 2x classical SR (e.g. 384 -> 768)
    upscaled = _run_sr_pass(upscaled, model_2x, processor_2x)
    # downscale to target (e.g. 768 -> 512)
    upscaled = upscaled.resize(
        (target_size[1], target_size[0]),
        Image.Resampling.LANCZOS,
    )
    # enhance edges
    return upscaled.filter(ImageFilter.EDGE_ENHANCE_MORE)


def upscale_stitched_image(
    stitched_path: str,
    output_dir: str | None = None,
    target_size: tuple[int, int] = STANDARD_SIZE,
    device: str | None = None,
) -> list[Image.Image]:
    # load models, split the stitched image, and upscale each tile
    stitched = Image.open(stitched_path).convert("RGB")
    camera_images = split_stitched_image(stitched)
    model_4x, processor_4x = load_model(model_id=MODEL_4X, device=device)
    model_2x, processor_2x = load_model(model_id=MODEL_2X, device=device)
    upscaled_images = []
    for i, img in enumerate(camera_images):
        upscaled = upscale_image(
            img,
            model_4x,
            processor_4x,
            model_2x,
            processor_2x,
            target_size=target_size,
        )
        upscaled_images.append(upscaled)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"camera_{i}_upscaled.png")
            upscaled.save(save_path)
            print(f"saved upscaled camera {i} to {save_path}")
    return upscaled_images


if __name__ == "__main__":
    main(parse_args())
