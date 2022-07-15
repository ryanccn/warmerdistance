from __future__ import division

import typer
from PIL import ImageColor

import torch
import cv2
import numpy as np

from utilities import letterbox_image

import logging
import math

print = typer.secho

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

log_prefix = typer.style(">", dim=True)

print(f"{log_prefix} Loading models")
logging.getLogger("yolov5").setLevel(logging.WARNING)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", verbose=False)
model.classes = [0]

if torch.cuda.is_available():
    print(f"{log_prefix} Using CUDA", fg="green")
    model.to(torch.device("cuda"))
# elif torch.backends.mps.is_available():
#     print(f"{log_prefix} Warning: using MPS on macOS, may encounter bugs", fg="yellow")
#     model.to(torch.device("mps"))
else:
    print(log_prefix + typer.style(" Using CPU", fg="yellow"))
    model.to(torch.device("cpu"))


def hex_to_bgra(hex_string):
    rgb = ImageColor.getrgb(hex_string)
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]), 255)


def line_drawing(frame, target1, target2, c):
    cv2.line(
        frame,
        (int(target1[0]), int(target1[1])),
        (int(target2[0]), int(target2[1])),
        c,
        2,
    )


def draw_some_cool_lines(frame, pointposi):
    dist = []
    for i, (x0, y0) in enumerate(pointposi):
        for j, (x1, y1) in enumerate(pointposi):
            if i == j:
                continue
            dist.append(int(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)))
    comp = np.mean(dist)
    safe = comp / 3
    for i, (x0, y0) in enumerate(pointposi):
        min_distance = 99999

        for j, (x1, y1) in enumerate(pointposi):
            if i == j:
                continue
            distance = int(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))
            if distance < min_distance:
                min_distance = distance
                finaltarget = (x1, y1)

        if safe <= min_distance <= comp:
            line_drawing(frame, (x0, y0), finaltarget, hex_to_bgra("#FDE047"))
        elif 0 < min_distance < safe:
            line_drawing(frame, (x0, y0), finaltarget, hex_to_bgra("#EF4444"))


def process_frame(frame):
    predictions = model(frame).xyxy[0]

    some_centers = []

    for pred in predictions:
        if pred[4].item() < 0.4:
            continue

        center = (
            round((pred[0].item() + pred[1].item()) / 2),
            round((pred[2].item() + pred[3].item()) / 2),
        )

        some_centers.append(center)

    # print(f"{log_prefix} Drawing lines")
    draw_some_cool_lines(frame, some_centers)

    return frame


app = typer.Typer()


@app.command()
def image(input_file: str, output_file: str):
    print(f"{log_prefix} Processing")

    frame = cv2.imread(input_file)
    processed_frame = process_frame(frame)

    print(f"{log_prefix} Exporting to {typer.style(output_file, bold=True)}")
    cv2.imwrite(output_file, processed_frame)

    print("> Done!", bold=True, fg="green")


@app.command()
def video(input_file: str, output_file: str):
    video_capture = cv2.VideoCapture(input_file)
    target_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frames_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    video_writer = None

    print(f"{log_prefix} Doing good stuff")

    with typer.progressbar(length=frames_total, color=True) as progress:
        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()

            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    output_file,
                    cv2.VideoWriter_fourcc(*"avc1"),
                    target_fps,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )

            if frame_is_read and frame is not None:
                proc = process_frame(frame)

                video_writer.write(proc)
                progress.update(1)

            else:
                print("! Could not read the frame.", fg="red", bold=True)
                break

    video_capture.release()
    video_writer.release()

    print("> Done!", bold=True, fg="green")


if __name__ == "__main__":
    app()