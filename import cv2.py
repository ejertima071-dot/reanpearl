from pathlib import Path

import cv2
import numpy as np


def create_drawing_animation(
    image_path,
    output_video_path="drawing_animation.mp4",
    duration_sec=6,
    fps=30,
    preview=True,
):
    if duration_sec <= 0 or fps <= 0:
        raise ValueError("duration_sec and fps must be greater than zero")

    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = img.shape[:2]
    max_dim = 800
    scale = min(1.0, max_dim / float(max(height, width)))
    resized_width = max(2, int(width * scale) // 2 * 2)
    resized_height = max(2, int(height * scale) // 2 * 2)
    if (resized_width, resized_height) != (width, height):
        img = cv2.resize(
            img,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(
        contours, key=lambda contour: cv2.arcLength(contour, False), reverse=True
    )

    total_segments = sum(max(len(contour) - 1, 0) for contour in contours)
    if total_segments == 0:
        raise ValueError(f"No drawable outlines found in {image_path}")

    output_video_path = Path(output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise OSError(f"Could not create video: {output_video_path}")

    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    drawing_frames = max(1, round(duration_sec * fps))
    contour_index = 0
    point_index = 1
    drawn_segments = 0

    print("Drawing the image, then revealing the full-color photo...")
    try:
        for frame_index in range(drawing_frames):
            target_segments = (
                total_segments * (frame_index + 1) + drawing_frames - 1
            ) // drawing_frames
            while drawn_segments < target_segments:
                while (
                    contour_index < len(contours)
                    and point_index >= len(contours[contour_index])
                ):
                    contour_index += 1
                    point_index = 1
                if contour_index >= len(contours):
                    break

                contour = contours[contour_index]
                pt1 = tuple(int(value) for value in contour[point_index - 1][0])
                pt2 = tuple(int(value) for value in contour[point_index][0])
                cv2.line(canvas, pt1, pt2, (45, 45, 45), 1, cv2.LINE_AA)
                point_index += 1
                drawn_segments += 1

            out.write(canvas)
            if preview:
                cv2.imshow("Drawing Animation - press Q to close preview", canvas)
                cv2.waitKey(1)

        transition_frames = max(1, fps)
        for frame_index in range(transition_frames):
            photo_alpha = (frame_index + 1) / transition_frames
            frame = cv2.addWeighted(canvas, 1 - photo_alpha, img, photo_alpha, 0)
            out.write(frame)
            if preview:
                cv2.imshow("Drawing Animation - press Q to close preview", frame)
                cv2.waitKey(1)

        for _ in range(fps * 2):
            out.write(img)
            if preview:
                cv2.imshow("Drawing Animation - press Q to close preview", img)
                cv2.waitKey(1)
    finally:
        out.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"Animation saved successfully as {output_video_path}")


if __name__ == "__main__":
    image_file = Path(__file__).resolve().parent / "Rean1.jpg"
    video_file = Path(__file__).resolve().parent / "drawing_animation.mp4"
    create_drawing_animation(image_file, video_file)
