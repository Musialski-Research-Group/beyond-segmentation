import os
import sys

ultralytics_path = os.path.abspath(os.path.join(os.getcwd(), "ultralytics"))
sys.path.insert(0, ultralytics_path)

from ultralytics.models.yolo import YOLO

model_dir = "train"
in_dir = "../data/yolo/test/images"
out_dir = f"runs/infer_{model_dir}"

# Number of images processed at a time
BATCH_SIZE = 16

# Load model
model = YOLO(f"runs/{model_dir}/weights/best.pt")

# Create output directory
os.makedirs(out_dir, exist_ok=True)

# Get image filenames
images = [
    filename for filename in os.listdir(in_dir)
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
]

print(f"Found {len(images)} images")
print(f"Processing in batches of {BATCH_SIZE}")

# Process dataset in batches
for start in range(0, len(images), BATCH_SIZE):

    end = min(start + BATCH_SIZE, len(images))
    batch_images = images[start:end]

    print(f"Processing images {start + 1}-{end} of {len(images)}")

    # Run inference on current batch
    batch_paths = [
        os.path.join(in_dir, filename)
        for filename in batch_images
    ]

    results = model.predict(
        batch_paths,
        batch=BATCH_SIZE,
        stream=False
    )

    # Save results
    for filename, result in zip(batch_images, results):

        basename = os.path.splitext(filename)[0]

        # Save image with labels
        result.save(
            filename=os.path.join(out_dir, basename + "_labels.jpg"),
            line_width=1,
            font_size=3
        )

        # Save image without labels
        result.save(
            filename=os.path.join(out_dir, filename),
            labels=False,
            conf=False
        )

        # Save detection coordinates
        result.save_txt(
            txt_file=os.path.join(out_dir, basename + ".txt"),
            save_conf=True
        )

    # Explicitly release batch results
    del results
    del batch_paths

    print(f"Finished batch {start // BATCH_SIZE + 1}")

print("Inference completed!")
