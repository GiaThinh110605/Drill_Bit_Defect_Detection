import os
import shutil

# Paths
data_after_handle = '/Users/mac/Detect_Drill_Bit/Data_After_Handle'
hard_review_images = '/Users/mac/Detect_Drill_Bit/hard_review/images'
hard_review_labels = '/Users/mac/Detect_Drill_Bit/hard_review/labels'

# Target directories
target_images = os.path.join(data_after_handle, 'images')
target_labels = os.path.join(data_after_handle, 'labels')

# Ensure images directory exists
if not os.path.exists(target_images):
    os.makedirs(target_images)

# Copy images from hard_review
print("Copying images...")
image_count = 0
for filename in os.listdir(hard_review_images):
    if filename.endswith('.jpg'):
        src = os.path.join(hard_review_images, filename)
        dst = os.path.join(target_images, filename)
        shutil.copy2(src, dst)
        image_count += 1

print(f"Copied {image_count} images")

# Clear existing labels (hashed filenames)
print("Clearing existing labels...")
for filename in os.listdir(target_labels):
    os.remove(os.path.join(target_labels, filename))

# Copy labels from hard_review with original filenames
print("Copying labels...")
label_count = 0
for filename in os.listdir(hard_review_labels):
    if filename.endswith('.txt'):
        src = os.path.join(hard_review_labels, filename)
        dst = os.path.join(target_labels, filename)
        shutil.copy2(src, dst)
        label_count += 1

print(f"Copied {label_count} labels")

# Create data.yaml
data_yaml_content = """path: /Users/mac/Detect_Drill_Bit/Data_After_Handle
train: images
val: images

nc: 5
names:
  0: Broken
  1: Chipped
  2: Scratched
  3: Severe_Rust
  4: Tip_Wear
"""

data_yaml_path = os.path.join(data_after_handle, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print(f"Created data.yaml at {data_yaml_path}")

print(f"\nFinal structure:")
print(f"Images: {len(os.listdir(target_images))}")
print(f"Labels: {len(os.listdir(target_labels))}")
print(f"Classes: {os.path.join(data_after_handle, 'classes.txt')}")
print(f"Config: {data_yaml_path}")
