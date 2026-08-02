# Label Studio Workflow Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Dataset Preparation](#dataset-preparation)
4. [Converting to Label Studio JSON Format](#converting-to-label-studio-json-format)
5. [Importing into Label Studio](#importing-into-label-studio)
6. [Exporting and Converting Back](#exporting-and-converting-back)
7. [Troubleshooting](#troubleshooting)

## Installation

### Install Label Studio
```bash
pip install label-studio
```

### Start Label Studio Server
```bash
label-studio start
```

- Label Studio will start at `http://localhost:8080`
- First time: Create an account (username/password)
- Login to access the dashboard

### Stop Label Studio Server
```bash
# Press Ctrl+C in the terminal where Label Studio is running
```

## Basic Usage

### Create a New Project
1. Click "Create Project" on the dashboard
2. Enter project name (e.g., "Drill Bit Defect Detection")
3. Click "Create"

### Configure Labeling Interface
1. Go to "Settings" → "Labeling Interface"
2. Choose "Object Detection" template or use custom configuration
3. Configure labels (classes):
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="Broken" background="red"/>
       <Label value="Chipped" background="blue"/>
       <Label value="Scratched" background="green"/>
       <Label value="Severe_Rust" background="yellow"/>
       <Label value="Tip_Wear" background="purple"/>
     </RectangleLabels>
   </View>
   ```

### Basic Labeling Workflow
1. **Import Data**: Upload images or import JSON with annotations
2. **Label**: Click on images to draw bounding boxes
3. **Submit**: Click "Submit" to save annotations
4. **Review**: Go to "All Tasks" to review labeled data

## Dataset Preparation

### Dataset Structure
Organize your dataset in YOLO format:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── data.yaml
```

### YOLO Label Format
Each label file contains one line per object:
```
class_id center_x center_y width height
```
- `class_id`: Integer class index (0, 1, 2, ...)
- `center_x`, `center_y`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized width and height (0-1)

Example:
```
0 0.5 0.5 0.3 0.4
1 0.7 0.8 0.2 0.1
```

### data.yaml Configuration
```yaml
path: /path/to/dataset
train: images
val: images

nc: 5
names:
  0: Broken
  1: Chipped
  2: Scratched
  3: Severe_Rust
  4: Tip_Wear
```

## Converting to Label Studio JSON Format

### Option 1: Base64 Encoding (Recommended for Local Files)

This method embeds images directly in JSON, avoiding file path issues.

**Python Script:**
```python
import json
import base64
import os

# Configuration
images_dir = '/path/to/dataset/images'
labels_dir = '/path/to/dataset/labels'
output_json = '/path/to/output/labelstudio_import.json'

# Class names
class_names = ['Broken', 'Chipped', 'Scratched', 'Severe_Rust', 'Tip_Wear']

def read_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            yolo_bbox = [float(x) for x in parts[1:5]]
            annotations.append({
                'class_id': class_id,
                'bbox': yolo_bbox
            })
    return annotations

def yolo_to_labelstudio(yolo_bbox):
    center_x, center_y, width, height = yolo_bbox
    x_pct = (center_x - width / 2) * 100
    y_pct = (center_y - height / 2) * 100
    width_pct = width * 100
    height_pct = height * 100
    return {
        "x": x_pct,
        "y": y_pct,
        "width": width_pct,
        "height": height_pct,
        "rotation": 0
    }

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Create Label Studio JSON
tasks = []
for filename in sorted(os.listdir(images_dir)):
    if not filename.endswith('.jpg'):
        continue
    
    # Encode image
    image_path = os.path.join(images_dir, filename)
    base64_data = image_to_base64(image_path)
    
    # Read labels
    label_file = filename.replace('.jpg', '.txt')
    label_path = os.path.join(labels_dir, label_file)
    
    results = []
    if os.path.exists(label_path):
        annotations = read_yolo_label(label_path)
        for i, ann in enumerate(annotations):
            bbox = yolo_to_labelstudio(ann['bbox'])
            results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    **bbox,
                    "rectanglelabels": [class_names[ann['class_id']]]
                }
            })
    
    task = {
        "data": {
            "image": f"data:image/jpeg;base64,{base64_data}"
        },
        "annotations": [
            {
                "result": results
            }
        ]
    }
    tasks.append(task)

# Save JSON
with open(output_json, 'w') as f:
    json.dump(tasks, f, indent=2)

print(f"Created {len(tasks)} tasks in {output_json}")
```

### Option 2: With Predictions (Ground Truth + Model Predictions)

Include both ground truth and model predictions for review.

**Python Script:**
```python
import json
import base64
import os
import pandas as pd

# Configuration
images_dir = '/path/to/dataset/images'
labels_dir = '/path/to/dataset/labels'
predictions_json = '/path/to/predictions.json'
output_json = '/path/to/output/labelstudio_with_predictions.json'

# Class names
class_names = ['Broken', 'Chipped', 'Scratched', 'Severe_Rust', 'Tip_Wear']

# Read predictions (COCO format)
with open(predictions_json, 'r') as f:
    predictions = json.load(f)

# Group predictions by image_id
predictions_by_image_id = {}
for pred in predictions:
    image_id = pred['image_id']
    if image_id not in predictions_by_image_id:
        predictions_by_image_id[image_id] = []
    predictions_by_image_id[image_id].append(pred)

# Create image_id to filename mapping (from your CSV or metadata)
# This depends on your specific dataset structure
image_id_to_filename = {}  # Populate this based on your data

def pixel_to_percentage(bbox, img_width=448, img_height=448):
    x, y, width, height = bbox
    x_pct = (x / img_width) * 100
    y_pct = (y / img_height) * 100
    width_pct = (width / img_width) * 100
    height_pct = (height / img_height) * 100
    return {
        "x": x_pct,
        "y": y_pct,
        "width": width_pct,
        "height": height_pct,
        "rotation": 0
    }

# Create tasks with GT and predictions
tasks = []
for filename in sorted(os.listdir(images_dir)):
    if not filename.endswith('.jpg'):
        continue
    
    # Encode image
    image_path = os.path.join(images_dir, filename)
    base64_data = image_to_base64(image_path)
    
    # Read GT labels
    label_file = filename.replace('.jpg', '.txt')
    label_path = os.path.join(labels_dir, label_file)
    
    gt_results = []
    if os.path.exists(label_path):
        annotations = read_yolo_label(label_path)
        for i, ann in enumerate(annotations):
            bbox = yolo_to_labelstudio(ann['bbox'])
            gt_results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    **bbox,
                    "rectanglelabels": [class_names[ann['class_id']]]
                }
            })
    
    # Get predictions
    pred_results = []
    image_id = image_id_to_filename.get(filename)
    if image_id and image_id in predictions_by_image_id:
        preds = sorted(predictions_by_image_id[image_id], key=lambda x: x['score'], reverse=True)
        for i, pred in enumerate(preds[:10]):  # Top 10 predictions
            bbox = pixel_to_percentage(pred['bbox'])
            pred_results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    **bbox,
                    "rectanglelabels": [class_names[pred['category_id'] - 1]]
                },
                "score": pred['score']
            })
    
    task = {
        "data": {
            "image": f"data:image/jpeg;base64,{base64_data}"
        },
        "annotations": [
            {
                "result": gt_results
            }
        ],
        "predictions": [
            {
                "result": pred_results
            }
        ]
    }
    tasks.append(task)

# Save JSON
with open(output_json, 'w') as f:
    json.dump(tasks, f, indent=2)
```

## Importing into Label Studio

### Step 1: Clean Existing Data (if any)
1. Go to your project
2. Click "Data" tab
3. Select all tasks (click checkbox)
4. Click "Delete" → Confirm

### Step 2: Import JSON
1. Click "Import" button
2. Select "Import JSON"
3. Choose your JSON file (e.g., `labelstudio_import.json`)
4. Click "Import"

### Step 3: Verify Import
1. Go to "Data" tab
2. Check that all tasks are imported
3. Click on a task to verify images and annotations display correctly

### Alternative: Upload Images Separately
If you prefer not to use base64 encoding:

1. **Upload Images:**
   - Click "Import" → "Upload Files"
   - Select images or a zip file
   - Upload

2. **Import Annotations:**
   - Click "Import" → "Import JSON"
   - Use JSON with filenames only (not base64)
   - Label Studio will match filenames

## Exporting and Converting Back

### Export from Label Studio

1. Go to your project
2. Click "Export" button
3. Choose export format:
   - **JSON**: For further processing
   - **YOLO**: For direct use in training
   - **COCO**: For compatibility with other tools

4. Click "Export" and download the file

### Export Structure
Label Studio exports typically contain:
```
export/
├── images/          # Image files
├── labels/          # Label files (hashed filenames)
├── classes.txt      # Class names
└── notes.json       # Metadata
```

### Converting Export Back to YOLO Format

If export has hashed filenames, use this script to restore original names:

```python
import os
import shutil

# Paths
export_dir = '/path/to/labelstudio_export'
original_images_dir = '/path/to/original/images'
original_labels_dir = '/path/to/original/labels'

# Copy original images to export
export_images_dir = os.path.join(export_dir, 'images')
if not os.path.exists(export_images_dir):
    os.makedirs(export_images_dir)

for filename in os.listdir(original_images_dir):
    if filename.endswith('.jpg'):
        src = os.path.join(original_images_dir, filename)
        dst = os.path.join(export_images_dir, filename)
        shutil.copy2(src, dst)

# Copy original labels to export (replace hashed labels)
export_labels_dir = os.path.join(export_dir, 'labels')
for filename in os.listdir(export_labels_dir):
    os.remove(os.path.join(export_labels_dir, filename))

for filename in os.listdir(original_labels_dir):
    if filename.endswith('.txt'):
        src = os.path.join(original_labels_dir, filename)
        dst = os.path.join(export_labels_dir, filename)
        shutil.copy2(src, dst)

print("Export directory updated with original filenames")
```

### Comparing Original vs Edited Labels

To check which labels were modified in Label Studio:

```python
import os

original_labels_dir = '/path/to/original/labels'
export_labels_dir = '/path/to/export/labels'

# Read labels
original_labels = {}
for filename in os.listdir(original_labels_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(original_labels_dir, filename), 'r') as f:
            original_labels[filename] = f.read().strip()

export_labels = {}
for filename in os.listdir(export_labels_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(export_labels_dir, filename), 'r') as f:
            export_labels[filename] = f.read().strip()

# Find differences
differences = []
for filename, content in export_labels.items():
    if content not in original_labels.values():
        differences.append(filename)

print(f"Modified labels: {len(differences)}")
for diff in differences:
    print(f"  {diff}")
```

## Troubleshooting

### Issue: "Issue loading URL from $image value"
**Cause:** Incorrect image path format  
**Solution:** Use base64 encoding or correct local file path format

### Issue: Images not displaying (404 error)
**Cause:** Label Studio cannot access local file paths  
**Solution:** Use base64 encoding to embed images in JSON

### Issue: Duplicate tasks after import
**Cause:** Uploading images and importing JSON separately  
**Solution:** 
- Delete existing tasks before importing
- Use single JSON import with base64 images
- Or upload images first, then import JSON with filenames only

### Issue: Export labels have hashed filenames
**Cause:** Label Studio's internal naming scheme  
**Solution:** Copy original labels from source directory to export

### Issue: Bounding boxes not displaying correctly
**Cause:** Coordinate format mismatch  
**Solution:** Ensure YOLO to Label Studio coordinate conversion is correct:
- YOLO: center_x, center_y, width, height (normalized 0-1)
- Label Studio: x, y, width, height (percentage 0-100)
- Conversion: `x_pct = (center_x - width/2) * 100`

## Quick Reference

### Label Studio JSON Structure
```json
{
  "data": {
    "image": "data:image/jpeg;base64,<base64_string>"
  },
  "annotations": [
    {
      "result": [
        {
          "from_name": "label",
          "to_name": "image",
          "type": "rectanglelabels",
          "value": {
            "x": 40.5,
            "y": 30.2,
            "width": 20.0,
            "height": 15.0,
            "rotation": 0,
            "rectanglelabels": ["Broken"]
          }
        }
      ]
    }
  ]
}
```

### Coordinate Conversion Summary
| Format | X | Y | Width | Height | Range |
|--------|---|---|-------|--------|-------|
| YOLO | center_x | center_y | width | height | 0-1 |
| Label Studio | x | y | width | height | 0-100 |
| COCO | x | y | width | height | pixels |

### Common Commands
```bash
# Start Label Studio
label-studio start

# Install dependencies
pip install label-studio

# Convert YOLO to Label Studio JSON
python convert_yolo_to_labelstudio.py

# Export from Label Studio
# Use web interface: Export button
```

## Best Practices

1. **Use Base64 Encoding** for local datasets to avoid file path issues
2. **Clean Before Import** to prevent duplicate tasks
3. **Test with Small Sample** before importing full dataset
4. **Backup Original Data** before making modifications
5. **Verify Export** by comparing with original labels
6. **Document Your Workflow** for reproducibility
7. **Use Version Control** for conversion scripts

## Additional Resources

- [Label Studio Documentation](https://labelstud.io/guide/)
- [YOLO Format Guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
