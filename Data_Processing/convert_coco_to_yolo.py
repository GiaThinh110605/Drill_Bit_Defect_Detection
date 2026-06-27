from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/test_preprocessed/Bright_Field/",
    save_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/test_preprocessed/Bright_Field",
    cls91to80=False
)
convert_coco(
    labels_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/test_preprocessed/Dark_Field/",
    save_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/test_preprocessed/Dark_Field",
    cls91to80=False
)

convert_coco(
    labels_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/train_preprocessed/",
    save_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/train_preprocessed/",
    cls91to80=False
)

convert_coco(
    labels_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/valid_preprocessed/",
    save_dir="/Users/mac/Detect_Drill_Bit/mui_khoan/valid_preprocessed/",
    cls91to80=False
)
