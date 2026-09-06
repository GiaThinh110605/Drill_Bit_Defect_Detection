import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io


# =========================
# CONFIG
# =========================

API_URL = "http://backend:8000/predict"

st.set_page_config(
    page_title="Drill Bit Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# =========================
# STYLE
# =========================

st.title("🔍 Drill Bit Defect Detection")
st.write("Upload an image of a drill bit to detect defects.")


# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# DRAW BOUNDING BOX
# =========================
def draw_detections(image, detections):

    image = image.copy()
    draw = ImageDraw.Draw(image)

    image_w, image_h = image.size

    # Font
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            18
        )
    except:
        font = ImageFont.load_default()

    for detection in detections:

        bbox = detection["bbox"]

        # -------------------------
        # GET COORDINATES
        # -------------------------

        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

        # -------------------------
        # FIX REVERSED COORDINATES
        # -------------------------

        x_min = min(x1, x2)
        x_max = max(x1, x2)

        y_min = min(y1, y2)
        y_max = max(y1, y2)

        # -------------------------
        # CLAMP TO IMAGE
        # -------------------------

        x_min = max(0, min(image_w - 1, x_min))
        x_max = max(0, min(image_w - 1, x_max))

        y_min = max(0, min(image_h - 1, y_min))
        y_max = max(0, min(image_h - 1, y_max))

        # Nếu bbox không hợp lệ thì bỏ qua
        if x_max <= x_min or y_max <= y_min:
            continue

        # -------------------------
        # CLASS + CONFIDENCE
        # -------------------------

        class_name = detection["class_name"]
        confidence = float(detection["confidence"])

        # API confidence = 0 -> 1
        label = f"{class_name} {confidence * 100:.1f}%"

        # -------------------------
        # DRAW BBOX
        # -------------------------

        draw.rectangle(
            [
                int(x_min),
                int(y_min),
                int(x_max),
                int(y_max)
            ],
            outline="green",
            width=4
        )

        # -------------------------
        # LABEL SIZE
        # -------------------------

        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # -------------------------
        # LABEL POSITION
        # -------------------------

        label_x = int(x_min)

        # Ưu tiên đặt label phía trên bbox
        label_y = int(y_min - text_height - 8)

        # Nếu quá sát mép trên thì đặt bên trong bbox
        if label_y < 0:
            label_y = int(y_min + 5)

        # Không cho label vượt bên phải ảnh
        label_x = min(
            label_x,
            image_w - text_width - 10
        )

        label_x = max(0, label_x)

        # -------------------------
        # DRAW LABEL BACKGROUND
        # -------------------------

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + text_width + 10,
                label_y + text_height + 8
            ],
            fill="green"
        )

        # -------------------------
        # DRAW TEXT
        # -------------------------

        draw.text(
            (
                label_x + 5,
                label_y + 4
            ),
            label,
            fill="white",
            font=font
        )

    return image

# =========================
# MAIN
# =========================

if uploaded_file:

    # Load original image
    image_bytes = uploaded_file.getvalue()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # =========================
    # PREVIEW
    # =========================

    st.subheader("Original Image")

    st.image(
        image,
        width="stretch"
    )


    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button(
        "🔍 Detect Defects",
        type="primary"
    ):

        with st.spinner("Detecting defects..."):

            try:

                response = requests.post(
                    API_URL,
                    files={
                        "image": (
                            uploaded_file.name,
                            image_bytes,
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )


                # =========================
                # CHECK API
                # =========================

                if response.status_code != 200:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.code(
                        response.text
                    )

                    st.stop()


                result = response.json()


                if not result.get("success", False):

                    st.error(
                        result.get(
                            "message",
                            "Prediction failed"
                        )
                    )

                    st.stop()


                detections = result.get(
                    "detections",
                    []
                )


                # =========================
                # DRAW RESULT
                # =========================

                result_image = draw_detections(
                    image,
                    detections
                )


                st.subheader(
                    "Detection Result"
                )

                st.image(
                    result_image,
                    width="stretch"
                )


                # =========================
                # SUMMARY
                # =========================

                if len(detections) == 0:

                    st.success(
                        "No defects detected."
                    )

                else:

                    st.success(
                        f"Detected {len(detections)} defect(s)."
                    )


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI."
                )

                st.info(
                    "Make sure FastAPI is running at "
                    "http://127.0.0.1:8000"
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ API request timed out."
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {str(e)}"
                )