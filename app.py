import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #a7adb8;
    margin-bottom: 2rem;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.stat-card {
    background-color: #1b1f2a;
    border: 1px solid #303642;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
}

.stat-label {
    color: #a7adb8;
    font-size: 0.9rem;
}

.footer {
    text-align: center;
    color: #777f8d;
    margin-top: 3rem;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------

class_names = {
    0: "Missing Hole",
    1: "Mouse Bite",
    2: "Open Circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious Copper",
    6: "Burnt IC",
    7: "Burnt Track"
}

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">PCB Defect Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-based detection of manufacturing defects in Printed Circuit Boards '
    'using a trained YOLO11n model.'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">8</div>
        <div class="stat-label">Defect Classes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">YOLO11n</div>
        <div class="stat-label">Detection Model</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">640</div>
        <div class="stat-label">Inference Image Size</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Upload PCB Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">Input Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.markdown(
            '<div class="section-title">Analysis</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The uploaded image will be analysed using the trained "
            "8-class PCB defect detection model."
        )

        detect_button = st.button(
            "🔍 Detect Defects",
            use_container_width=True
        )

    # --------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------

    if detect_button:

        with st.spinner("Analysing PCB image..."):

            results = model.predict(
                source=image,
                conf=0.40,
                imgsz=640
            )

        result = results[0]
        boxes = result.boxes

        # --------------------------------------------------
        # RESULT IMAGE
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">Detection Result</div>',
            unsafe_allow_html=True
        )

        annotated_image = result.plot()

        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

        # --------------------------------------------------
        # DETECTION RESULTS
        # --------------------------------------------------

        st.markdown(
            '<div class="section-title">Detection Summary</div>',
            unsafe_allow_html=True
        )

        if boxes is not None and len(boxes) > 0:

            detected_names = []
            confidences = []

            for box in boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                detected_names.append(
                    class_names[class_id]
                )

                confidences.append(
                    confidence
                )

            total_detections = len(detected_names)
            unique_defects = len(set(detected_names))
            highest_confidence = max(confidences)

            # Summary cards
            stat1, stat2, stat3 = st.columns(3)

            with stat1:
                st.metric(
                    "Total Detections",
                    total_detections
                )

            with stat2:
                st.metric(
                    "Defect Types",
                    unique_defects
                )

            with stat3:
                st.metric(
                    "Highest Confidence",
                    f"{highest_confidence * 100:.1f}%"
                )

            # --------------------------------------------------
            # GROUP DEFECT TYPES
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">Detected Defects</div>',
                unsafe_allow_html=True
            )

            defect_counts = Counter(detected_names)

            for defect, count in defect_counts.items():

                defect_confidences = [
                    confidences[i]
                    for i in range(len(detected_names))
                    if detected_names[i] == defect
                ]

                best_confidence = max(defect_confidences)

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### {defect}")

                with col2:
                    st.metric(
                        "Confidence",
                        f"{best_confidence * 100:.1f}%"
                    )

                if count == 1:
                    st.caption("1 detection")
                else:
                    st.caption(f"{count} detections")

                st.divider()

        else:

            st.success(
                "No defects were detected in this image."
            )

# --------------------------------------------------
# SUPPORTED CLASSES
# --------------------------------------------------

with st.expander("View supported defect classes"):

    for class_id, class_name in class_names.items():

        st.write(
            f"**{class_id}.** {class_name}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        PCB Defect Detection • YOLO11n • 8-Class Model
    </div>
    """,
    unsafe_allow_html=True
)