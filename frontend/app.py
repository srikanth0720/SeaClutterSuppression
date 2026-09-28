import sys
import os

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from backend.pipeline import run_pipeline


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Maritime Radar AI",
    page_icon="⚓",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0b1220;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: white;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #263244;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.status {
    background-color: #123524;
    color: #4ade80;
    padding: 8px 15px;
    border-radius: 20px;
    text-align: center;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
# Pass vertical_alignment="center" to the column layout
header_left, header_right = st.columns([4, 1], vertical_alignment="center")

with header_left:
    st.title("📡Navika-AI")
    st.write("AI-Based Clutter Rejection & Real-Time Suppression")

with header_right:
    st.markdown(
        '<div class="status">● SYSTEM ONLINE</div>',
        unsafe_allow_html=True
    )

st.divider()


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.title("Radar Controls")

n_targets = st.sidebar.slider(
    "Number of Targets",
    min_value=1,
    max_value=6,
    value=3
)

threshold = st.sidebar.slider(
    "AI Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

run_button = st.sidebar.button(
    "▶ RUN RADAR SIMULATION",
    use_container_width=True
)


# ---------------------------------------------------------
# INITIAL STATE
# ---------------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None


# ---------------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------------

if run_button:

    with st.spinner("Running radar simulation and AI detection..."):

        result = run_pipeline(
            n_targets=n_targets,
            threshold=threshold
        )

        st.session_state.result = result


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

result = st.session_state.result

if result is None:

    st.info(
        "Configure the radar parameters from the sidebar "
        "and click **RUN RADAR SIMULATION**."
    )

else:

    frames = result["frames"]
    probability_map = result["probability_map"]
    detection_map = result["detection_map"]

    ai_time = result["ai_inference_time_ms"]
    simulation_time = result["simulation_time_ms"]
    total_time = result["total_processing_time_ms"]

    targets = len(result["targets"])
    detected_pixels = result["detected_pixels"]


    # -----------------------------------------------------
    # RADAR VISUALIZATION
    # -----------------------------------------------------

    st.subheader("Radar Processing Pipeline")

    col1, col2 = st.columns(2)


    # RAW RADAR
    with col1:

        st.markdown("### 📡 Raw Radar RDM")

        fig1, ax1 = plt.subplots(figsize=(7, 5))

        ax1.imshow(
            frames[-1],
            aspect="auto",
            origin="lower"
        )

        ax1.set_xlabel("Doppler / Velocity")
        ax1.set_ylabel("Range")
        ax1.set_title("Radar Range-Doppler Map")

        st.pyplot(
            fig1,
            use_container_width=True
        )

        plt.close(fig1)


    # AI OUTPUT
    with col2:

        st.markdown("### 🤖 AI Detection Map")

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        ax2.imshow(
            probability_map,
            aspect="auto",
            origin="lower"
        )

        ax2.set_xlabel("Doppler / Velocity")
        ax2.set_ylabel("Range")
        ax2.set_title("AI Probability Map")

        st.pyplot(
            fig2,
            use_container_width=True
        )

        plt.close(fig2)


    st.divider()


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    st.subheader("System Performance")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">TARGETS SIMULATED</div>
                <div class="metric-value">{targets}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">DETECTED PIXELS</div>
                <div class="metric-value">{detected_pixels}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">AI INFERENCE</div>
                <div class="metric-value">{ai_time:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">TOTAL PROCESSING</div>
                <div class="metric-value">{total_time:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.divider()


    # -----------------------------------------------------
    # PROCESSING INFORMATION
    # -----------------------------------------------------

    st.subheader("Processing Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.write("**Radar Input**")
        st.write(f"3-frame sequence")
        st.write(f"{frames.shape[1]} × {frames.shape[2]} RDM")

    with info2:
        st.write("**AI Model**")
        st.write("Temporal U-Net")
        st.write("Pretrained Tversky model")

    with info3:
        st.write("**Detection Threshold**")
        st.write(f"{threshold:.2f}")
        st.write(f"Simulation: {simulation_time:.1f} ms")


    # -----------------------------------------------------
    # DETECTION MAP
    # -----------------------------------------------------

    st.divider()

    st.subheader("Final AI Detection")

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.imshow(
        detection_map,
        aspect="auto",
        origin="lower"
    )

    ax3.set_xlabel("Doppler / Velocity")
    ax3.set_ylabel("Range")
    ax3.set_title("Thresholded AI Detection Map")

    st.pyplot(
        fig3,
        use_container_width=True
    )

    plt.close(fig3)

