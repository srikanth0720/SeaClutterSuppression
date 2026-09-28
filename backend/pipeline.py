import sys
import os
import time
import numpy as np
import torch

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Existing project modules
# ---------------------------------------------------------

import models
import sea_clutter

from src.generate_data import (
    simulate_sequence_with_realistic_targets_and_masks
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "pretrained",
    "tversky.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"

)
AI_ENGINE = None


# ---------------------------------------------------------
# AI MODEL
# ---------------------------------------------------------

class RadarAI:

    def __init__(self):

        print(f"Loading AI model on: {DEVICE}")

        self.model = models.UNet(
            n_channels=3,
            n_classes=1,
            base_filters=64
        ).to(DEVICE)

        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )
        )

        self.model.eval()

        print("AI model loaded successfully.")


    def predict(self, frames):

        """
        frames shape:

        (3, 128, 128)

        Returns:

        probability_map
        inference_time_ms
        """

        frames = np.asarray(
            frames,
            dtype=np.float32
        )

        if frames.shape != (3, 128, 128):

            raise ValueError(
                f"Expected (3, 128, 128), "
                f"received {frames.shape}"
            )

        # NumPy → PyTorch

        input_tensor = torch.tensor(
            frames,
            dtype=torch.float32
        )

        # Add batch dimension
        #
        # (3,128,128)
        #       ↓
        # (1,3,128,128)

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        # Measure AI inference time

        start = time.perf_counter()

        with torch.no_grad():

            output = self.model(
                input_tensor
            )

            probability_map = torch.sigmoid(
                output
            )

        end = time.perf_counter()

        inference_time_ms = (
            end - start
        ) * 1000

        # Remove:
        #
        # batch dimension
        # channel dimension
        #
        # (1,1,128,128)
        #       ↓
        # (128,128)

        probability_map = (
            probability_map
            .squeeze()
            .cpu()
            .numpy()
        )

        return (
            probability_map,
            inference_time_ms
        )


# ---------------------------------------------------------
# AI DETECTION
# ---------------------------------------------------------

def create_detection_map(
    probability_map,
    threshold=0.5
):

    detection_map = (
        probability_map >= threshold
    )

    return detection_map.astype(
        np.uint8
    )


def count_detections(
    detection_map
):

    return int(
        np.sum(detection_map)
    )


# ---------------------------------------------------------
# RADAR SIMULATOR
# ---------------------------------------------------------

def generate_radar_sequence(
    n_targets=3
):

    """
    Generate one realistic 3-frame
    maritime radar sequence.

    This uses the EXISTING
    SeaClutterSuppression simulator.
    """

    # Radar configuration

    radar_params = (
        sea_clutter.RadarParams()
    )

    # Clutter configuration

    clutter_params = (
        sea_clutter.ClutterParams()
    )

    # Sequence configuration

    sequence_params = (
        sea_clutter.SequenceParams()
    )

    sequence_params.n_frames = 3

    sequence_params.frame_rate_hz = 2

    # Target type

    target_type = (
        sea_clutter.TargetType.SPEEDBOAT
    )

    # Create realistic targets

    targets = []

    for _ in range(n_targets):

        target = (
            sea_clutter.create_realistic_target(
                target_type,
                np.random.randint(
                    30,
                    radar_params.n_ranges - 30
                ),
                radar_params
            )
        )

        targets.append(target)

    # Run existing simulator

    rdm_list, mask_list = (
        simulate_sequence_with_realistic_targets_and_masks(
            radar_params,
            clutter_params,
            sequence_params,
            targets
        )
    )

    # Convert RDMs to dB

    processed_frames = []

    for rdm in rdm_list:

        rdm_db = (
            20 *
            np.log10(
                np.abs(rdm) + 1e-10
            )
        )

        processed_frames.append(
            rdm_db
        )

    sequence = np.stack(
        processed_frames,
        axis=0
    )

    masks = np.stack(
        mask_list,
        axis=0
    )

    return (
        sequence.astype(np.float32),
        masks,
        targets
    )


# ---------------------------------------------------------
# COMPLETE PIPELINE
# ---------------------------------------------------------

def run_pipeline(
    n_targets=3,
    threshold=0.5
):

    """
    Complete radar → AI pipeline.

    Returns everything required
    by the future dashboard.
    """

    # -----------------------------------------------------
    # 1. Generate radar data
    # -----------------------------------------------------

    simulation_start = (
        time.perf_counter()
    )

    frames, ground_truth, targets = (
        generate_radar_sequence(
            n_targets=n_targets
        )
    )

    simulation_time_ms = (
        time.perf_counter()
        - simulation_start
    ) * 1000

    # -----------------------------------------------------
    # 2. AI inference
    # -----------------------------------------------------
    global AI_ENGINE

    if AI_ENGINE is None:
        AI_ENGINE = RadarAI()

    probability_map, inference_time_ms = AI_ENGINE.predict(frames)


    # -----------------------------------------------------
    # 3. Detection map
    # -----------------------------------------------------

    detection_map = (
        create_detection_map(
            probability_map,
            threshold
        )
    )

    # -----------------------------------------------------
    # 4. Detection statistics
    # -----------------------------------------------------

    detected_pixels = (
        count_detections(
            detection_map
        )
    )

    # -----------------------------------------------------
    # 5. Return dashboard data
    # -----------------------------------------------------

    return {

        "frames": frames,

        "ground_truth": ground_truth,

        "probability_map":
            probability_map,

        "detection_map":
            detection_map,

        "targets":
            targets,

        "detected_pixels":
            detected_pixels,

        "threshold":
            threshold,

        "simulation_time_ms":
            simulation_time_ms,

        "ai_inference_time_ms":
            inference_time_ms,

        "total_processing_time_ms":
            simulation_time_ms
            + inference_time_ms
    }


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)

    print(
        "MARITIME RADAR AI "
        "REAL PIPELINE TEST"
    )

    print("=" * 60)

    result = run_pipeline(
        n_targets=3,
        threshold=0.5
    )

    print()

    print(
        "Radar sequence shape:",
        result["frames"].shape
    )

    print(
        "Ground truth shape:",
        result["ground_truth"].shape
    )

    print(
        "AI probability map:",
        result["probability_map"].shape
    )

    print(
        "AI detection map:",
        result["detection_map"].shape
    )

    print(
        "Number of simulated targets:",
        len(result["targets"])
    )

    print(
        "Detected pixels:",
        result["detected_pixels"]
    )

    print(
        "AI inference:",
        f"{result['ai_inference_time_ms']:.2f} ms"
    )

    print(
        "Simulation time:",
        f"{result['simulation_time_ms']:.2f} ms"
    )

    print(
        "Total processing:",
        f"{result['total_processing_time_ms']:.2f} ms"
    )

    print()

    print(
        "REAL RADAR → AI PIPELINE "
        "TEST COMPLETE"
    )