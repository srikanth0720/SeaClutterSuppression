# Maritime Radar AI — AI-Based Sea-Clutter Rejection

An AI-powered software demonstration for maritime radar target detection and sea-clutter rejection using a pretrained Temporal U-Net model, synthetic radar data, and an interactive Streamlit dashboard.

The system demonstrates how deep-learning-based processing can assist target detection in Range-Doppler Maps affected by simulated maritime sea clutter.

> **Project Scope:** This is a software demonstration using simulated radar and sea-clutter data. It is not connected to live radar hardware and should not be interpreted as an operational or military radar system.

---

## Overview

Maritime radar systems can experience strong sea clutter that can make small or low-power targets difficult to detect.

This project demonstrates an AI-based processing pipeline that generates simulated maritime radar data, processes it into Range-Doppler Maps, applies a pretrained Temporal U-Net model, and visualizes the resulting target-detection information through a Streamlit dashboard.

### Processing Pipeline

```text
Radar Simulation
       ↓
Sea-Clutter + Target Generation
       ↓
Range-Doppler Map
       ↓
3-Frame Temporal Input
       ↓
Pretrained Temporal U-Net
       ↓
AI Probability Map
       ↓
Clutter Rejection / Target Detection
       ↓
Streamlit Dashboard

Key Features

Synthetic maritime radar simulation
Sea-clutter generation
Moving target simulation
Range-Doppler Map generation
3-frame temporal radar input
Pretrained Temporal U-Net inference
AI probability-map generation
Threshold-based target detection
Configurable detection threshold
Processing-time measurement
Interactive Streamlit dashboard
Visual radar-processing pipeline
Existing CA-CFAR implementation for future comparison

                         MARITIME RADAR AI
                                │
                                ▼
                    ┌─────────────────────┐
                    │   Radar Simulator   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sea Clutter +       │
                    │ Maritime Targets    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Range-Doppler Maps  │
                    │      128 × 128      │
                    └──────────┬──────────┘
                               │
                         3-frame input
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Temporal U-Net    │
                    │   Tversky Model     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  AI Probability Map │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Target Detection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └─────────────────────┘

                    SeaClutterSuppression/
│
├── frontend/
│   └── app.py
│
├── backend/
│   └── pipeline.py
│
├── models/
│   ├── Unet.py
│   ├── end_to_end.py
│   └── swin_transformer.py
│
├── pretrained/
│   ├── tversky.pt
│   ├── behemoth.pt
│   └── behemoth2.pt
│
├── sea_clutter/
│   ├── parameters.py
│   ├── physics.py
│   ├── sea_helper.py
│   └── load_data.py
│
├── src/
│   ├── generate_data.py
│   ├── evaluate.py
│   ├── helper.py
│   └── unet_training.py
│
├── evaluation_results/
│
├── CA-CFAR.py
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore

                    GitHub Repository
                           │
                           ▼
                    Deployment Platform
                           │
                           ▼
                  Streamlit Application
                           │
                           ▼
                    frontend/app.py
                           │
                           ▼
                   backend/pipeline.py
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        Radar Simulator   U-Net       CA-CFAR
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    Detection Results
                           │
                           ▼
                       Dashboard