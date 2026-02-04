# Car_Damage_Detection
YOLOv8 Model
# 🚗 YOLO11 Car Damage Detector

A deep learning-based vehicle damage detection and cost estimation system using YOLO11m and YOLOv8 models. This project automatically detects vehicles in images, identifies various types of damage, and provides repair cost estimates.

## ✨ Features

- **Multi-Vehicle Detection**: Automatically identifies and focuses on the main vehicle in images
- **Damage Classification**: Detects 9 types of vehicle damage including dents, scratches, and component damage
- **Cost Estimation**: Provides automated repair cost calculations based on detected damage
- **Detailed Reporting**: Generates comprehensive damage reports with visual annotations
- **High Accuracy**: Uses YOLO11m model trained on specialized car damage dataset

## 🎯 Supported Damage Types

| Damage Type | Base Repair Cost (USD) |
|------------|------------------------|
| Front Windscreen Damage | $500 |
| Headlight Damage | $350 |
| Bonnet Dent | $250 |
| Front Bumper Dent | $200 |
| Rear Bumper Dent | $180 |
| Door Outer Dent | $150 |
| Fender Dent | $120 |
| Door Outer Scratch | $50 |
| Other Damage | $100 |

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- Internet connection (for initial model download)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/YOLO11-Car-Damage-Detector.git
cd YOLO11-Car-Damage-Detector

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('trained.pt')

# Run detection on images
results = model.predict(source='path/to/images', save=True, conf=0.07)
```

For detailed usage examples, see the [Usage Guide](docs/usage.md).

## 📁 Project Structure

```
YOLO11-Car-Damage-Detector/
├── trained.pt              # Pre-trained YOLO11m model
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
├── docs/
│   ├── api.md            # API documentation
│   └── usage.md          # Detailed usage guide
└── examples/
    ├── basic_detection.py
    ├── cost_estimation.py
    └── vehicle_focus.py
```

## 🔧 Main Workflows

### 1. Basic Damage Detection
Detect damage on vehicle images and save annotated results.

### 2. Cost Estimation Analysis
Generate detailed damage reports with repair cost calculations.

### 3. Multi-Vehicle Scene Processing
Automatically identify the main vehicle and analyze damage while ignoring background vehicles.

See [examples/](examples/) for complete code samples.

## 📊 Model Performance

- **Model**: YOLO11m
- **Training Dataset**: Roboflow Car Damage Detection
- **Input Size**: 320x320
- **Confidence Threshold**: 0.07-0.10 (adjustable)
- **Optimizer**: SGD with AMP

## 🎓 Training Your Own Model

```python
from ultralytics import YOLO

model = YOLO('yolo11m.pt')

model.train(
    data="path/to/data.yaml",
    epochs=35,
    imgsz=320,
    batch=4,
    optimizer='SGD',
    amp=True,
    lr0=0.01
)
```

For complete training instructions, see [docs/usage.md](docs/usage.md).

## 📸 Example Results

The system processes images and provides:
- Visual bounding boxes around detected damage
- Damage type labels
- Individual and total repair cost estimates
- Summary reports in DataFrame format

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/) for the car damage detection dataset
- YOLOv8 and YOLO11 model architectures

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## ⚠️ Disclaimer

Cost estimates are approximate and for reference only. Actual repair costs may vary based on location, service provider, and vehicle specifications. Always consult with professional mechanics for accurate quotes.
