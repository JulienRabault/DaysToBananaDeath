# BananaCheck - MLOps Learning Project

A modern ML project showcasing **MLOps best practices** with banana ripeness classification.

**Live Demo:** https://daystobananadeath.streamlit.app/

## Project Goals

This project was built as a learning exercise to master key MLOps technologies:

- **Configuration Management** with Hydra
- **Experiment Tracking & Model Versioning** with Weights & Biases
- **API Development** with FastAPI
- **Model Deployment** with ONNX Runtime
- **Web Interface** with Streamlit

## Features

- **AI-powered banana ripeness estimation** (days remaining before spoilage)
- **REST API** for programmatic access
- **Web interface** for easy image upload and prediction
- **Model versioning** and artifact management via W&B
- **Production-ready deployment** with Docker support

## Tech Stack

### Core ML/MLOps
- **PyTorch Lightning** - Training framework
- **Hydra** - Configuration management
- **Weights & Biases** - Experiment tracking, model versioning
- **ONNX Runtime** - Model inference optimization

### API & Web
- **FastAPI** - REST API with automatic documentation
- **Streamlit** - Interactive web interface
- **Uvicorn** - ASGI server

### Data & Vision
- **Albumentations** - Image augmentation pipeline
- **PIL/OpenCV** - Image processing

## Model Performance

- **Architecture**: ResNet50 & Vision Transformer (ViT-B/16)
- **Dataset**: 20K+ banana images across 5 ripeness stages
- **Classes**: unripe, ripe, overripe, rotten, unknowns
- **Deployment**: ONNX optimized for CPU inference

## Quick Start

### API Usage
```bash
# Start the API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@banana.jpg"
```

### Web Interface
```bash
# Start Streamlit app
streamlit run streamlit_app.py

# Navigate to http://localhost:8501
```

### Environment Setup
```bash
pip install -r requirements.txt

# For W&B model loading
export WANDB_API_KEY=your_api_key
```

## Project Structure

```
├── src/
│   ├── api.py              # FastAPI REST endpoint
│   ├── model.py            # Model architectures
│   ├── train.py            # Training pipeline
│   └── inference.py        # Inference utilities
├── configs/                # Hydra configuration files
├── streamlit_app.py        # Web interface
├── requirements.txt        # Python dependencies
└── Procfile               # Deployment configuration
```

## Key Learning Outcomes

### Hydra Configuration Management
- Structured config files for different components (model, data, training)
- Easy experiment switching with config overrides
- Environment-specific configurations

### W&B MLOps Pipeline
- Automated experiment tracking with metrics logging
- Model artifact versioning and storage
- Seamless model deployment from artifacts
- Training curve visualization and comparison

### Production API Design
- RESTful API with proper error handling
- Automatic model loading from W&B artifacts
- Health checks and monitoring endpoints
- ONNX optimization for fast inference

## Dataset Sources

Built using established vision datasets:
- [BananaRipeness Dataset](https://github.com/luischuquim/BananaRipeness/) by Chuquimarca et al.
- Additional synthetic and augmented data

## Deployment

The project includes production-ready deployment configurations:
- **Heroku/Railway**: via Procfile
- **Docker**: containerized deployment
- **Environment variables**: for secure API key management

---

*Built to explore modern MLOps practices and deployment strategies*
