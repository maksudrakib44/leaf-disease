# Tomato Plant Leaf Disease Detection

A deep learning-based solution for automated detection and classification of tomato plant diseases using Convolutional Neural Networks (CNN) with Transfer Learning.

## 📋 Overview

This project implements a state-of-the-art deep learning model for identifying diseases in tomato plants from leaf images. The model achieves high accuracy in classifying 10 different tomato plant conditions, including 9 disease types and healthy leaves.

View on colab: https://colab.research.google.com/drive/1thOOZatkITWDyulKV5Ix80EGAXHBqw4W?usp=sharing

### 🎯 Key Features
- Multi-class classification of tomato leaf diseases
- Transfer learning with pre-trained CNN architecture
- Comprehensive data preprocessing and augmentation
- High accuracy on validation dataset
- Real-time prediction capabilities

### 📊 Dataset
The model is trained on the PlantVillage dataset, specifically focusing on tomato plants:
- **Total Images**: 16,011
- **Number of Classes**: 10
- **Image Resolution**: 224×224 pixels

### 🔬 Detected Classes
| Class ID | Disease Name |
|----------|--------------|
| 0 | Bacterial spot |
| 1 | Early blight |
| 2 | Late blight |
| 3 | Leaf Mold |
| 4 | Septoria leaf spot |
| 5 | Spider mites / Two-Spotted spider mite |
| 6 | Target Spot |
| 7 | Yellow Leaf Curl Virus |
| 8 | Mosaic virus |
| 9 | Healthy |

## 🛠️ Technical Architecture

### Model Specifications
- **Base Architecture**: Pre-trained CNN (Transfer Learning)
- **Input Size**: 224×224 pixels
- **Output**: 10 classes (softmax activation)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy

### Training Configuration
- **Batch Size**: 32
- **Initial Learning Rate**: 1e-3
- **Minimum Learning Rate**: 1e-7
- **Patience**: 10 epochs
- **Dropout Rate**: 0.5
- **Fine-tuning Layer**: 100

### Data Augmentation
To improve model generalization, the following augmentation techniques are applied:
- Rotation range: 40°
- Width shift: 20%
- Height shift: 20%
- Zoom range: 20%
- Horizontal flip
- Vertical flip
- Brightness range: 0.8-1.2

## 📈 Performance

The model demonstrates robust performance across all disease classes with high accuracy metrics. Detailed performance metrics including confusion matrix and classification report are available in the notebook.

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow==2.18.0 keras==3.8.0 opencv-python==4.12.0
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/maksudrakib44/leaf-disease.git
cd leaf-disease
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the PlantVillage dataset and place it in the appropriate directory structure:
```
/PlantVillage/
├── Tomato_Bacterial_spot/
├── Tomato_Early_blight/
├── Tomato_Late_blight/
├── Tomato_Leaf_Mold/
├── Tomato_Septoria_leaf_spot/
├── Tomato_Spider_mites_Two_spotted_spider_mite/
├── Tomato__Target_Spot/
├── Tomato__Tomato_YellowLeaf__Curl_Virus/
├── Tomato__Tomato_mosaic_virus/
└── Tomato_healthy/
```

### Usage

#### Training
Run the Jupyter notebook `draft_v2_leaf_disease.ipynb` for:
- Data preprocessing and augmentation
- Model training with transfer learning
- Performance evaluation
- Model saving

#### Prediction
```python
# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Predict on new image
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    
    return class_names[class_idx], predictions[0][class_idx]

# Example usage
disease_name, confidence = predict_disease('path/to/leaf/image.jpg')
print(f"Disease: {disease_name}, Confidence: {confidence:.2f}")
```

## 📁 Project Structure
```
leaf-disease/
│
├── draft_v2_leaf_disease.ipynb   # Main training notebook
├── best_model.h5                  # Trained model weights
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── logs/                          # Training logs
│   └── training_log.csv
│
└── sample_images/                 # Sample prediction images
    ├── bacterial_spot_sample.jpg
    ├── healthy_sample.jpg
    └── ...
```

## 🎯 Results

The model achieves high accuracy in distinguishing between different tomato diseases. Key metrics include:
- **Accuracy**: Detailed in notebook
- **Precision**: Per-class metrics available
- **Recall**: Comprehensive evaluation
- **F1-Score**: Balanced performance measure

## 🔮 Future Improvements

1. **Model Enhancements**
   - Implement ensemble methods
   - Experiment with different architectures (ResNet, EfficientNet)
   - Add attention mechanisms

2. **Dataset Expansion**
   - Include more plant species
   - Add images from different growth stages
   - Incorporate field-captured images



## 📚 References

- PlantVillage Dataset: [[https://plantvillage.psu.edu/](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)]([https://plantvillage.psu.edu/](https://www.kaggle.com/datasets/arjuntejaswi/plant-village))
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras Applications: [https://keras.io/api/applications/](https://keras.io/api/applications/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Maksud Rakib - [@maksudrakib44](https://github.com/maksudrakib44)

Project Link: [https://github.com/maksudrakib44/leaf-disease](https://github.com/maksudrakib44/leaf-disease)

---
**Note**: This project is for research and educational purposes. Always consult with agricultural experts for actual disease diagnosis and treatment.
