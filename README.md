# DETR: Object Detection Transformer Implementation

## **Overview**
This project implements the **DETR (DEtection TRansformer)** model from scratch and fine-tunes a pretrained version for detecting and classifying **bone fractures in X-ray images**. DETR combines a convolutional neural network (CNN) backbone with a transformer-based encoder-decoder architecture for object detection.

---


### **Structure**
The dataset follows the COCO format, which includes:
- **Images:** X-ray images for training, validation, and testing.
- **Annotations:** Bounding boxes and class labels for each image.

---

## **Key Components**

### **Model Architecture**
1. **Backbone:**
   - ResNet50 extracts features from input images.
2. **Transformer:**
   - Consists of multi-head self-attention, cross-attention, and feedforward layers.
   - Uses **object queries** for detecting multiple objects per image.
3. **Prediction Heads:**
   - Classification head for predicting object classes.
   - Bounding box regression head for localizing objects.
4. **Positional Encoding:**
   - Ensures the spatial structure of the image is preserved.

---

### **Training and Fine-Tuning**
- **From Scratch:**
  - The DETR model is trained from scratch using the custom dataset.
  - Data augmentations like resizing, flipping, and normalization are applied.
- **Fine-Tuning:**
  - A pretrained DETR model (from Hugging Face) is fine-tuned on the dataset for faster convergence and better performance.

### **Loss Function**
- **Hungarian Matching:** Matches ground truth objects with predictions using the optimal assignment.
- **Set Loss:**
  - Classification loss (Cross-Entropy).
  - Bounding box regression loss (L1 Loss and GIoU).

---

## **Project Structure**

```plaintext
A2/
├── data/
│   ├── coco.py             # Dataset loading and preprocessing
│   ├── transforms.py       # Data augmentations
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   ├── test/               # Test images
│   ├── annotations/        # COCO-format annotations
├── models/
│   ├── detr.py             # DETR architecture implementation
│   ├── transformer.py      # Transformer components (encoder-decoder)
│   ├── loss.py             # Loss functions (Hungarian Matching, Set Loss)
│   ├── backbone.py         # CNN backbone (ResNet50)
├── utils/
│   ├── box_ops.py          # Operations for bounding boxes (IoU, GIoU, etc.)
│   ├── utils.py            # Helper functions (data loading, logging)
├── train.py                # Training the DETR model
├── eval.py                 # Evaluation script
├── finetune.py             # Fine-tuning pretrained DETR
├── visualise.py            # Visualize predictions and ground truth
├── outputs/                # Model checkpoints
│   ├── best_model.pth      # Best scratch-trained model
│   ├── best_finetuned_model.pth  # Best fine-tuned model
├── requirements.txt        # Python dependencies
├── ReadMe                  # Project documentation
├── resnet50-0676ba61.pth   # Pretrained resnet50 backbone



Running Instructions:

Download the pretrained models:

Resnet_Backbone: https://drive.google.com/file/d/1Hnxp4-hSJjsbo_pI0f8Ran7x6Mwhf5DL/view?usp=sharing 
Pretrained_scratch:https://drive.google.com/file/d/1fp_OK4WQ3wpACCkr__YJg7INQ8WA86AH/view?usp=sharing 
Pretrained_finetuned:https://drive.google.com/file/d/13Kurxxl17bYfrucYYndbvDn74ialiT-W/view?usp=sharing 



1. Train DETR from Scratch:
python3.10 train.py 

2. Fine-Tune Pre-Trained DETR:
python3.10 finetune.py 

3. Evaluate the Model:
python3.10 eval.py 

3. Visualise Model Outputs:
python3.10 visualise.py 
