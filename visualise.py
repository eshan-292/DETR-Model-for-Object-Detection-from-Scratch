# visualize.py

import torch
from models.detr import DETR  # Update the path as per your project

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
from PIL import Image,ImageDraw
# import transformers.models.detr.modeling_detr.DetrObjectDetectionOutput as DetrObjectDetectionOutput
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput

from transformers import DetrForObjectDetection, DetrConfig, DetrFeatureExtractor
# ---------------------------
# Helper Functions
# ---------------------------
from data.coco import CocoDetection 

import matplotlib.pyplot as plt
from data.coco import make_coco_transforms


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        # text = f'{model.config.id2label[label]}: {score:0.2f}'
        text = f'Label {label}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def visualise_image(index, dataset, model, device, feature_extractor,finetune=False):
    
    pixel_values, target = dataset[index]
    pixel_values = pixel_values.unsqueeze(0).to(device)
    # print(pixel_values.shape)

    
    
    with torch.no_grad():
        # forward pass to get class logits and bounding boxes
        if finetune:
            outputs = model(pixel_values=pixel_values)
        else:
            outputs = model(pixel_values)    
        
        print("Outputs:", outputs.keys())
    # exit()
    

    image_id = target['image_id'].item()
    image = dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join('data/BoneFractureData/val2017/images', image['file_name']))

    annotations = dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image,"RGBA")
    cats = dataset.coco.cats
    for ann in annotations:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        # cat = cats[cat_id]
        x,y,w,h = tuple(bbox)
        draw.rectangle([x,y,x+w,y+h],outline='red', width=1)
        cat_id = str(cat_id)
        print("Cat_id: ",cat_id)
        draw.text((x,y),cat_id,fill='black')
    # image.show()

    # replace 'pred_logits' and 'pred_boxes' with 'logits' and 'boxes' if you are using a finetuned model
    if finetune==False:
        #create a DETR Output object
        outputs = DetrObjectDetectionOutput(
            logits = outputs['pred_logits'],
            pred_boxes = outputs['pred_boxes']
        )
    # print("Outputs:", outputs)

    # post process the results
    width, height = image.size
    postprocessed_outputs = feature_extractor.post_process_object_detection(outputs,
                                                                target_sizes=[(height, width)],
                                                                threshold=0)
    results = postprocessed_outputs[0]
    plot_results(image, results['scores'], results['labels'], results['boxes'])


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    finetune = True


    # Define the number of classes
    num_classes = 8
    num_queries = 4
    
     # Initialize the feature extractor
    # feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        'facebook/detr-resnet-50',
        size=800,       # Ensure this is an integer
        # max_size=1333   # Ensure this is an integer
        max_size = 800
    )

    val_dataset = CocoDetection(
        img_folder='data/BoneFractureData/val2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        # transforms=make_coco_transforms('val'),
        transforms=None,
        train=False
    )
    train_dataset = CocoDetection(
        img_folder='data/BoneFractureData/train2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        # transforms=make_coco_transforms('train'),
        transforms=None,
        train=True
    )


    if finetune:

        ### Fine-tuned DETR

        # Load the pre-trained DETR model
        model_name = 'facebook/detr-resnet-50'
        config = DetrConfig.from_pretrained(model_name)
        config.num_labels = num_classes + 1  # +1 for the background class
        config.num_queries = num_queries
        # model = DetrForObjectDetection.from_pretrained(model_name, config=config)
        model = DetrForObjectDetection.from_pretrained(
        model_name, 
        config=config, 
        ignore_mismatched_sizes=True
    )
        model.load_state_dict(torch.load('outputs/best_finetuned_model.pth', map_location=torch.device('cpu')))
        model.to(device)

    else:
            
        ### From Scratch DETR
        # Load the saved model
        model = DETR(num_classes=num_classes, num_queries=num_queries, pretrained_weights_path='resnet50-0676ba61.pth')
        model.load_state_dict(torch.load('outputs/best_model.pth', map_location=torch.device('cpu')))
        model.to(device)



    visualise_image(0, val_dataset, model, device, feature_extractor, finetune=finetune)