# eval.py
import torch
from torch.utils.data import DataLoader
from models.detr import DETR
from utils.utils import collate_fn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.box_ops import box_cxcywh_to_xyxy
import json
import os
from transformers import DetrForObjectDetection, DetrConfig, DetrFeatureExtractor
import pycocotools
from coco_eval import CocoEvaluator
from tqdm import tqdm
import numpy as np
from data.coco import CocoDetection
from utils.utils import collate_fn
import torch.nn.functional as F
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from data.coco import make_coco_transforms


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue
        if len(prediction["scores"]) == 0:
            continue  # Skip empty predictions


        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    finetune = False
    num_classes = 8
    num_queries = 4
    batch_size = 16
    iou_threshold= 0.2

    feature_extractor = DetrFeatureExtractor.from_pretrained(
            'facebook/detr-resnet-50',
            size=800,       # Ensure this is an integer
            # max_size=1333   # Ensure this is an integer
            max_size = 800
        )
    
    val_dataset = CocoDetection(
         img_folder='data/BoneFractureData/test2017/images',
         ann_folder='data/BoneFractureData/annotations',
         processor=feature_extractor,
        #  transforms=make_coco_transforms('val'),
         transforms=None,
         train=False,
         test=True
     )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    evaluator = CocoEvaluator(coco_gt=val_dataset.coco, iou_types=["bbox"])
    evaluator.iouThrs = [iou_threshold]


    # FINETUNED DETR
    if finetune:
        
        # Load the pre-trained DETR model
        model_name = 'facebook/detr-resnet-50'
        config = DetrConfig.from_pretrained(model_name)
        config.num_labels = num_classes + 1  # +1 for the background class
        config.num_queries = num_queries
        # model = DetrForObjectDetection.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True, use_safetensors=False)
        model = DetrForObjectDetection.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.to(device)
        model.load_state_dict(torch.load('outputs/best_finetuned_model.pth', map_location=torch.device('cpu')))
        model.eval()

    else:
        # SCRATCH DETR MODEL

        ### From Scratch DETR
        # num_classes = 1+num_classes  # +1 for the background class
        # Load the saved model
        model = DETR(num_classes=num_classes, num_queries=num_queries, pretrained_weights_path='resnet50-0676ba61.pth')
        model.load_state_dict(torch.load('outputs/best_model.pth', map_location=torch.device('cpu')))
        model.to(device)
        model.eval()




    print("Running evaluation...")
    for idx, batch in enumerate(tqdm(val_loader)):
        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

        if finetune==False:
            # replace 'class_labels' with 'labels' 
            for label in labels:
                label['labels'] = label.pop('class_labels')
        
        # forward pass
        with torch.no_grad():
            if finetune:
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            else:
                outputs = model(pixel_values)    
            
        # print (outputs.keys())

        if finetune==False:
            #create a DETR Output object
            outputs = DetrObjectDetectionOutput(
                logits = outputs['pred_logits'],
                pred_boxes = outputs['pred_boxes']
            )
        
        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = feature_extractor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)            # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        # store the results in the COCO format in results.json
        with open('outputs/results.json', 'w') as f:
            json.dump(predictions, f)
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    print("Evaluation complete.")
    
if __name__ == '__main__':
    main()
