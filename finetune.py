# train_finetune.py

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrConfig, DetrFeatureExtractor
from utils.utils import collate_fn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

from data.coco import CocoDetection
from data.coco import make_coco_transforms


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define the number of classes
    num_classes = 7
    batch_size = 4
    num_queries = 4
    
     # Initialize the feature extractor
    # feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        'facebook/detr-resnet-50',
        size=800,       # Ensure this is an integer
        max_size=1333   # Ensure this is an integer
        # max_size = 800
    )


    # Load the pre-trained DETR model
    model_name = 'facebook/detr-resnet-50'
    config = DetrConfig.from_pretrained(model_name)
    config.num_labels = num_classes + 1  # +1 for the background class
    config.num_queries = num_queries
    # model = DetrForObjectDetection.from_pretrained(model_name, config=config)
    model = DetrForObjectDetection.from_pretrained(
    model_name, 
    config=config, 
    # num_labels = num_classes + 1,
    ignore_mismatched_sizes=True
)
    # model.load_state_dict(torch.load('detr_final_finetuned_epoch_30.pth', map_location=torch.device('cpu')))
    model.to(device)


    train_dataset = CocoDetection(
        img_folder='data/BoneFractureData/train2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('train'),
        train=True
    )
    val_dataset = CocoDetection(
        img_folder='data/BoneFractureData/val2017/images',
        ann_folder='data/BoneFractureData/annotations',
        processor=feature_extractor,
        transforms=make_coco_transforms('val'),
        train=False
    )

    


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Define optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

    # #  freeze the backbone during training
    # for param in model.model.backbone.parameters():
    #     param.requires_grad = False
    
    best_val_loss = 10e5
    print("Starting training...")
    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True, position=0)
        
        # for i, (images, targets) in enumerate(progress_bar):
        for i,batch in enumerate(progress_bar):

            pixel_values = batch['pixel_values'].to(device)  # Extract pixel values
            pixel_mask = batch['pixel_mask'].to(device)      # Extract pixel mask
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]  # Process labels

           
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Optionally, perform validation and save checkpoints
        if (epoch +1) % 2 == 0 or epoch == 0:
            if (epoch+1)%10 == 0:
                torch.save(model.state_dict(), f'detr_7_finetuned_epoch_{epoch + 1}.pth')
            evaluate(model, val_loader, device, best_val_loss)

    # Save the final model
    # torch.save(model.state_dict(), 'detr_finetuned.pth')


def evaluate(model, data_loader, device, best_val_loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        # for images, targets in data_loader:
        for batch in data_loader:


            pixel_values = batch['pixel_values'].to(device)  # Extract pixel values
            pixel_mask = batch['pixel_mask'].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]


            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)

            # Compute loss
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), 'detr_finetuned_best.pth')
        print("Best Model saved with val loss: ", avg_loss)




if __name__ == '__main__':
    main()
