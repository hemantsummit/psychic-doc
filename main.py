from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split
import imgkit
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
from typing import List
import json
from torchmetrics import Accuracy

image_paths = sorted(list(Path("images").glob("*/*.jpg")))
len(image_paths)

feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

def scale_bounding_box(box: List[int], width_scale: float, height_scale: float) -> List[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]



DOCUMENT_CLASSES = [p.name for p in list(Path("images").glob("*"))]

class DocumentClassificationDataset(Dataset):
    
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        json_path = image_path.with_suffix(".json")
        
        with json_path.open("r") as f:
            ocr_result = json.load(f)
            
        width_scale = (1000/width)%1000
        height_scale = (1000/height)%1000
        
        words = []
        boxes = []
        for row in ocr_result:
            boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
            words.append(row["word"])
            
    
        encoding = processor(
            image,
            words,
            boxes=boxes,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        label = DOCUMENT_CLASSES.index(image_path.parent.name)
        print(encoding["bbox"])
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            bbox=encoding["bbox"].flatten(end_dim=1),
            pixel_values=encoding["pixel_values"].flatten(end_dim=1),
            labels=torch.tensor(label, dtype=torch.long)
        )
    
train_images, test_images = train_test_split(image_paths, test_size=.15)
len(train_images), len(test_images)

train_dataset = DocumentClassificationDataset(train_images, processor)
test_dataset = DocumentClassificationDataset(test_images, processor)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=1
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=1
)

class ModelModule(pl.LightningModule):
    
    def __init__(self, n_classes: int):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=n_classes
        )
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        
    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(
            batch["input_ids"], 
            batch["attention_mask"], 
            batch["bbox"], 
            batch["pixel_values"],
            labels
        )
        loss = outputs.loss
        self.log("train_loss", loss)
        self.train_accuracy(outputs.logits, labels)
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self(
            batch["input_ids"], 
            batch["attention_mask"], 
            batch["bbox"], 
            batch["pixel_values"],
            labels
        )
        loss = outputs.loss
        self.log("val_loss", loss)
        self.val_accuracy(outputs.logits, labels)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.00001)
    
model_module = ModelModule(len(DOCUMENT_CLASSES))

model_checkpoint = ModelCheckpoint(
    filename="{epoch}-{step}-{val_loss:.4f}",
    save_last=True,
    save_top_k=3,
    monitor="val_loss",
    mode="min"
)

trainer = pl.Trainer(
    accelerator="gpu",
    precision=16,
    devices=1,
    max_epochs=4,
    callbacks=[
        model_checkpoint
    ]
)

trainer.fit(model_module, train_data_loader, test_data_loader)

