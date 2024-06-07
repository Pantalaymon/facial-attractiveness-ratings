
import os
from pathlib import Path
from PIL import Image
import random

import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt


def standardize_path(cell):
    # We only want the architecture of the path from {gender}.
    # That way later we can provide whatever root path we want for the dataset
    path = Path(cell)
    return "/".join(path.parts[-3:])

def is_image_loadable(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it is, indeed an image
        return True
    except (IOError, SyntaxError) as e:
        print(f'Bad file: {image_path}')
        return False

def filter_corrupted_images(df, images_dir):
    df['full_path'] = df['path'].apply(lambda p: os.path.join(images_dir, p))
    df['is_valid'] = df['full_path'].apply(is_image_loadable)
    df_cleaned = df[df['is_valid']].drop(columns=['full_path', 'is_valid']).reset_index(drop=True)
    return df_cleaned

def load_images(df, image_dir, num_images=5, gender=None, ethnicity=None):
    '''
     loads a set number of random images within the directory 
    '''
    # Extract gender and ethnicity from the path and add them as columns to the DataFrame
    gender_col = df['path'].apply(lambda x: x.split('/')[0])
    ethnicity_col = df['path'].apply(lambda x: x.split('/')[1])
    
    # Apply gender filter if specified
    if gender is not None:
        df = df[gender_col == gender]
    
    # Apply ethnicity filter if specified
    if ethnicity is not None:
        df = df[ethnicity_col.isin(ethnicity)]

    selected_df = df.sample(n=num_images)
    
    images = []
    real_scores = []
    for idx, row in selected_df.iterrows():
        img_path = os.path.join(image_dir, row['path'])

        image = Image.open(img_path).convert('RGB')
        images.append(image)
        real_scores.append(row['mean'])
    
    # Drop the temporary columns
    #df.drop(columns=['gender', 'ethnicity'], inplace=True)

    
    return images, real_scores

def display_images(images, real_scores, predicted_scores=None):
    max_per_row = 5
    num_images = len(images)
    num_rows = (num_images + max_per_row - 1) // max_per_row  # Calculate number of rows needed
    
    plt.figure(figsize=(15, 3 * num_rows))  # Adjust height based on the number of rows
    
    if predicted_scores is None:
        predicted_scores = [None] * len(real_scores)
    
    for i, (image, predicted_score, real_score) in enumerate(zip(images, predicted_scores, real_scores)):
        ax = plt.subplot(num_rows, max_per_row, i + 1)
        ax.imshow(image)
        
        if predicted_score is None:
            ax.set_xlabel(f'Rating: {real_score:.2f}', fontsize=12)
        else:
            ax.set_title(f'Predicted: {predicted_score:.2f}', fontsize=12)
            ax.set_xlabel(f'Real: {real_score:.2f}', fontsize=12)
        
        ax.set_xticks([])  # Hide x ticks
        ax.set_yticks([])  # Hide y ticks
        for spine in ax.spines.values():
            spine.set_visible(False)  # Hide the spines
    
    plt.tight_layout()
    plt.show()

def load_model(device):
    model = models.mobilenet_v2(pretrained=True)
    
    # Modify the classifier for regression
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    # Move the model to the GPU if available

    model = model.to(device)
    return model

def load_checkpoint(model, optimizer, path='models/best_model.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    return model, optimizer, start_epoch, best_val_loss

def predict_attractiveness(model, images, device='cuda'):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    images_tensor = torch.stack([transform(image) for image in images]).to(device)
    
    with torch.no_grad():
        outputs = model(images_tensor)
        scores = outputs.squeeze().cpu().numpy()
    
    return scores

