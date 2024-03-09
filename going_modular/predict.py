
import torch

import torchvision
from torchvision import transforms
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt

from going_modular import model_builder

def pred_and_plot_image(model_path: str,
                        image_path: str,
                        class_names: List[str] = None,
                        transform: transforms.Compose = None,
                        device: torch.device = "cpu"):
    """Makes a prediction on a target image with a trained model and
    plots the image and prediction

    Args:
        model_path: Path to the saved PyTorch model.
        image_path: Path to the image to make a prediction on.
        class_names: An optional list of class names for the target dataset.
        transform: An optional sequence of transformations to perform on the image.
        device: A target device to perform the model prediction on.
    
    Returns:
        None, plots an image and prediction.

    Example usage:
    pred_and_plot_image(model_path="models/model_0.pth",
                        image_path="data/pizza_steak_sushi/test/sushi/100810.jpg",
                        class_names=["pizza", "steak", "sushi"],
                        transform=data_transform,
                        device="cuda")
    """
    # Create new model
    # Hardcoding the model type and hidden_units because I can :)
    model = model_builder.TinyVGG(input_shape=3,
                                  hidden_units=10,
                                  output_shape=3)
    
    # Load the model from path
    model.load_state_dict(torch.load(f=model_path))

    # Load an image, divide by 255 to normalize
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255.

    # Add the batch dimension
    target_image = target_image.unsqueeze(0)

    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    # Put model on the target device
    model.to(device)

    model.eval()
    with torch.inference_mode():
        target_image_pred = model(target_image.to(device))

        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    # Plot the image
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # remove batch dimension, rearrange

    if class_names:
        target_label = class_names[torch.argmax(target_image_pred_probs)]
        title = f"Pred: {target_label} | Prob: {target_image_pred_probs.max():.3f}"
    else:
        title = f"Pred: {torch.argmax(target_image_pred_probs)} | Prob: {target_image_pred_probs.max():.3f}"    
    plt.title(title)
    plt.axis('off')  
