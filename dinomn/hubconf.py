dependencies = ['torch']
import detection
import torch
from torch.hub import load_state_dict_from_url

def dinomn(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    DinoMN model
    """
    # Call the model, load pretrained weights
    device = 'cuda' if torch.cuda.is_avalaible() else 'cpu'
    url = 'https://github.com/CaicedoLab/micronuclei-detection/releases/download/v0.0.1/DinoMN.pth'
    model = detection.DetectionModel(device=device)
    
    if pretrained:
        checkpoint = load_state_dict_from_url(url, map_location=device) 
        model.load_state_dict(checkpoint)
        
    model.to(device)
    return model