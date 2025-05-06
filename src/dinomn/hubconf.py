dependencies = ['torch']
import detection
import torch

# resnet18 is the name of entrypoint
def dinomn(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    DinoMN model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = detection.DetectionModel(device=device)
    model.load_state_dict(torch.load('../models/DinoMN_oversampled.pth'))
    model.to(device)
    return model