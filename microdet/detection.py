import torch
import torch.nn.functional as F

import vision_transformer as vit

class TruncViT(vit.VisionTransformer):
    
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=torch.nn.LayerNorm, **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer)
        del(self.head)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x#[:, 0]
    
def trunc_vit_tiny(patch_size=16, **kwargs):
    model = TruncViT(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=vit.partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
    return model

class DetectionModel(torch.nn.Module):
    
    def __init__(self, device, stride=8):
        super(DetectionModel, self).__init__()
        
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device) # dinov2 vit small model
        # self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device) # dinov2 vit base model

        # With VIT-tiny
        #self.vit_model = trunc_vit_tiny(patch_size=1, in_chans=384, device=device)
        #self.vit_model.to(device)
        #self.classifier = torch.nn.Conv1d(192, 1, 1, 1)

        # With linear classifier
        # self.classifier = torch.nn.Conv2d(384, 1, (1,1)) # num of features for small model
        # self.classifier = torch.nn.Conv2d(768, 1, (1,1)) # num of features for base model
        # self.classifier.to(device)
        
        self.conv1_layer = torch.nn.Conv2d(in_channels=384, out_channels=768, kernel_size=(1,1))
        self.conv1_layer.to(device)
        self.relu1 = torch.nn.ReLU()
        self.relu1.to(device)


        self.conv2_layer = torch.nn.Conv2d(in_channels=768, out_channels=256, kernel_size=(1,1))
        self.conv2_layer.to(device)
        self.relu2 = torch.nn.ReLU()
        self.relu2.to(device)

        self.conv3_layer = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1))
        self.conv3_layer.to(device)
        self.relu3 = torch.nn.ReLU()
        self.relu3.to(device)

        # classification layer
        self.classifier = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1))
        self.classifier.to(device)

                
    def forward(self, x):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, (448,448))
            x = self.feature_extractor.forward_features(x)['x_norm_patchtokens']
        B,T,C = x.shape # 32, 1024, 384; Batch, Token, Channel
        W,H = 32,32

        # With VIT-tiny
        #x = x.permute(0,2,1).reshape((B,C,W,H))
        #x = self.vit_model(x)
        #x = self.classifier(x.permute(0,2,1))
        #x = torch.reshape(x, (-1, W*H+1))[:,1:]

        
        # With linear classifier
        # x (batch of tokens) shape: (32, 384, 32, 32), input image format for CNN (batch, # of channels, height, width)
        x = x.reshape(B,W,H,C).permute(0,3,1,2)
        
        # x = self.classifier(x) # classifier turn 384 channels to 1 here
        
        x = self.conv1_layer(x)
        x = self.relu1(x)
        x = self.conv2_layer(x)
        x = self.relu2(x)
        x = self.conv3_layer(x)
        x = self.relu3(x)
        x = self.classifier(x)

        x = torch.reshape(x, (B, W*H)) # reshape to Batch size (32) * # of Tokens (1024)
        #x = F.softmax(x, dim=1)
        
        return x
