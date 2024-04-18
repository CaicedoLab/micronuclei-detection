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
    
    def __init__(self, device, stride=8, finetune=False):
        super(DetectionModel, self).__init__()
        
        self.finetune = finetune
        
        # pretrained backbone has patch size 14 x 14, split into 14 row and columns
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device) # dinov2 vit small model

        # self.decoder = torch.nn.Sequential(
        #     # No activation after upscaling performs better
        #     torch.nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=(2,2), stride=2),
        #     torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1)),
        #     # Normalization here
        #     torch.nn.LayerNorm([192, 64, 64]), # nn.LayerNorm([C, H, W])
        #     torch.nn.ReLU(),
            
        #     torch.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=(2,2), stride=2),
        #     torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1)),
        #     # Normalization here
        #     torch.nn.LayerNorm([96, 128, 128]), # best
        #     torch.nn.ReLU(),
        # )
        # self.decoder.to(device)
        
        self.upscale1 = torch.nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=(2,2), stride=2)
        self.upscale1.to(device)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([192, 64, 64]), # nn.LayerNorm([C, H, W])
            torch.nn.ReLU(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([192, 64, 64]), # nn.LayerNorm([C, H, W])
            torch.nn.ReLU(),
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([192, 64, 64]), # nn.LayerNorm([C, H, W])
            torch.nn.ReLU(),
        )
        self.block1.to(device)
        self.block2.to(device)
        self.block3.to(device)
        
        self.upscale2 = torch.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=(2,2), stride=2)
        self.upscale2.to(device)
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([96, 128, 128]), # best
            torch.nn.ReLU(),
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([96, 128, 128]), # best
            torch.nn.ReLU(),
        )
        self.block6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1)),
            torch.nn.LayerNorm([96, 128, 128]), # best
            torch.nn.ReLU(),
        )
        self.block4.to(device)
        self.block5.to(device)
        self.block6.to(device)
        
        # classification layer
        self.classifier = torch.nn.Conv2d(in_channels=96, out_channels=2, kernel_size=(1,1))
        self.classifier.to(device)
                
    def forward(self, x):
        # Skip Connections
        if self.finetune:
            x = torch.nn.functional.interpolate(x, (448,448))
            x = self.feature_extractor.forward_features(x)['x_norm_patchtokens']
            B,T,C = x.shape # Batch, Token size * Toekn size, Channel
            W,H = 32,32
            x = x.reshape(B,W,H,C).permute(0,3,1,2)
            # x = self.decoder(x)
            
            x = self.upscale1(x)
            residual1 = x
            x = self.block1(x)
            residual2 = x
            x = self.block2(x) + residual1
            x = self.block3(x) + residual2
            
            x = self.upscale2(x)
            residual3 = x
            x = self.block4(x)
            residual4 = x
            x = self.block5(x) + residual3
            x = self.block6(x) + residual4
            
            x = self.classifier(x)
        else:
            with torch.no_grad():
                x = torch.nn.functional.interpolate(x, (448,448))
                x = self.feature_extractor.forward_features(x)['x_norm_patchtokens']
            B,T,C = x.shape # Batch, Token size * Toekn size, Channel
            W,H = 32,32
            x = x.reshape(B,W,H,C).permute(0,3,1,2)
            # x = self.decoder(x)
            
            x = self.upscale1(x)
            residual1 = x
            x = self.block1(x)
            residual2 = x
            x = self.block2(x) + residual1
            x = self.block3(x) + residual2
            
            x = self.upscale2(x)
            residual3 = x
            x = self.block4(x)
            residual4 = x
            x = self.block5(x) + residual3
            x = self.block6(x) + residual4
            
            x = self.classifier(x)
        
        return x
