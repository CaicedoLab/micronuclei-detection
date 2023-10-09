import torch
import torch.nn.functional as F

import extractor
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
        
        self.feature_extractor = extractor.ViTExtractor('dino_vits8', stride, device=device)
        self.vit_model = trunc_vit_tiny(patch_size=1, in_chans=384, device=device)
        self.vit_model.to(device)
        self.classifier = torch.nn.Conv1d(192, 1, 1, 1)
        self.classifier.to(device)
        
    def forward(self, x):
        x = self.feature_extractor.preprocess_patch(x)
        x = self.feature_extractor.extract_descriptors(x, 11, 'key', False)
        B,_,_,C = x.shape
        T = 32
        x = x.transpose(3,1).reshape((B,C,T,T))
        x = self.vit_model(x)
        x = self.classifier(x.transpose(2,1))
        x = torch.reshape(x, (-1, T*T+1))[:,1:]
        x = F.softmax(x, dim=1)
        
        return x