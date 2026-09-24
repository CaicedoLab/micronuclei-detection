import torch
import torch.nn.functional as F
import torch.nn as nn
# import vision_transformer as vit


# class TruncViT(vit.VisionTransformer):
    
#     def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., norm_layer=torch.nn.LayerNorm, **kwargs):
#         super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
#                  num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
#                  drop_path_rate, norm_layer)
#         del(self.head)

#     def forward(self, x):
#         x = self.prepare_tokens(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.norm(x)
#         return x#[:, 0]
    
# def trunc_vit_tiny(patch_size=16, **kwargs):
#     model = TruncViT(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, norm_layer=vit.partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

class DetectionModel(torch.nn.Module):
    
    def __init__(self, device, stride=8):
        super().__init__()
        
        # pretrained backbone has patch size 14 x 14, split into 14 row and columns
        self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(device) # dinov2 vit small model
        # self.feature_extractor = torch.hub.load(repo_or_dir='facebookresearch/dinov3', model='dinov3_vits16', weights = '/mnt/cephfs/mir/jcaicedo/projects/micronuclei_detection/dinov3_weights/dinov3_vits16_pretrain.pth').to(device) # dinov3
        
        def conv_block(in_channels, out_channels, kernel_size=(3,3), padding=(1,1), norm_shape=[96, 128, 128]):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                torch.nn.LayerNorm(norm_shape),
                torch.nn.ReLU()
            )
        
        self.upscale1 = torch.nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=(2,2), stride=2)
        self.upscale1.to(device)
        
        self.block1 = conv_block(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1), norm_shape=[192, 64, 64])
        self.block2 = conv_block(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1), norm_shape=[192, 64, 64])
        self.block3 = conv_block(in_channels=192, out_channels=192, kernel_size=(3,3), padding=(1,1), norm_shape=[192, 64, 64])
        
        self.block1.to(device)
        self.block2.to(device)
        self.block3.to(device)
        
        # here number of features (d_model) represents channels
        self.decoder_layer1 = torch.nn.TransformerDecoderLayer(d_model=192, nhead=8, activation='relu', batch_first=True)  # Assuming d_model is 192
        self.transformer_decoder1 = torch.nn.TransformerDecoder(self.decoder_layer1, num_layers=4)
        self.transformer_decoder1.to(device)

        self.projection1 = torch.nn.Linear(384, 192)
        self.projection1.to(device)
        
        self.upscale2 = torch.nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=(2,2), stride=2)
        self.upscale2.to(device)

        self.block4 = conv_block(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1), norm_shape=[96, 128, 128])
        self.block5 = conv_block(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1), norm_shape=[96, 128, 128])
        self.block6 = conv_block(in_channels=96, out_channels=96, kernel_size=(3,3), padding=(1,1), norm_shape=[96, 128, 128])
        
        self.block4.to(device)
        self.block5.to(device)
        self.block6.to(device)
        
        self.projection2 = torch.nn.Linear(192, 96)
        self.projection2.to(device)
        
        # classification layer
        self.classifier = torch.nn.Conv2d(in_channels=96, out_channels=2, kernel_size=(1,1))
        self.classifier.to(device)
                
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, (448,448)) # dinov2
        # x = torch.nn.functional.interpolate(x, (512, 512), mode='bicubic') # dinov3
        
        # finetuning, using frozen backbone: with torch.no_grad()
        x = self.feature_extractor.forward_features(x)['x_norm_patchtokens']
        
        B,T,C = x.shape # Batch, Token size * Toekn size, Channel
        H,W = 32,32
        
        memory = x # original image features B,1024,384
        x = x.reshape(B,H,W,C).permute(0,3,1,2) # B,384,32,32
        
        # Pixel Decoder Block 1
        x = self.upscale1(x)
        residual1 = x
        x = self.block1(x)
        residual2 = x
        x = self.block2(x) + residual1
        x = self.block3(x) + residual2 # x: 1,192,64,64
        
        # Transformer Decoder Block 1
        B,C,H,W = x.shape
        target = x.reshape(B,H*W,C)
        memory = self.projection1(memory)
        transformer_x = self.transformer_decoder1(tgt=target, memory=memory)
        # 1,4096,192
        
        # Pixel Decoder Block 2
        x = self.upscale2(x)
        residual3 = x
        x = self.block4(x)
        residual4 = x
        x = self.block5(x) + residual3
        x = self.block6(x) + residual4 # x: 1,96,128,128
        
        
        transformer_x = self.projection2(transformer_x)
        transformer_x = transformer_x.reshape(B,96,H,W)
        transformer_x = F.interpolate(x, (128,128))
        
        # Sum up information
        x = x + transformer_x
        
        x = self.classifier(x)
        
        return x




################# ------------------- UNet Model ------------------- #################
class _DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks (the core UNet unit)."""
 
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
 
 
class UNet2D(nn.Module):
    """
    Vanilla 2-D U-Net baseline for micronuclei / nuclei segmentation.
 
    Output shape:  (B, out_channels, H, W)  — same spatial size as the input.
    Output values: raw logits (no sigmoid / softmax applied).
                   Matches the convention used by DetectionModel so all loss
                   functions in train.py work without any changes.
 
    Args:
        in_channels  : number of input image channels (default 3 for RGB).
        out_channels : number of segmentation classes (default 2: MN + N).
        base_filters : channel count at the first encoder stage; doubles at
                       each subsequent stage (default 64 → 128 → 256 → 512).
        depth        : number of encoder / decoder stages (default 4).
        device       : torch.device — model is moved there inside __init__.
 
    Usage (drop-in replacement)::
 
        model = UNet2D(device=device)
        # identical forward signature to DetectionModel
        logits = model(x)   # x: (B, 3, H, W)
    """
 
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_filters: int = 64,
        depth: int = 4,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
 
        self.depth = depth
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # ──- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        ch_in = in_channels
        ch_out = base_filters
        for _ in range(depth):
            self.encoder_blocks.append(_DoubleConv(ch_in, ch_out))
            ch_in = ch_out
            ch_out = ch_out * 2

        # --- Bottleneck ---
        self.bottleneck = _DoubleConv(ch_in, ch_out)
 
        # ──- Decoder ---
        self.upconv_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        dec_in = ch_out
        for _ in range(depth):
            dec_out = dec_in // 2
            self.upconv_blocks.append(
                nn.ConvTranspose2d(dec_in, dec_out, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(_DoubleConv(dec_in, dec_out))
            dec_in = dec_out
 
        # --- Classification head (1×1 conv)---
        # Mirrors `self.classifier` in DetectionModel so that
        # MicronucleiModel.predict() can read `self.model.classifier.out_channels`
        self.classifier = nn.Conv2d(dec_in, out_channels, kernel_size=1)
 
        self.to(device)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]  # save original input size (256×256)
        x = torch.nn.functional.interpolate(x, (448,448), mode='bilinear', align_corners=False)
        
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
 
        x = self.bottleneck(x)
 
        # --- Decoder: upsample + skip + conv ---
        for upconv, dec, skip in zip(
            self.upconv_blocks, self.decoder_blocks, reversed(skips)
        ):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                               0, skip.shape[2] - x.shape[2]])
 
            x = torch.cat([skip, x], dim=1)  # channel-wise concat
            x = dec(x)
 
        # ──- Classification head ---
        x = self.classifier(x)
        x = F.interpolate(x, target_size, mode='bilinear', align_corners=False)  # back to 256×256
        
        return x
 