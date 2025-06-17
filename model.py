import clip
import torch
import timm
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear, ReLU, MultiheadAttention, TransformerEncoder, TransformerEncoderLayer, Parameter, ModuleList, LayerNorm
from functools import partial
from clip_prompt_utils import (PromptConfig, PromptLearner, TextEncoder)

# --------------------------------------------------
# 1) Cross-attention decoder
# --------------------------------------------------
class CrossAttentionDecoder(Module):
    """
    A simple cross-attention block with a residual MLP.
    """
    def __init__(self, embed_dim, num_heads=8, xavier_kaiming_init=False, dropout=0.0, qkNorm=False):
        super().__init__()
        self.cross_attn = MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = LayerNorm(embed_dim)

        # LayerNorms for query and context (key/value)
        self.qkNorm = qkNorm
        if self.qkNorm:
            self.norm_q = LayerNorm(embed_dim)
            self.norm_k = LayerNorm(embed_dim)
        
        self.mlp = Sequential(
            Linear(embed_dim, embed_dim * 4),
            ReLU(inplace=True),
            Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = LayerNorm(embed_dim)

        if xavier_kaiming_init:
            self._init_weights()

    def forward(self, x, context):
        """
        x:       [B, T, D]   (latent queries) -> T = number of latent tokens
        context: [B, N, D]   (CLIP patch/token embeddings) -> N = number of tokens (patch embeddings from CLIP)
        """
        # Cross-attention
        residual = x

        # Normalize queries and context (keys/values)
        if self.qkNorm:
            x = self.norm_q(x)
            context = self.norm_k(context)

        # allow the latent queries to "look at" image tokens, update themselves with the info
        attended, _ = self.cross_attn(query=x, key=context, value=context)
        x = self.norm1(attended + residual)

        # Feed-forward (MLP)
        residual = x
        x = self.mlp(x)
        x = self.norm2(x + residual)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)

            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Center the output of the final MLP layer
        if isinstance(self.mlp[-1], Linear) and self.mlp[-1].bias is not None:
            nn.init.constant_(self.mlp[-1].bias, 0)

# --------------------------------------------------
# 2) Custom Forward Functions for Different Backbones
# --------------------------------------------------

# 2a) For ViT-B/32 or ViT-L/14
def custom_vit_forward(visual, x):
    """
    Monkey-patch forward for CLIP's ViT-based vision module.
    Returns:
      - global_feat:  [B, 512]  (final CLS after projection)
      - token_feats:  [B, N, 512] (all patch tokens after projection)
    """
    # 1) Standard patch embedding
    x = visual.conv1(x)                        # => [B, c, h, w]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # => [B, c, hw]
    x = x.permute(0, 2, 1)                     # => [B, hw, c]

    # 2) Insert the CLS token
    class_token = visual.class_embedding.to(x.dtype)
    cls = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) # [B, 1, c]
    x = torch.cat([class_token + cls, x], dim=1)  # => [B, hw+1, c]

    # 3) Add positional embeddings
    x = x + visual.positional_embedding[: x.shape[1], :].to(x.dtype)

    # 4) Optional LN before transformer
    if hasattr(visual, "ln_pre"):
        x = visual.ln_pre(x)

    # 5) Run the ViT transformer => shape [B, n_patches+1, 768] for ViT-B/32
    x = visual.transformer(x)

    # 6) Optional LN post
    if hasattr(visual, "ln_post"):
        x = visual.ln_post(x)

    # 7) If there's a projection (768 -> 512), apply it to *all* tokens
    #    so your patch tokens and CLS are 512-d
    if visual.proj is not None:
        x = x @ visual.proj  # => [B, n_patches+1, 512]

    # 8) CLS token => [B, 512] 
    global_feat = x[:, 0, :]

    return global_feat, x  # x is [B, n_patches+1, 512]

# 2b) For RN50
def custom_resnet_forward(visual, x):
    """
    Custom forward for CLIP's RN50 visual module.
    
    This function performs the following steps:
      1. Runs a custom stem on the input.
      2. Passes the result through layer1 and layer2.
      3. Uses layer3 to extract an intermediate feature map (x_l3).
      4. Passes x_l3 through layer4 and then applies the attention pooling (attnpool)
         to produce a global feature.
      5. Flattens x_l3 into a token sequence.
      6. Prepends the global feature as a CLS token to the token sequence.
      7. Applies a learnable projection (proj_reduction) to reduce from 1024-d to 512-d.
    
    Returns:
      - global_feat:  [B, 512]  (global image feature)
      - token_feats:  [B, N, 512]  (token embeddings; N = 1 + (H*W) from x_l3)
    """
    def stem(x):
        x = visual.relu1(visual.bn1(visual.conv1(x)))
        x = visual.relu2(visual.bn2(visual.conv2(x)))
        x = visual.relu3(visual.bn3(visual.conv3(x)))
        x = visual.avgpool(x)
        return x

    x = x.type(visual.conv1.weight.dtype)
    x = stem(x)
    x = visual.layer1(x)
    x = visual.layer2(x)
    x_l3 = visual.layer3(x)  # Extract intermediate feature map from layer3
    x = visual.layer4(x_l3)
    global_feat = visual.attnpool(x)  # Global feature; shape: [B, 1024]

    # Create token features from x_l3:
    # x_l3: [B, 1024, H, W] -> flatten spatial dims -> [B, H*W, 1024]
    token_feats = x_l3.flatten(2).transpose(1, 2)
    # Prepend global feature as the CLS token
    global_feat_unsq = global_feat.unsqueeze(1)  # [B, 1, 1024]
    token_feats = torch.cat([global_feat_unsq, token_feats], dim=1)  # [B, 1+H*W, 1024], 14x14+1=196

    # Apply a projection from 1024-d to 512-d so the rest of the model can remain unchanged.
    if not hasattr(visual, 'proj_reduction'):
        visual.proj_reduction = Linear(1024, 512).to(x.device)
    global_feat = visual.proj_reduction(global_feat)      # [B, 512]
    token_feats = visual.proj_reduction(token_feats)        # [B, 1+H*W, 512]

    return global_feat, token_feats

def custom_encode_image(self, image):
    """
    Overwrite the original encode_image to call our custom forward.
    """
    image = image.to(self.visual.conv1.weight.dtype)
    return self.visual.forward(image)

# 2c) For ConvNeXt (convnext_base_in22k from timm)
def custom_convnext_forward(model, x):
    """
    Custom forward for a ConvNeXt model.
    Uses model.forward_features to obtain a spatial feature map, then computes a global feature (via average pooling),
    flattens the feature map to create tokens, prepends the global feature as a CLS token,
    and applies a learnable projection (proj_reduction) to map from native dimension to 512.
    """
    features = model.forward_features(x)  # [B, C, H, W]
    global_feat = features.mean(dim=(-2, -1))  # [B, C]
    token_feats = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
    global_feat_unsq = global_feat.unsqueeze(1)  # [B, 1, C]
    token_feats = torch.cat([global_feat_unsq, token_feats], dim=1)  # [B, 1+H*W, C]
    if not hasattr(model, 'proj_reduction'):
        model.proj_reduction = Linear(features.shape[1], 512).to(x.device)
    global_feat = model.proj_reduction(global_feat)  # [B, 512]
    token_feats = model.proj_reduction(token_feats)  # [B, 1+H*W, 512]
    return global_feat, token_feats

def custom_convnext_forward_intermediate(model, x):
    """
    Custom forward for a ConvNeXt model (convnext_base_in22k) that extracts an intermediate feature map.
    This function:
      1. Applies the stem,
      2. Passes through stage 0 and stage 1,
      3. Extracts an intermediate feature map from stage 2,
      4. Passes that intermediate feature map through stage 3 to compute a global feature,
      5. Computes the global feature via average pooling,
      6. Flattens the intermediate feature map to obtain token features,
      7. Preprends the global feature as a CLS token,
      8. Applies a projection to map features to 512 dimensions.
    """
    # Initial stem processing
    x = model.stem(x)
    x = model.stages[0](x)
    x = model.stages[1](x)
    x_l3 = model.stages[2](x)
    x = model.stages[3](x_l3)

    # Compute global feature via average pooling over spatial dimensions
    global_feat = x.mean(dim=(-2, -1))  # shape: [B, C]
    
    # Flatten the intermediate feature map for token features
    token_feats = x_l3.flatten(2).transpose(1, 2)  # shape: [B, H*W, C]

    # Project global feature from 1024 to 512 dimensions
    if not hasattr(model, 'proj_global'):
        model.proj_global = Linear(global_feat.shape[-1], token_feats.shape[-1]).to(global_feat.device)
    global_feat = model.proj_global(global_feat)  # shape: [B, 512]

    # Prepend global feature as CLS token to the token features
    global_feat_unsq = global_feat.unsqueeze(1)  # shape: [B, 1, C]
    token_feats = torch.cat([global_feat_unsq, token_feats], dim=1)  # shape: [B, 1+H*W, C]
    
    # Ensure projection exists for mapping feature dimension to 512
    if not hasattr(model, 'proj_reduction'):
        model.proj_reduction = Linear(token_feats.shape[-1], 512).to(token_feats.device)

    global_feat = model.proj_reduction(global_feat)      # shape: [B, 512]
    token_feats = model.proj_reduction(token_feats)        # shape: [B, 1+H*W, 512]
    
    return global_feat, token_feats

def custom_convnext_encode_image(model, image):
    return model.forward(image)

# --------------------------------------------------
# 3) Intra-Latent Self-Attention
# --------------------------------------------------

class IntraLatentTransformer(Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.0):
        """
        A small self-attention 'transformer encoder' 
        that allows the T=timesteps to attend to each other.
        """
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # We'll keep shape as [B, T, dim]
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: shape [B, T, embed_dim]
        return self.transformer(x)

# --------------------------------------------------
# 4) The final classifier
# --------------------------------------------------

# 4a) Stacked and Separate processing model
class TemporalCLIPLatentClassifier(Module):
    def __init__(self, 
                 num_class=2,
                 clip_type="ViT-B/32",
                 num_timesteps=5,
                 num_heads=8,
                 self_attention_latents=False,
                 weighted_average=False,
                 single_decoder=False,
                 process_latents_separately=False,
                 decoders_per_timestep=1,
                 use_cls_token=False,
                 positional_embeddings=False,
                 xavier_kaiming_init=False,
                 return_clip_global_feats=False,
                 latent_visual_refinement=True, 
                 prompt_tuning=False,
                 qkNorm=False):
        """
        Temporal CLIP Classifier that uses intermediate latents and cross-attends
        them to the (now 512-d) patch tokens.
        """
        super().__init__()

        if clip_type == "convnext_base_in22k":
            self.clip_model = timm.create_model("convnext_base_in22k", pretrained=True)
            self.clip_model.forward = partial(custom_convnext_forward_intermediate, self.clip_model)
            self.clip_model.encode_image = partial(custom_convnext_encode_image, self.clip_model)
        else:
            self.clip_model, _ = clip.load(clip_type, device="cpu", jit=False)

            if clip_type == "RN50":
                self.clip_model.visual.forward = partial(custom_resnet_forward, self.clip_model.visual)
            elif clip_type in ["ViT-B/32", "ViT-L/14"]:
                self.clip_model.visual.forward = partial(custom_vit_forward, self.clip_model.visual)
            self.clip_model.encode_image = partial(custom_encode_image, self.clip_model)

        # Confirm embedding dimension by a dummy forward
        test_dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            gf, tokens = self.clip_model.encode_image(test_dummy)
        self.embed_dim = gf.shape[-1]  # = 512 for ViT-B/32

        # Dynamic projection for diffusion latent tokens
        self.query_proj = Linear(3136, self.embed_dim)  

        self.single_decoder = single_decoder
        self.num_timesteps = num_timesteps
        self.process_latents_separately = process_latents_separately
        self.use_cls_token = use_cls_token
        self.positional_embeddings = positional_embeddings
        self.xavier_kaiming_init = xavier_kaiming_init
        self.return_clip_global_feats = return_clip_global_feats 
        self.latent_visual_refinement = latent_visual_refinement
        self.qkNorm = qkNorm

        # If using a CLS token, create a learnable parameter.
        if self.use_cls_token:
            self.cls_token = Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings (learned)
        if not self.process_latents_separately and self.positional_embeddings:
            self.latent_pos_embedding = Parameter(torch.zeros(1, self.num_timesteps, self.embed_dim))
            nn.init.trunc_normal_(self.latent_pos_embedding, std=0.02)

        # Cross-attention decoders
        if self.process_latents_separately:
            self.decoders_per_timestep = decoders_per_timestep
            self.decoder_layers = ModuleList([
                ModuleList([
                    CrossAttentionDecoder(self.embed_dim, num_heads=num_heads, xavier_kaiming_init=self.xavier_kaiming_init, qkNorm=self.qkNorm)
                    for _ in range(self.decoders_per_timestep)
                ])
                for _ in range(self.num_timesteps)
            ])
        else:
            self.decoder_layers = ModuleList([
                CrossAttentionDecoder(self.embed_dim, num_heads=num_heads, xavier_kaiming_init=self.xavier_kaiming_init, qkNorm=self.qkNorm)
                for _ in range(1 if self.single_decoder else self.num_timesteps)
            ])

        # Intra-Latent Self-Attn: let timesteps talk to each other
        self.self_attention_latents = self_attention_latents
        if self_attention_latents:
            self.intra_latent_transformer = IntraLatentTransformer(
                embed_dim=self.embed_dim,
                num_heads=4, 
                num_layers=2, 
                dropout=0.0
            )

        # Weight-based pooling: a tiny gate that outputs a scalar per step
        self.weighted_average = weighted_average
        if weighted_average:
            self.timestep_gate = Linear(self.embed_dim, 1, bias=False)

        # Prompt tuning or fine-tuning
        self.prompt_tuning = prompt_tuning
        if self.prompt_tuning:
            cfg = PromptConfig(
                n_ctx=16,
                ctx_init="a photo of a",
                csc=False,
                class_token_position="end",
                image_size=224,
            )
            classnames = ["real", "fake"]
            self.prompt_learner = PromptLearner(cfg, classnames, self.clip_model)
            self.text_encoder = TextEncoder(self.clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        # Final classifier head: concatenates global image feature and refined latent
        self.fc = Linear(self.embed_dim * 2, num_class)

    def forward(self, images, latent_seq):
        """
        images:     [B, 3, 224, 224]
        latent_seq: [B, T, C, H_lat, W_lat]
                    e.g., (C=4, H_lat=28, W_lat=28) or (C=4, 32, 32)
        """
        B = images.shape[0]

        # 1) Run the monkey-patched CLIP => (global_feat=512, tokens=512)
        clip_global_feat, clip_token_feats = self.clip_model.encode_image(images) # clip_global_feat => [B, 512], clip_token_feats => [B, n_patches+1, 512]

        # If prompt_tuning, augment CLIP patch tokens with prompt tokens
        if self.prompt_tuning:
            # a) Build and encode prompts every forward
            prompts = self.prompt_learner()  # [n_cls=2, seq_len, D]
            text_features = self.text_encoder(prompts, self.tokenized_prompts)  # [2, D]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # b) Tile and concat onto patch tokens
            batch_text = text_features.unsqueeze(0).expand(B, -1, -1)  # [B,2,D]
            combined_context = torch.cat([clip_token_feats, batch_text], dim=1)
        else:
            combined_context = clip_token_feats  # [B, num_patches+1, D]

        # 2) Flatten latents => shape [B, T, C*H_lat*W_lat]
        B, T, C, H_lat, W_lat = latent_seq.shape
        flatten_dim = C * H_lat * W_lat
        latents_2d = latent_seq.view(B, T, flatten_dim)

        # 3) Project to embed_dim [flatten_dim -> 512]
        if self.query_proj.in_features != flatten_dim:
            self.query_proj = Linear(flatten_dim, self.embed_dim).to(latent_seq.device)
        latent_queries = self.query_proj(latents_2d)  # => [B, T, 512]

        # Apply positional embeddings (if not processing latents separately)
        if not self.process_latents_separately and self.positional_embeddings:
                latent_queries = latent_queries + self.latent_pos_embedding  # [B, T, D]

        # 4) Self-attention among the T latents first
        if self.self_attention_latents:
            latent_queries = self.intra_latent_transformer(latent_queries) # shape remains [B, T, embed_dim], but each step is fused

        if self.latent_visual_refinement:
            # 5 & 6) Cross-attention with decoders
            if self.single_decoder:
                # Single decoder for the entire sequence
                latent_queries = self.decoder_layers[0](latent_queries, combined_context)
            elif self.process_latents_separately:
                if self.use_cls_token:
                    # We'll loop over T, prepend CLS, run each step's decoder, gather the CLS outputs
                    refined_latents = []
                    for t_idx in range(T):
                        # shape [B,1,512] for this step
                        latent_t = latent_queries[:, t_idx : t_idx+1, :]  

                        # Expand the shared CLS token to batch size [B,1,512]
                        cls_tokens = self.cls_token.expand(B, -1, -1)

                        # Prepend CLS to the single-latent query => shape [B,2,512]
                        latent_t_with_cls = torch.cat([cls_tokens, latent_t], dim=1)

                        # Pass through the t-th decoder => output shape [B,2,512]
                        for decoder in self.decoder_layers[t_idx]:
                            refined_t = decoder(latent_with_cls, combined_context)

                        # Typically we treat refined_t[:,0,:] as the updated CLS embedding; shape => [B,1,512]
                        step_cls = refined_t[:, 0:1, :]
                        refined_latents.append(step_cls)

                    # Now we have a list of [B,1,512] for each step => cat => [B,T,512]
                    latent_queries = torch.cat(refined_latents, dim=1)
                else:
                    refined_latents = []
                    for t_idx in range(T):
                        latent_t = latent_queries[:, t_idx:t_idx+1, :]
                        for decoder in self.decoder_layers[t_idx]:
                            latent_t = decoder(latent_t, combined_context)
                        refined_latents.append(latent_t)
                    latent_queries = torch.cat(refined_latents, dim=1)
            else:
                # Default: apply decoder layers sequentially over the entire sequence.
                for decoder in self.decoder_layers:
                    latent_queries = decoder(latent_queries, combined_context)

        # 7) Aggregate latents over timesteps:
        if self.use_cls_token and not self.process_latents_separately:
            # When using a CLS token in the full sequence, take the first token.
            refined_query = latent_queries[:, 0, :]
        else:
            # In all other cases, aggregate across the timestep dimension.
            if self.weighted_average:
                scores = self.timestep_gate(latent_queries)      # [B, T, 1]
                weights = torch.softmax(scores.squeeze(-1), dim=1)  # [B, T]
                weights = weights.unsqueeze(-1)                   # [B, T, 1]
                refined_query = torch.sum(latent_queries * weights, dim=1)
            else:
                refined_query = torch.mean(latent_queries, dim=1)

        # 8) Concat [global_clip, refined_query], then classify
        combined_feat = torch.cat([clip_global_feat, refined_query], dim=1)
        logits = self.fc(combined_feat)  # => [B, num_class]

        if self.return_clip_global_feats:
            return logits, refined_query, clip_global_feat
        else:
            return logits, refined_query

# 4b) CLS aggregation model
class TemporalCLIPLatentClassifierSingleCLS(Module):
    def __init__(self, 
                 num_class=2,
                 clip_type="ViT-B/32",
                 num_timesteps=5,
                 num_heads=8,
                 cls_self_attn_layers=2, 
                 cls_self_attn_heads=4,
                 positional_embeddings=False,
                 xavier_kaiming_init=False,
                 return_clip_global_feats=False,
                 use_cls_token=True,
                 qkNorm=False):
        """
        Temporal CLIP Latent Classifier variant that decodes each timestep's latent with a dedicated decoder, then prepends a single 
        learnable CLS token to the entire sequence of decoded latents. The whole sequence is refined by a self-attention module, 
        and the final CLS token is used for classification. 
        """
        super().__init__()

        self.use_cls_token = use_cls_token 
        self.return_clip_global_feats = return_clip_global_feats 

        if clip_type == "convnext_base_in22k":
            self.clip_model = timm.create_model("convnext_base_in22k", pretrained=True)
            self.clip_model.forward = partial(custom_convnext_forward_intermediate, self.clip_model)
            self.clip_model.encode_image = partial(custom_convnext_encode_image, self.clip_model)
        else:
            self.clip_model, _ = clip.load(clip_type, device="cpu", jit=False)

            if clip_type == "RN50":
                self.clip_model.visual.forward = partial(custom_resnet_forward, self.clip_model.visual)
            elif clip_type in ["ViT-B/32", "ViT-L/14"]:
                self.clip_model.visual.forward = partial(custom_vit_forward, self.clip_model.visual)
            self.clip_model.encode_image = partial(custom_encode_image, self.clip_model)

        # Confirm embedding dimension by a dummy forward
        test_dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            gf, tokens = self.clip_model.encode_image(test_dummy)
        self.embed_dim = gf.shape[-1]  # = 512 for ViT-B/32

        # Dynamic projection for diffusion latent tokens
        self.query_proj = Linear(3136, self.embed_dim)  

        self.xavier_kaiming_init = xavier_kaiming_init
        self.qkNorm = qkNorm

        # Create one cross-attention decoder per timestep.
        self.decoder_layers = ModuleList([
            CrossAttentionDecoder(self.embed_dim, num_heads=num_heads, xavier_kaiming_init=self.xavier_kaiming_init, qkNorm=self.qkNorm)
            for _ in range(num_timesteps)
        ])

        # Learnable single CLS token.
        self.cls_token = Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.positional_embeddings = positional_embeddings
        if self.positional_embeddings:
            # [1, T, D] so it can broadcast across batch dimension
            self.latent_pos_embedding = Parameter(torch.zeros(1, num_timesteps, self.embed_dim))
            nn.init.trunc_normal_(self.latent_pos_embedding, std=0.02)
        
        # Self-attention module to refine the entire sequence (CLS token + T decoded tokens).
        self.cls_self_attn = IntraLatentTransformer(embed_dim=self.embed_dim,
                                                    num_heads=cls_self_attn_heads,
                                                    num_layers=cls_self_attn_layers)
        
        # Final classifier head: concatenates CLIP global image feature with the final CLS token.
        self.fc = Linear(self.embed_dim * 2, num_class)

    def forward(self, images, latent_seq):
        """
        Args:
            images: [B, 3, 224, 224] input images.
            latent_seq: [B, T, C, H_lat, W_lat] diffusion latent sequences.
            
        Returns:
            logits: [B, num_class] classification scores.
            final_cls: [B, embed_dim] the refined single CLS token.
        """
        B = images.shape[0]
        
        # 1) Use CLIP to get global image feature and token features.
        clip_global_feat, clip_token_feats = self.clip_model.encode_image(images)
        
        # 2) Flatten latent_seq: [B, T, C*H_lat*W_lat]
        B, T, C, H_lat, W_lat = latent_seq.shape
        flatten_dim = C * H_lat * W_lat
        latents_2d = latent_seq.view(B, T, flatten_dim)
    
        # 3) Project to embed_dim: [B, T, embed_dim]
        latent_queries = self.query_proj(latents_2d)
        
        # 4) Decode each timestep's latent with its own decoder.
        decoded_tokens = []
        for t in range(T):
            latent_t = latent_queries[:, t:t+1, :]  # [B, 1, embed_dim]
            # Decode latent_t using the t-th decoder with CLIP token features as context.
            refined_t = self.decoder_layers[t](latent_t, clip_token_feats)  # [B, 1, embed_dim]
            decoded_tokens.append(refined_t)
        # Concatenate all decoded tokens to get [B, T, embed_dim]
        decoded_seq = torch.cat(decoded_tokens, dim=1)

        # 5) Positional encodings
        if self.positional_embeddings:
            # shape: [B,T,D] + [1,T,D] => [B,T,D]
            decoded_seq = decoded_seq + self.latent_pos_embedding
            
        if self.use_cls_token:
            # 6) Prepend the single learnable CLS token (expanded to batch) to the decoded sequence.
            cls_tok = self.cls_token.expand(B, -1, -1)      # [B, 1, embed_dim]
            full_seq = torch.cat([cls_tok, decoded_seq], dim=1)  # [B, 1+T, embed_dim]
        
            # 7) Refine the full sequence with self-attention.
            attn_out = self.cls_self_attn(full_seq)  # [B, 1+T, embed_dim]

            # Extract the updated CLS token (position 0).
            refined_query = attn_out[:, 0, :]  # [B, embed_dim]
        else:
            attn_out = self.cls_self_attn(decoded_seq)
            refined_query = torch.mean(attn_out, dim=1)      
   
        # 8) Concatenate the final refined query with the global CLIP image feature and classify.
        combined_feat = torch.cat([clip_global_feat, refined_query], dim=1)
        logits = self.fc(combined_feat)

        if self.return_clip_global_feats:
            return logits, refined_query, clip_global_feat
        else:
            return logits, refined_query

# 4c) Latents only model
class LatentOnlyClassifier(Module):
    def __init__(self, 
                 num_class: int = 2,
                 num_timesteps: int = 5,
                 latent_channels: int = 4,
                 latent_h: int = 28,
                 latent_w: int = 28,
                 embed_dim: int = 512):
        """
        A simple classifier that uses only the diffusion latents:
        - Flattens each timestep's latent (CxHxW) to a vector
        - Projects to `embed_dim`
        - Averages over timesteps
        - Final linear layer to `num_class`
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        flatten_dim = latent_channels * latent_h * latent_w
        # learnable projection from flattened latent to embed_dim
        self.query_proj = Linear(flatten_dim, embed_dim)
        # final classifier head
        self.fc = Linear(embed_dim, num_class)

    def forward(self, latent_seq: torch.Tensor):
        """
        Args:
            latent_seq: [B, T, C, H, W]
        Returns:
            logits: [B, num_class]
        """
        B, T, C, H, W = latent_seq.shape
        # 1) flatten per timestep: [B, T, C*H*W]
        x = latent_seq.view(B, T, C * H * W)
        # 2) project each to embed_dim: [B, T, embed_dim]
        x = self.query_proj(x)
        # 3) aggregate across T (simple mean): [B, embed_dim]
        embed = x.mean(dim=1)
        # 4) classify
        logits = self.fc(embed)
        return logits, embed