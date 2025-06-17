import torch
import torch.nn as nn
import torch.nn.functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from functools import partial

_tokenizer = _Tokenizer()

# ──────────────────────────────────────────────────────────────────────────────
# 1) PROMPT CONFIG (minimal)
# ──────────────────────────────────────────────────────────────────────────────
class PromptConfig:
    """
    Minimal config to drive PromptLearner exactly as in CLIPping-the-Deception.
    """
    def __init__(
        self,
        n_ctx=16,
        ctx_init="a photo of a",
        csc=False,
        class_token_position="end",
        image_size=224,
    ):
        # Simulate cfg.TRAINER.COOP
        self.TRAINER = type("Trainer", (), {})()
        self.TRAINER.COOP = type("COOP", (), {})()
        self.TRAINER.COOP.N_CTX = n_ctx
        self.TRAINER.COOP.CTX_INIT = ctx_init
        self.TRAINER.COOP.CSC = csc
        self.TRAINER.COOP.CLASS_TOKEN_POSITION = class_token_position

        # Simulate cfg.INPUT.SIZE
        self.INPUT = type("Input", (), {})()
        self.INPUT.SIZE = (image_size, image_size)


# ──────────────────────────────────────────────────────────────────────────────
# 2) TEXT ENCODER (from CLIPping-the-Deception)
# ──────────────────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # prompts:   [n_cls, seq_len, dim]
        # tokenized_prompts: [n_cls, seq_len]
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)      # [seq_len, n_cls, dim]
        x = self.transformer(x)    # [seq_len, n_cls, dim]
        x = x.permute(1, 0, 2)      # [n_cls, seq_len, dim]
        x = self.ln_final(x).type(self.dtype)

        # Gather features at the end-of-sequence token for each prompt
        eos_indices = tokenized_prompts.argmax(dim=-1)  # [n_cls]
        x = x[torch.arange(x.shape[0]), eos_indices]    # [n_cls, dim]
        x = x @ self.text_projection                     # [n_cls, 512]
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 3) PROMPT LEARNER (CoOp-style, minimal)
# ──────────────────────────────────────────────────────────────────────────────
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # typically 512

        # 1) Initialize context vectors (either from ctx_init or random)
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)  # [1, seq_len]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            # Take tokens 1 : 1+n_ctx as initial context
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                # class-specific context (rarely used)
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"  —  Tokens: {n_ctx}')
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx, 512) or (n_cls, n_ctx, 512)

        # 2) Build the full tokenized prompts: "<prefix> <class_name>."
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts], dim=0)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # SOS token prefix: embedding[:, :1, :]
        self.register_buffer("token_prefix", embedding[:, :1, :].clone())
        # CLS+EOS suffix: skip past the context region
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :].clone())

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # [n_cls, seq_len]
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        """
        Returns: prompts as a tensor of shape [n_cls, full_seq_len, dim]
        where full_seq_len = 1 (SOS) + n_ctx (soft tokens) + len(class_name) + rest + 1 (EOS).
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            # Expand to all classes if only one shared context
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, 512]

        prefix = self.token_prefix     # [n_cls, 1, 512]
        suffix = self.token_suffix     # [n_cls, seq_len - 1 - n_ctx, 512]

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half = self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                prefix_i = prefix[i : i + 1, :, :]                    # [1,1,512]
                class_i = suffix[i : i + 1, : self.name_lens[i], :]   # [1,name_len,512]
                suffix_i = suffix[i : i + 1, self.name_lens[i] :, :]  # [1,rest,512]
                ctx_i_h1 = ctx[i : i + 1, :half, :]                   # [1, half, 512]
                ctx_i_h2 = ctx[i : i + 1, half :, :]                   # [1, half, 512]
                prompt_i = torch.cat(
                    [prefix_i, ctx_i_h1, class_i, ctx_i_h2, suffix_i], dim=1
                )
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=0)

        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                prefix_i = prefix[i : i + 1, :, :]                    # [1,1,512]
                class_i = suffix[i : i + 1, : self.name_lens[i], :]   # [1,name_len,512]
                suffix_i = suffix[i : i + 1, self.name_lens[i] :, :]  # [1,rest,512]
                ctx_i = ctx[i : i + 1, :, :]                           # [1,n_ctx,512]
                prompt_i = torch.cat(
                    [prefix_i, class_i, ctx_i, suffix_i], dim=1
                )
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=0)

        else:
            raise ValueError(f"Unknown CLASS_TOKEN_POSITION: {self.class_token_position}")

        return prompts  # [n_cls, full_seq_len, 512]