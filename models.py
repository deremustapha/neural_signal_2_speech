import math, time
import torch.nn.functional as F
import torch.nn as nn
import torch
import os, json, time, math, random
from typing import List, Dict, Tuple, Optional



class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.gelu(x + res)
    


class DownBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.res = ResBlock(c_in, c_out, stride=stride)
    def forward(self, x):
        return self.res(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=stride, padding=1)
        self.res = ResBlock(c_out, c_out, stride=1)
    def forward(self, x):
        x = self.up(x)
        return self.res(x)
    


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_codes=512, code_dim=128, decay=0.99, eps=1e-5, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost

        embed = torch.randn(num_codes, code_dim)
        self.register_buffer("codebook", embed)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embed", embed.clone())

    @torch.no_grad()
    def _ema_update(self, flat_x, codes):
        one_hot = F.one_hot(codes, num_classes=self.num_codes).type_as(flat_x)
        cluster_size = one_hot.sum(dim=0)
        ema_cluster_size = self.ema_cluster_size * self.decay + cluster_size * (1 - self.decay)
        embed_sum = flat_x.t() @ one_hot
        ema_embed = self.ema_embed * self.decay + embed_sum.t() * (1 - self.decay)
        n = ema_cluster_size.sum()
        cluster_size = (ema_cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.ema_cluster_size.copy_(ema_cluster_size)
        self.ema_embed.copy_(ema_embed)
        self.codebook.copy_(self.ema_embed / cluster_size.unsqueeze(1))

    def forward(self, z):
        # z: (B, C_lat, T_lat)
        B, C, T = z.shape
        flat = z.permute(0, 2, 1).contiguous().view(-1, C)       # (B*T, C)

        x2 = (flat ** 2).sum(dim=1, keepdim=True)                # (B*T, 1)
        e2 = (self.codebook ** 2).sum(dim=1).unsqueeze(0)        # (1, K)
        xe = flat @ self.codebook.t()                            # (B*T, K)
        dist = x2 + e2 - 2 * xe

        codes = dist.argmin(dim=1)                               # (B*T,)
        z_q = self.codebook[codes].view(B, T, C).permute(0, 2, 1).contiguous()

        if self.training:
            self._ema_update(flat, codes)

        vq_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()                             # straight-through
        codes = codes.view(B, T)                                  # integer ids per time step
        return z_q, codes, vq_loss
    


class Conv1dVAE(nn.Module):
    """
    1-D VAE for sEMG or acoustic features.
    - Input:  (B, C_in, T)
    - Output: reconstruction (B, C_out, T_out == T if configured symmetrically)

    Latent is a *sequence* z_t (B, C_latent, T_latent), good for tokenizers and downstream models.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: list = (64, 128, 256),
        latent_dim: int = 128,
        use_vq: bool = False,
        codebook_size: int = 512,
        beta_kl: float = 1.0,
    ):
        super().__init__()
        self.beta_kl = beta_kl
        self.use_vq = use_vq

        # Encoder
        enc_layers = []
        c_prev = in_channels
        for c in channels:
            enc_layers += [DownBlock(c_prev, c, stride=2)]
            c_prev = c
        self.encoder = nn.Sequential(*enc_layers)

        # Map to μ, logσ² (time-distributed 1x1 convs)
        self.to_mu = nn.Conv1d(channels[-1], latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(channels[-1], latent_dim, kernel_size=1)

        # Optional VQ on latents
        if self.use_vq:
            self.vq = VectorQuantizerEMA(num_codes=codebook_size, code_dim=latent_dim)

        # Decoder: mirror the encoder
        dec_layers = []
        c_prev = latent_dim
        for c in reversed(channels):
            dec_layers += [UpBlock(c_prev, c, stride=2)]
            c_prev = c
        self.decoder = nn.Sequential(*dec_layers)

        # Final projection to output feature channels
        self.to_out = nn.Conv1d(channels[0], out_channels, kernel_size=3, padding=1)
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)           # (B, C_enc, T_lat)
        mu = self.to_mu(h)            # (B, latent_dim, T_lat)
        logvar = self.to_logvar(h)    # (B, latent_dim, T_lat)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        h = self.decoder(z)
        y = self.to_out(h)
        return y

    def forward(self, x):
        """
        Returns:
          recon, mu, logvar, z_or_zq, tokens (or None), aux_losses (dict)
        """
        z, mu, logvar = self.encode(x)
        tokens = None
        aux_losses = {}

        if self.use_vq:
            # z_q, tokens, vq_loss = self.vq(z)
            z_q, tokens, vq_loss = self.vq(mu)
            aux_losses["vq_loss"] = vq_loss
            y_hat = self.decode(z_q)
            z_out = z_q
        else:
            y_hat = self.decode(z)
            z_out = z

        return y_hat, mu, logvar, z_out, tokens, aux_losses

    def extract_latent_tokens(self, x, detach=True):
        """
        For downstream SSL/tokenization:
          returns sequence latents as (B, T_lat, C_lat)
          if use_vq=True, returns integer token ids instead (B, T_lat)
        """
        with torch.no_grad():
            z, mu, logvar = self.encode(x)
            if self.use_vq:
                # z_q, codes, _ = self.vq(z)
                z_q, codes, _ = self.vq(mu)
                return codes  # (B, T_lat)
            if detach:
                z = z.detach()
            return z.permute(0, 2, 1).contiguous()  # (B, T_lat, C_lat)
        


class Conv1dTokenizer(nn.Module):
    """
    Encodes (B, T, C_in) -> latent sequence z (B, C_lat, T_lat)
    If use_vq=True: returns discrete codes (B, T_lat) and z_q for features.
    """
    def __init__(
        self,
        in_channels: int,
        channels=(64, 128, 256),
        latent_dim: int = 128,
        use_vq: bool = True,
        codebook_size: int = 512,
    ):
        super().__init__()
        self.use_vq = use_vq

        # Encoder pyramid (channels-first Conv1d expects (B, C, T))
        enc_layers = []
        c_prev = in_channels
        for c in channels:
            enc_layers.append(DownBlock(c_prev, c, stride=2))
            c_prev = c
        self.encoder = nn.Sequential(*enc_layers)

        # Project to latent_dim per time step (1x1 conv)
        self.to_latent = nn.Conv1d(channels[-1], latent_dim, kernel_size=1)

        if use_vq:
            self.vq = VectorQuantizerEMA(num_codes=codebook_size, code_dim=latent_dim)

    def forward(self, x_btc):
        """
        x_btc: (B, T, C_in)
        returns:
          if use_vq:  codes (B, T_lat), z_q (B, C_lat, T_lat), aux={'vq_loss': ...}
          else:       z (B, C_lat, T_lat), aux={}
        """
        # (B, T, C) -> (B, C, T)
        x = x_btc.transpose(1, 2).contiguous()
        h = self.encoder(x)                      # (B, C_enc, T_lat)
        z = self.to_latent(h)                    # (B, latent_dim, T_lat)
        if self.use_vq:
            z_q, codes, vq_loss = self.vq(z)
            return codes, z_q, {"vq_loss": vq_loss}
        else:
            return z, {}, {}



class TransformerModel(nn.Module):
    def __init__(self, model_size, audio_features,
                 num_phonemes, num_layers=6, nhead=8,
                 dim_feedforward=3072, dropout=0.2):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),  
            ResBlock(model_size, model_size, 2), 
            ResBlock(model_size, model_size, 2), 
        )
        self.w_raw_in = nn.Linear(model_size, model_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_size, nhead=nhead)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers)
        self.audio_out = nn.Linear(model_size, audio_features)
        self.phoneme_out = nn.Linear(model_size, num_phonemes)

    def forward(self, x_raw):

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:]
                x_raw[:,-r:,:] = 0


        x_raw = x_raw.transpose(1,2) 
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = self.w_raw_in(x_raw)
        x = x_raw
        x = x.transpose(0,1) 
        x = self.trans(x)
        x = x.transpose(0,1)



        return self.audio_out(x), self.phoneme_out(x)
    


class TemporalStack(nn.Module):
    """
    Multi-dilation temporal processing with depthwise separable convolutions
    Uses multiple dilation rates to capture temporal patterns at different scales
    Input/Output: (B, L, D)
    """
    def __init__(self, d_model: int, kernel_size: int = 5, dilations: Tuple[int, ...] = (1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for dilation in dilations:
            padding = ((kernel_size - 1) // 2) * dilation
            
            # Depthwise separable convolution block
            block = nn.Sequential(
                # Depthwise convolution - each channel processed independently
                nn.Conv1d(d_model, d_model, kernel_size, 
                         padding=padding, dilation=dilation, groups=d_model),
                nn.GELU(),
                # Pointwise convolution - mix channels
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )
            
            self.blocks.append(block)
            self.norms.append(nn.LayerNorm(d_model))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multiple dilated temporal blocks with residual connections"""
        residual = x
        
        for block, norm in zip(self.blocks, self.norms):
            # Pre-normalization and transpose for conv1d: (B, L, D) -> (B, D, L)
            normalized = norm(residual).transpose(1, 2)
            processed = block(normalized).transpose(1, 2)  # Back to (B, L, D)
            residual = residual + self.dropout(processed)  # Residual connection per dilation
        
        return residual


class FrequencyStack(nn.Module):
    """
    Multi-scale frequency processing using FFTs at different resolutions
    Combines full-resolution and half-resolution frequency analysis
    Input/Output: (B, L, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Project complex FFT features (2*D real components) back to D dimensions
        self.proj_full = nn.Linear(2 * d_model, d_model)
        self.proj_half = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _apply_fft_projection(self, x: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        """Apply real FFT and project complex features to real space"""
        # Real FFT along time dimension
        fft_result = torch.fft.rfft(x, dim=1)  # (B, L_freq, D) complex
        
        # Convert complex to real representation
        real_features = torch.view_as_real(fft_result)  # (B, L_freq, D, 2)
        B, L_freq, D, _ = real_features.shape
        real_features = real_features.reshape(B, L_freq, 2 * D)  # (B, L_freq, 2D)
        
        # Project back to original dimension
        return projection(real_features)  # (B, L_freq, D)

    def _resize_to_target_length(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resize tensor to target length by padding or cropping"""
        B, current_length, D = tensor.shape
        
        if current_length < target_length:
            # Pad with zeros
            padding = torch.zeros(B, target_length - current_length, D, 
                                device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=1)
        else:
            # Crop to target length
            return tensor[:, :target_length, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-scale frequency analysis"""
        B, L, D = x.shape
        
        # Full-resolution frequency analysis
        freq_full = self._apply_fft_projection(x, self.proj_full)
        
        # Half-resolution frequency analysis (captures slower envelopes)
        x_downsampled = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True
        ).transpose(1, 2)  # Downsample by factor of 2
        freq_half = self._apply_fft_projection(x_downsampled, self.proj_half)
        
        # Resize both frequency representations to original length
        freq_full = self._resize_to_target_length(freq_full, L)
        freq_half = self._resize_to_target_length(freq_half, L)
        
        # Combine multi-scale frequency features
        return self.dropout(freq_full + freq_half)


class TimeFrequencyBlock(nn.Module):
    """
    Advanced time-frequency fusion block supporting multiple combination strategies
    Processes input through both temporal and frequency stacks, then fuses results
    """
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 kernel_size: int = 5,
                 dilations: Tuple[int, ...] = (1, 2, 4),
                 combine_method: str = "cross"):
        super().__init__()
        
        valid_methods = {"cross", "gated_sum"}
        if combine_method not in valid_methods:
            raise ValueError(f"combine_method must be one of {valid_methods}, got {combine_method}")
        
        self.combine_method = combine_method
        
        # Temporal and frequency processing stacks
        self.temporal_stack = TemporalStack(d_model, kernel_size, dilations, dropout)
        self.frequency_stack = FrequencyStack(d_model, dropout)
        
        if combine_method == "cross":
            # Cross-attention fusion: time queries attend to frequency keys/values
            self.query_norm = nn.LayerNorm(d_model)
            self.key_value_norm = nn.LayerNorm(d_model)
            self.cross_attention = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
        elif combine_method == "gated_sum":
            # Learnable gated combination with per-feature weights
            self.time_gate = nn.Parameter(torch.ones(d_model))
            self.freq_gate = nn.Parameter(torch.ones(d_model))
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse temporal and frequency representations"""
        # Process through both pathways
        time_features = self.temporal_stack(x)
        freq_features = self.frequency_stack(x)
        
        if self.combine_method == "cross":
            # Cross-attention fusion
            queries = self.query_norm(time_features)
            key_values = self.key_value_norm(freq_features)
            
            attended_output, _ = self.cross_attention(queries, key_values, key_values)
            attended_output = self.dropout(attended_output)
            
            # Residual connection with time features and final normalization
            return self.output_norm(attended_output + time_features)
        
        else:  # gated_sum
            # Learnable weighted combination
            gated_combination = (time_features * self.time_gate + 
                               freq_features * self.freq_gate)
            gated_combination = self.dropout(gated_combination)
            return self.norm(gated_combination)
        


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence modeling
    Adds position information to input embeddings
    """
    def __init__(self, d_model: int, max_length: int = 100000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal pattern
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)
    


class TokenChoiceRouter(nn.Module):
    """
    Adaptive routing mechanism that predicts optimal recursion depth per token
    Uses Gumbel-Softmax for differentiable discrete sampling during training
    """
    def __init__(self, 
                 d_model: int, 
                 max_depth: int, 
                 hidden_dim: Optional[int] = None, 
                 temperature: float = 1.0):
        super().__init__()
        
        self.max_depth = max_depth
        self.temperature = temperature
        hidden_dim = hidden_dim or d_model
        
        # Router network: input -> hidden -> depth logits
        self.norm = nn.LayerNorm(d_model)
        self.router_network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_depth)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict recursion depth for each token
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            depths: Chosen depth for each token (B, L) in range [1, max_depth]
            logits: Raw router logits (B, L, max_depth) for auxiliary losses
        """
        # Normalize input and compute routing logits
        normalized_input = self.norm(x)
        logits = self.router_network(normalized_input)  # (B, L, max_depth)
        
        if self.training:
            # Gumbel-Softmax for differentiable discrete sampling
            gumbel_noise = -torch.empty_like(logits).exponential_().log()
            gumbel_logits = (logits + gumbel_noise) / self.temperature
            soft_assignment = F.softmax(gumbel_logits, dim=-1)
            
            # Get hard assignment for forward pass
            hard_indices = soft_assignment.argmax(dim=-1)
            hard_assignment = F.one_hot(hard_indices, num_classes=self.max_depth).type_as(soft_assignment)
            
            # Straight-through estimator: hard forward pass, soft gradients
            assignment = (hard_assignment - soft_assignment).detach() + soft_assignment
            depths = hard_indices + 1  # Convert from 0-based to 1-based indexing
            
        else:
            # Deterministic assignment during evaluation
            depths = logits.argmax(dim=-1) + 1
        
        return depths, logits


class RecursiveTransformerBlock(nn.Module):
    """
    Transformer block designed for recursive application with shared weights
    Uses pre-normalization for better training stability
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention components with pre-normalization
        self.attention_norm = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        
        # Feed-forward network components with pre-normalization
        self.ffn_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply one recursive step of transformer processing
        
        Args:
            x: Input tensor (B, L, D)
            attn_mask: Attention mask (L, L) or (B*H, L, L)
            key_padding_mask: Key padding mask (B, L)
        """
        # Self-attention with pre-normalization and residual connection
        attention_input = self.attention_norm(x)
        attention_output, _ = self.self_attention(
            attention_input, attention_input, attention_input,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.attention_dropout(attention_output)
        
        # Feed-forward with pre-normalization and residual connection
        ffn_input = self.ffn_norm(x)
        ffn_output = self.feed_forward(ffn_input)
        x = x + self.ffn_dropout(ffn_output)
        
        return x


class MixtureOfRecursionsEncoder(nn.Module):
    """
    Mixture-of-Recursions encoder implementing adaptive computation
    Each token can choose its own optimal recursion depth dynamically
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 max_depth: int = 6,
                 dropout: float = 0.1,
                 router_hidden_dim: Optional[int] = None,
                 router_temperature: float = 1.0):
        super().__init__()
        
        self.max_depth = max_depth
        
        # Shared transformer block applied recursively
        self.transformer_block = RecursiveTransformerBlock(d_model, n_heads, dropout)
        
        # Router for predicting optimal recursion depth per token
        self.depth_router = TokenChoiceRouter(
            d_model, max_depth, router_hidden_dim, router_temperature
        )

    def forward(self, 
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mixture-of-recursions processing
        
        Args:
            x: Input tensor (B, L, D)
            attn_mask: Attention mask
            key_padding_mask: Padding mask
            
        Returns:
            output: Final representations (B, L, D)
            depths: Chosen recursion depths (B, L)
            router_logits: Router logits for auxiliary losses (B, L, max_depth)
        """
        # Predict optimal recursion depth for each token
        depths, router_logits = self.depth_router(x)
        
        # Apply recursive computation up to maximum depth
        current_representations = x
        
        for recursion_level in range(1, self.max_depth + 1):
            # Apply transformer block to all tokens
            updated_representations = self.transformer_block(
                current_representations, 
                attn_mask=attn_mask, 
                key_padding_mask=key_padding_mask
            )
            
            # Selective update: only tokens with depth >= current level get updated
            should_continue = (depths >= recursion_level).unsqueeze(-1)  # (B, L, 1)
            current_representations = torch.where(
                should_continue, updated_representations, current_representations
            )
        
        return current_representations, depths, router_logits

    def compute_router_auxiliary_losses(self, 
                                      router_logits: torch.Tensor,
                                      depths: torch.Tensor,
                                      target_avg_depth: float = 3.0,
                                      load_balance_weight: float = 0.01) -> torch.Tensor:
        """
        Compute auxiliary losses to guide router training
        
        Args:
            router_logits: Raw router outputs (B, L, max_depth)
            depths: Chosen depths (B, L)
            target_avg_depth: Target average recursion depth
            load_balance_weight: Weight for load balancing loss
            
        Returns:
            Combined auxiliary loss
        """
        # Load balancing: encourage uniform distribution across depths
        depth_distribution = torch.bincount(depths.flatten(), minlength=self.max_depth)
        depth_probabilities = depth_distribution.float() / depths.numel()
        uniform_distribution = torch.ones_like(depth_probabilities) / self.max_depth
        
        load_balance_loss = F.kl_div(
            depth_probabilities.log(), 
            uniform_distribution, 
            reduction='sum'
        )
        
        # Average depth control: maintain reasonable computational budget
        current_avg_depth = depths.float().mean()
        target_tensor = torch.tensor(target_avg_depth, device=depths.device)
        avg_depth_loss = F.mse_loss(current_avg_depth, target_tensor)
        
        return load_balance_weight * load_balance_loss + avg_depth_loss


# ---------- Main EMG-to-Phoneme Model ----------

class MoREMG2Phoneme(nn.Module):
    """
    End-to-end EMG to Phoneme/Acoustic feature model with Mixture-of-Recursions
    
    Architecture Pipeline:
    1. Input projection: (B, C_lat, L) -> (B, L, D)
    2. Time-frequency processing: Multi-scale temporal and spectral analysis
    3. Positional encoding: Add sequence position information
    4. MoR encoder: Adaptive-depth recursive processing
    5. Output heads: Generate acoustic and phoneme predictions
    """
    def __init__(self,
                 # Input/output dimensions
                 latent_dim: int,
                 n_phones: int,
                 audio_features: int = 80,
                 
                 # Model architecture
                 d_model: int = 512,
                 n_layers: int = 6,  # Used as max recursion depth
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 
                 # Time-frequency processing parameters
                 tf_kernel_size: int = 5,
                 tf_dilations: Tuple[int, ...] = (1, 2, 4),
                 tf_combine_method: str = "cross",  # "cross" or "gated_sum"
                 
                 # Mixture-of-Recursions parameters
                 mor_router_hidden: Optional[int] = None,
                 mor_router_temperature: float = 1.0,
                 mor_target_depth: float = 3.0,
                 mor_load_balance_weight: float = 0.01):
        super().__init__()
        
        # Store configuration for loss computation
        self.d_model = d_model
        self.mor_target_depth = mor_target_depth
        self.mor_load_balance_weight = mor_load_balance_weight
        
        # Input projection: map latent features to model dimension

        self.conv_blocks = nn.Sequential(
            ResBlock(8, latent_dim, 2),  # input (8, seq_len*8=1600); output (model_size, seq_len*4=800)
            ResBlock(latent_dim, latent_dim, 2), # input (model_size, seq_len*4=800); output (model_size, seq_len*2=400)
            ResBlock(latent_dim, latent_dim, 2), # input (model_size, seq_len*2=400); output (model_size, seq_len=200)
        )


        self.input_projection = nn.Linear(latent_dim, d_model)
        
        # Time-frequency processing: multi-scale temporal and spectral analysis
        self.time_freq_processor = TimeFrequencyBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            kernel_size=tf_kernel_size,
            dilations=tf_dilations,
            combine_method=tf_combine_method
        )
        
        # Positional encoding: add sequence position information
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Mixture-of-Recursions encoder: adaptive-depth processing
        self.mor_encoder = MixtureOfRecursionsEncoder(
            d_model=d_model,
            n_heads=n_heads,
            max_depth=n_layers,
            dropout=dropout,
            router_hidden_dim=mor_router_hidden,
            router_temperature=mor_router_temperature
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(d_model)
        self.acoustic_head = nn.Linear(d_model, audio_features)
        self.phoneme_head = nn.Linear(d_model, n_phones)

    def forward(self, 
                z_q_bcl: torch.Tensor,
                x_raw: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete EMG-to-phoneme pipeline
        
        Args:
            z_q_bcl: Quantized latent representations (B, C_lat, L)
            attn_mask: Attention mask (L, L)
            key_padding_mask: Padding mask (B, L)
            
        Returns:
            acoustic_output: Acoustic feature predictions (B, L, audio_features)
            phoneme_output: Phoneme predictions (B, L, n_phones)
            depths: Recursion depths used per token (B, L)
            router_logits: Router logits for auxiliary losses (B, L, max_depth)
        """
        # Step 1: Reshape and project input
        # (B, C_lat, L) -> (B, L, C_lat) -> (B, L, D)
        #print(f'Before transpose: {z_q_bcl.shape}')

        if self.training:
            # print(f'Raw input shape before augmentation: {x_raw.shape}')
            r = random.randrange(8)
            if r > 0:
                x_raw[:,:-r,:] = x_raw[:,r:,:] # shift left r
                x_raw[:,-r:,:] = 0

        # print(f'Raw input shape after augmentation: {x_raw.shape}')
        x_raw = x_raw.transpose(1,2)
        # print(f'After transpose: {x_raw.shape}')
        x_raw = self.conv_blocks(x_raw)
        # print(f'After Raw conv blocks: {x_raw.shape}')
        x_raw = x_raw.transpose(1,2)
        # print(f'After Raw conv blocks transpose: {x_raw.shape}')

        x_q = z_q_bcl.transpose(1, 2)
        # print(f'After  Toekn EMbed transpose: {x_q.shape}')

        x = x_raw + x_q


        x = self.input_projection(x)
        #print(f'After input projection: {x.shape}')
        
        # Step 2: Time-frequency processing
        x = self.time_freq_processor(x)
        #print(f'After time-frequency processing: {x.shape}')
        
        # Step 3: Add positional encoding
        x = self.positional_encoding(x)
        #print(f'After positional encoding: {x.shape}')
        
        # Step 4: Mixture-of-Recursions encoding
        x, depths, router_logits = self.mor_encoder(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        
        #print(f'After MoR encoding: {x.shape}, depths: {depths.shape}, router_logits: {router_logits.shape}')
        # Step 5: Final normalization and output generation
        x = self.output_norm(x)
        #print(f'After output normalization: {x.shape}')
        acoustic_output = self.acoustic_head(x)
        phoneme_output = self.phoneme_head(x)
        #print(f'Acoustic output: {acoustic_output.shape}, Phoneme output: {phoneme_output.shape}')
        
         # Return predictions and auxiliary information
        
        #return acoustic_output, phoneme_output, depths, router_logits
        return acoustic_output, phoneme_output

    def compute_total_loss(self,
                          # Predictions
                          acoustic_pred: torch.Tensor,
                          phoneme_pred: torch.Tensor,
                          # Targets
                          acoustic_target: torch.Tensor,
                          phoneme_target: torch.Tensor,
                          # Router outputs
                          router_logits: torch.Tensor,
                          depths: torch.Tensor,
                          # Loss weights
                          acoustic_weight: float = 1.0,
                          phoneme_weight: float = 1.0,
                          auxiliary_weight: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Compute comprehensive loss including main tasks and auxiliary router losses
        
        Args:
            acoustic_pred: Predicted acoustic features (B, L, audio_features)
            phoneme_pred: Predicted phonemes (B, L, n_phones)
            acoustic_target: Target acoustic features
            phoneme_target: Target phonemes
            router_logits: Router logits for auxiliary loss
            depths: Chosen recursion depths
            acoustic_weight: Weight for acoustic loss
            phoneme_weight: Weight for phoneme loss
            auxiliary_weight: Weight for auxiliary router loss
            
        Returns:
            total_loss: Combined weighted loss
            loss_breakdown: Dictionary with individual loss components
        """
        # Main task losses
        acoustic_loss = F.mse_loss(acoustic_pred, acoustic_target)
        
        # Reshape for cross-entropy: (B*L, n_phones)
        phoneme_pred_flat = phoneme_pred.reshape(-1, phoneme_pred.size(-1))
        phoneme_target_flat = phoneme_target.reshape(-1)
        phoneme_loss = F.cross_entropy(phoneme_pred_flat, phoneme_target_flat)
        
        # Auxiliary router losses
        router_aux_loss = self.mor_encoder.compute_router_auxiliary_losses(
            router_logits, depths, 
            self.mor_target_depth, 
            self.mor_load_balance_weight
        )
        
        # Combine all losses
        total_loss = (
            acoustic_weight * acoustic_loss +
            phoneme_weight * phoneme_loss +
            auxiliary_weight * router_aux_loss
        )
        
        # Create detailed loss breakdown for monitoring
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'acoustic_loss': acoustic_loss.item(),
            'phoneme_loss': phoneme_loss.item(),
            'router_aux_loss': router_aux_loss.item(),
            'avg_recursion_depth': depths.float().mean().item(),
            'max_recursion_depth': depths.max().item(),
            'min_recursion_depth': depths.min().item(),
        }
        
        return total_loss, loss_breakdown

    def get_depth_statistics(self, depths: torch.Tensor) -> dict:
        """Get detailed statistics about recursion depth usage"""
        depth_counts = torch.bincount(depths.flatten(), minlength=self.mor_encoder.max_depth)
        depth_percentages = (depth_counts.float() / depths.numel() * 100).tolist()
        
        return {
            'depth_distribution': depth_counts.tolist(),
            'depth_percentages': depth_percentages,
            'avg_depth': depths.float().mean().item(),
            'std_depth': depths.float().std().item(),
        }



def create_emg2phoneme_model(
    latent_dim: int = 256,
    n_phones: int = 50,
    audio_features: int = 80,
    d_model: int = 512,
    n_layers: int = 6,
    combine_method: str = "cross"
) -> MoREMG2Phoneme:
    """
    Factory function to create EMG-to-phoneme model with sensible defaults
    
    Args:
        latent_dim: Dimension of input latent features
        n_phones: Number of phoneme classes
        audio_features: Number of acoustic features (e.g., mel coefficients)
        d_model: Model hidden dimension
        n_layers: Maximum recursion depth
        combine_method: Time-frequency fusion method ("cross" or "gated_sum")
    
    Returns:
        Initialized model ready for training
    """
    return MoREMG2Phoneme(
        latent_dim=latent_dim,
        n_phones=n_phones,
        audio_features=audio_features,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=8,
        dropout=0.1,
        tf_kernel_size=5,
        tf_dilations=(1, 2, 4),
        tf_combine_method=combine_method,
        mor_router_temperature=1.0,
        mor_target_depth=3.0,
        mor_load_balance_weight=0.01
    )
