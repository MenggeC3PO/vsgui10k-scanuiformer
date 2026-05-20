import torch
import torch.nn as nn
import timm


class ResidualMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(hidden_dim or d_model * 2)
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class ConvGNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        y = self.drop(self.act(self.norm(self.conv(x))))
        return x + y if self.use_skip else y


class StrongCropEncoder(nn.Module):
    """Small but stronger CNN for UI/target crops.

    Compared with uionly3's 3 plain convs, this adds normalization, residual
    blocks, and deeper channels while keeping output shape compatible: [B, 256, 1, 1].
    """
    def __init__(self, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNBlock(3, 32, stride=2, dropout=dropout),
            ConvGNBlock(32, 32, stride=1, dropout=dropout),
            ConvGNBlock(32, 64, stride=2, dropout=dropout),
            ConvGNBlock(64, 64, stride=1, dropout=dropout),
            ConvGNBlock(64, 128, stride=2, dropout=dropout),
            ConvGNBlock(128, 128, stride=1, dropout=dropout),
            ConvGNBlock(128, 256, stride=2, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x)


class HybridTemporalTargetDurationDecoderModel(nn.Module):
    """UI-only v4-strong scanpath decoder with target visual cue, timestep encoding,
    UI-target similarity, candidate-target relevance, FOUND prediction, and candidate-conditioned STOP prediction.

    Inputs:
      - GUI image patches from ViT
      - UI element tokens: geometry + type + crop image
      - fixation history
      - target visual crop image
      - current rollout timestep

    Outputs:
      - pred_xy: next fixation coordinate in normalized [0, 1]
      - pred_dur: predicted next fixation duration feature
      - pred_stop_logit: stop logit, trained with BCEWithLogitsLoss
      - aux: intermediate predictions for movement/duration diagnostics/losses
    """

    def __init__(
        self,
        vit_name: str,
        pretrained: bool,
        cue_vocab_size: int,
        ui_type_vocab_size: int,
        history_len: int,
        max_scanpath_len: int = 20,
        ui_geom_dim: int = 4,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.1,
        ui_memory_scale: float = 1.0,
        freeze_patch_backbone: bool = False,
        target_crop_size: int = 48,
        max_delta: float = 0.65,
        duration_output: str = "softplus",
        use_patch_memory: bool = True,
        use_target_aware_stop: bool = False,
        use_ui_target_similarity: bool = True,
        use_candidate_stop: bool = True,
        ui_num_layers: int = 2,
        patch_num_layers: int = 1,
        state_refine_layers: int = 2,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.max_scanpath_len = int(max_scanpath_len)
        self.d_model = int(d_model)
        self.ui_memory_scale = float(ui_memory_scale)
        self.freeze_patch_backbone = bool(freeze_patch_backbone)
        self.target_crop_size = int(target_crop_size)
        self.max_delta = float(max_delta)
        self.duration_output = str(duration_output).lower()
        self.use_patch_memory = bool(use_patch_memory)
        self.use_target_aware_stop = bool(use_target_aware_stop)
        self.use_ui_target_similarity = bool(use_ui_target_similarity)
        self.use_candidate_stop = bool(use_candidate_stop)
        self.ui_num_layers = int(ui_num_layers)
        self.patch_num_layers = int(patch_num_layers)
        self.state_refine_layers = int(state_refine_layers)

        # ============================================================
        # 1) Patch memory branch
        # ============================================================
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        if self.freeze_patch_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        vit_embed_dim = self.vit.patch_embed.proj.out_channels
        self.patch_proj = nn.Identity() if vit_embed_dim == d_model else nn.Linear(vit_embed_dim, d_model)

        self.patch_memory_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.patch_memory_transformer = nn.TransformerEncoder(patch_layer, num_layers=max(1, self.patch_num_layers))
        self.patch_memory_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 2) UI memory branch
        # ============================================================
        self.ui_geom_encoder = nn.Sequential(
            nn.Linear(ui_geom_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ui_type_embedding = nn.Embedding(ui_type_vocab_size, d_model)
        # Slightly stronger UI crop encoder than uionly1. This is still cheap,
        # but gives UI elements more visual capacity when full-image patches are removed.
        self.ui_crop_encoder = StrongCropEncoder(dropout=min(0.10, dropout))
        self.ui_crop_proj = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Optional scalar similarity between each UI crop embedding and the target token.
        # This helps UI memory expose candidate target elements to the decoder.
        ui_fusion_in_dim = d_model * 3 + 1
        self.ui_fusion = nn.Sequential(
            nn.Linear(ui_fusion_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ui_memory_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        ui_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.ui_memory_transformer = nn.TransformerEncoder(ui_layer, num_layers=max(1, self.ui_num_layers))
        self.ui_memory_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 3) Target cue branch: cue id + visual target crop
        # ============================================================
        self.cue_embedding = nn.Embedding(cue_vocab_size, d_model)
        self.target_crop_encoder = StrongCropEncoder(dropout=min(0.10, dropout))
        self.target_crop_proj = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.target_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.memory_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        # ============================================================
        # 4) Timestep / progress embedding
        # ============================================================
        # +2 gives safe indexing for max_scanpath_len and a possible overflow clamp.
        self.step_embedding = nn.Embedding(self.max_scanpath_len + 2, d_model)
        self.progress_mlp = nn.Sequential(
            nn.Linear(2, d_model),  # [t/max_len, remaining/max_len]
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.temporal_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # ============================================================
        # 5) History encoder
        # ============================================================
        self.history_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.history_pos_embedding = nn.Embedding(history_len, d_model)
        hist_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_transformer = nn.TransformerEncoder(hist_layer, num_layers=num_layers)
        self.history_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 6) UI-led dual attention
        # ============================================================
        self.ui_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ui_cross_norm1 = nn.LayerNorm(d_model)
        self.ui_cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.ui_cross_norm2 = nn.LayerNorm(d_model)

        self.patch_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.patch_cross_norm1 = nn.LayerNorm(d_model)
        self.patch_cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.patch_cross_norm2 = nn.LayerNorm(d_model)

        # ============================================================
        # 7) Search state pooling + output heads
        # ============================================================
        self.query_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.state_norm = nn.LayerNorm(d_model)

        self.state_context_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.state_refiner = nn.Sequential(*[
            ResidualMLP(d_model, hidden_dim=ff_dim, dropout=dropout)
            for _ in range(max(1, self.state_refine_layers))
        ])

        self.coord_abs_head = nn.Sequential(
            ResidualMLP(d_model, hidden_dim=ff_dim, dropout=dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),
        )
        self.delta_head = nn.Sequential(
            ResidualMLP(d_model, hidden_dim=ff_dim, dropout=dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Tanh(),
        )
        self.coord_gate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.dur_head = nn.Sequential(
            ResidualMLP(d_model, hidden_dim=ff_dim, dropout=dropout),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # Candidate relevance head: predicts which UI token is the likely target.
        # This is the main uionly3 change. It produces a target-candidate belief
        # from UI tokens instead of handing the STOP head the true target bbox.
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # FOUND estimates whether the predicted fixation is on/near the model's
        # own predicted UI target candidate.
        self.found_head = nn.Sequential(
            nn.Linear(d_model + 10, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        # STOP is conditioned on decoder state + candidate-relative geometry + FOUND,
        # not directly on true target bbox. This should make stopping less threshold-only.
        self.stop_head = nn.Sequential(
            nn.Linear(d_model + 11, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def apply_duration_output(self, raw_dur):
        """Convert raw duration head output to a non-negative duration feature.

        The training script normally uses log1p-normalized duration features, so
        softplus is a safe default: it is positive, unbounded, and stable.
        """
        if self.duration_output == "sigmoid":
            return torch.sigmoid(raw_dur)
        if self.duration_output == "relu":
            return torch.relu(raw_dur)
        if self.duration_output == "identity":
            return raw_dur
        # default: softplus
        return nn.functional.softplus(raw_dur)

    def encode_target_token(self, cue_id, target_crop_image):
        cue = self.cue_embedding(cue_id)
        crop = self.target_crop_encoder(target_crop_image).flatten(1)
        crop = self.target_crop_proj(crop)
        target_token = self.target_fusion(torch.cat([cue, crop], dim=-1))
        return target_token

    def encode_temporal_token(self, step_idx, device, batch_size):
        if step_idx is None:
            step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif not torch.is_tensor(step_idx):
            step_idx = torch.full((batch_size,), int(step_idx), dtype=torch.long, device=device)
        else:
            step_idx = step_idx.to(device=device, dtype=torch.long).view(-1)
            if step_idx.numel() == 1 and batch_size > 1:
                step_idx = step_idx.expand(batch_size)

        step_idx_clamped = step_idx.clamp(0, self.max_scanpath_len + 1)
        step_emb = self.step_embedding(step_idx_clamped)

        denom = float(max(1, self.max_scanpath_len - 1))
        progress = step_idx.float().clamp_min(0.0) / denom
        remaining = (self.max_scanpath_len - step_idx.float()).clamp_min(0.0) / float(max(1, self.max_scanpath_len))
        progress_feat = torch.stack([progress, remaining], dim=-1)
        progress_emb = self.progress_mlp(progress_feat)
        return self.temporal_fusion(torch.cat([step_emb, progress_emb], dim=-1))

    def encode_patch_memory(self, image, target_token):
        patch_tokens = self.vit.patch_embed(image)
        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            N = patch_tokens.size(1)
            pos_embed = self.vit.pos_embed[:, 1:1 + N, :]
            if pos_embed.size(-1) == patch_tokens.size(-1):
                patch_tokens = patch_tokens + pos_embed
        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = self.patch_memory_mlp(patch_tokens)
        gate = self.memory_gate(target_token)
        patch_tokens = patch_tokens * gate.unsqueeze(1)
        patch_tokens = patch_tokens + target_token.unsqueeze(1)
        patch_tokens = self.patch_memory_transformer(patch_tokens)
        return self.patch_memory_norm(patch_tokens)

    def encode_ui_crop(self, ui_crop_images):
        B, Ku, C, H, W = ui_crop_images.shape
        x = ui_crop_images.view(B * Ku, C, H, W)
        x = self.ui_crop_encoder(x).flatten(1)
        x = self.ui_crop_proj(x)
        return x.view(B, Ku, self.d_model)

    def encode_ui_memory(self, ui_geom, ui_type_id, ui_mask, ui_crop_images, target_token):
        geom_emb = self.ui_geom_encoder(ui_geom)
        type_emb = self.ui_type_embedding(ui_type_id)
        crop_emb = self.encode_ui_crop(ui_crop_images)
        if self.use_ui_target_similarity:
            target_norm = nn.functional.normalize(target_token, dim=-1).unsqueeze(1)
            crop_norm = nn.functional.normalize(crop_emb, dim=-1)
            sim = (crop_norm * target_norm).sum(dim=-1, keepdim=True)
        else:
            sim = torch.zeros((*crop_emb.shape[:2], 1), device=crop_emb.device, dtype=crop_emb.dtype)
        ui_tokens = torch.cat([geom_emb, type_emb, crop_emb, sim], dim=-1)
        ui_tokens = self.ui_fusion(ui_tokens)
        ui_tokens = self.ui_memory_mlp(ui_tokens)
        gate = self.memory_gate(target_token)
        ui_tokens = ui_tokens * gate.unsqueeze(1)
        ui_tokens = ui_tokens + target_token.unsqueeze(1)
        ui_key_padding_mask = (ui_mask == 0)
        ui_tokens = self.ui_memory_transformer(ui_tokens, src_key_padding_mask=ui_key_padding_mask)
        return self.ui_memory_norm(ui_tokens), ui_key_padding_mask

    def encode_history(self, history_xydur, history_mask=None):
        B, T, _ = history_xydur.shape
        device = history_xydur.device
        hist_tokens = self.history_encoder(history_xydur)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        hist_tokens = hist_tokens + self.history_pos_embedding(pos_ids)
        key_padding_mask = (history_mask == 0) if history_mask is not None else None
        hist_out = self.history_transformer(hist_tokens, src_key_padding_mask=key_padding_mask)
        if history_mask is not None:
            hist_out = hist_out.masked_fill(history_mask.unsqueeze(-1) == 0, 0.0)
        return self.history_norm(hist_out)

    def pool_sequence(self, x, mask=None):
        attn_logits = self.query_pool(x).squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        if mask is not None:
            attn = attn * mask.unsqueeze(-1)
            denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom
            x = x.masked_fill(mask.unsqueeze(-1) == 0, 0.0)
        return torch.sum(attn * x, dim=1)

    def get_last_xy(self, history_xydur, history_mask=None):
        if history_mask is None:
            return history_xydur[:, -1, :2]
        idx = history_mask.long().sum(dim=1).clamp_min(1) - 1
        # Because history is left-padded, the last valid token is normally at the
        # end. This generic gather is still safe.
        B = history_xydur.size(0)
        # Convert valid-count index to sequence index in left-padded sequence.
        # If there are V valid tokens in length T, valid start is T-V, last valid is T-1.
        last_idx = torch.full_like(idx, history_xydur.size(1) - 1)
        return history_xydur[torch.arange(B, device=history_xydur.device), last_idx, :2]



    def get_ui_centers_wh_area(self, ui_geom):
        """Extract UI center/size from either 10-D uionly3 or old 4-D geometry."""
        if ui_geom.size(-1) >= 10:
            centers = ui_geom[..., 4:6]
            wh = ui_geom[..., 6:8].clamp_min(1e-6)
            area = ui_geom[..., 8:9].clamp(0.0, 1.0)
        else:
            centers = ui_geom[..., 0:2]
            wh = ui_geom[..., 2:4].clamp_min(1e-6)
            area = (wh[..., 0:1] * wh[..., 1:2]).clamp(0.0, 1.0)
        return centers, wh, area

    def compute_candidate_belief(self, ui_memory, ui_geom, ui_mask):
        logits = self.candidate_head(ui_memory).squeeze(-1)
        logits = logits.masked_fill(ui_mask <= 0, -1e9)
        probs = torch.softmax(logits, dim=1)
        centers, wh, area = self.get_ui_centers_wh_area(ui_geom)
        candidate_xy = torch.sum(probs.unsqueeze(-1) * centers, dim=1)
        candidate_wh = torch.sum(probs.unsqueeze(-1) * wh, dim=1)
        candidate_area = torch.sum(probs.unsqueeze(-1) * area, dim=1).squeeze(-1)
        confidence = probs.max(dim=1).values
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1)
        denom = torch.log(ui_mask.sum(dim=1).clamp_min(2.0))
        entropy_norm = (entropy / denom).clamp(0.0, 1.0)
        return logits, probs, candidate_xy, candidate_wh, candidate_area, confidence, entropy_norm

    def compute_candidate_rel_features(self, pred_xy, candidate_xy, candidate_wh, candidate_area, confidence, entropy_norm):
        dx = pred_xy[:, 0] - candidate_xy[:, 0]
        dy = pred_xy[:, 1] - candidate_xy[:, 1]
        center_dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
        w = candidate_wh[:, 0].clamp_min(1e-6)
        h = candidate_wh[:, 1].clamp_min(1e-6)
        # Soft inside candidate bbox approximated from center + wh.
        x1 = candidate_xy[:, 0] - 0.5 * w
        y1 = candidate_xy[:, 1] - 0.5 * h
        x2 = candidate_xy[:, 0] + 0.5 * w
        y2 = candidate_xy[:, 1] + 0.5 * h
        px = pred_xy[:, 0]
        py = pred_xy[:, 1]
        dx_out = torch.maximum(torch.maximum(x1 - px, px - x2), torch.zeros_like(px))
        dy_out = torch.maximum(torch.maximum(y1 - py, py - y2), torch.zeros_like(py))
        bbox_dist = torch.sqrt(dx_out * dx_out + dy_out * dy_out + 1e-8)
        inside_soft = torch.sigmoid((0.05 - bbox_dist) / 0.02)
        return torch.stack([
            dx, dy, dx.abs(), dy.abs(), center_dist, bbox_dist,
            confidence, entropy_norm, candidate_area, inside_soft
        ], dim=-1)

    def compute_target_rel_features(self, pred_xy, target_bbox_norm):
        """Return differentiable target-relative geometry features.

        Features: dx, dy, |dx|, |dy|, center_dist, bbox_dist, w, h, area, inside_soft.
        All coordinates are normalized. This does not use GT scanpath length.
        """
        if target_bbox_norm is None:
            B = pred_xy.size(0)
            return torch.zeros(B, 10, device=pred_xy.device, dtype=pred_xy.dtype)

        bbox = target_bbox_norm.to(device=pred_xy.device, dtype=pred_xy.dtype)
        x1 = bbox[:, 0].clamp(0.0, 1.0)
        y1 = bbox[:, 1].clamp(0.0, 1.0)
        x2 = bbox[:, 2].clamp(0.0, 1.0)
        y2 = bbox[:, 3].clamp(0.0, 1.0)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = (x2 - x1).clamp_min(1e-6)
        h = (y2 - y1).clamp_min(1e-6)
        dx = pred_xy[:, 0] - cx
        dy = pred_xy[:, 1] - cy
        center_dist = torch.sqrt(dx * dx + dy * dy + 1e-8)

        # Distance outside bbox; zero if inside.
        px = pred_xy[:, 0]
        py = pred_xy[:, 1]
        dx_out = torch.maximum(torch.maximum(x1 - px, px - x2), torch.zeros_like(px))
        dy_out = torch.maximum(torch.maximum(y1 - py, py - y2), torch.zeros_like(py))
        bbox_dist = torch.sqrt(dx_out * dx_out + dy_out * dy_out + 1e-8)
        inside_soft = torch.sigmoid((0.05 - bbox_dist) / 0.02)
        area = (w * h).clamp(0.0, 1.0)
        return torch.stack([
            dx, dy, dx.abs(), dy.abs(), center_dist, bbox_dist, w, h, area, inside_soft
        ], dim=-1)

    def forward(
        self,
        image,
        cue_id,
        target_crop_image,
        history_xydur,
        ui_geom,
        ui_type_id,
        ui_mask,
        ui_crop_images,
        target_bbox_norm=None,
        history_mask=None,
        step_idx=None,
        return_aux: bool = True,
    ):
        B = image.size(0)
        device = image.device

        target_token = self.encode_target_token(cue_id, target_crop_image)
        temporal_token = self.encode_temporal_token(step_idx, device=device, batch_size=B)
        context_token = target_token + temporal_token

        patch_memory = None
        if self.use_patch_memory:
            patch_memory = self.encode_patch_memory(image, context_token)

        ui_memory, ui_key_padding_mask = self.encode_ui_memory(
            ui_geom=ui_geom,
            ui_type_id=ui_type_id,
            ui_mask=ui_mask,
            ui_crop_images=ui_crop_images,
            target_token=context_token,
        )
        ui_memory = self.ui_memory_scale * ui_memory
        candidate_logits, candidate_probs, candidate_xy, candidate_wh, candidate_area, candidate_confidence, candidate_entropy = self.compute_candidate_belief(
            ui_memory=ui_memory,
            ui_geom=ui_geom,
            ui_mask=ui_mask,
        )

        hist_out = self.encode_history(history_xydur, history_mask=history_mask)
        query_seq = hist_out + context_token.unsqueeze(1)

        ui_attended, _ = self.ui_cross_attn(
            query=query_seq,
            key=ui_memory,
            value=ui_memory,
            key_padding_mask=ui_key_padding_mask,
            need_weights=False,
        )
        x = self.ui_cross_norm1(query_seq + ui_attended)
        x = self.ui_cross_norm2(x + self.ui_cross_ffn(x))

        if self.use_patch_memory:
            patch_attended, _ = self.patch_cross_attn(query=x, key=patch_memory, value=patch_memory, need_weights=False)
            x = self.patch_cross_norm1(x + patch_attended)
            x = self.patch_cross_norm2(x + self.patch_cross_ffn(x))

        search_state = self.pool_sequence(x, mask=history_mask)
        search_state = self.state_norm(search_state + temporal_token)
        candidate_token = torch.sum(candidate_probs.unsqueeze(-1) * ui_memory, dim=1)
        search_state = self.state_context_fusion(torch.cat([
            search_state, target_token, temporal_token, candidate_token
        ], dim=-1))
        search_state = self.state_refiner(search_state)

        pred_xy_abs = self.coord_abs_head(search_state)
        pred_delta = self.max_delta * self.delta_head(search_state)
        last_xy = self.get_last_xy(history_xydur, history_mask=history_mask)
        pred_xy_delta = (last_xy + pred_delta).clamp(0.0, 1.0)
        gate = self.coord_gate_head(search_state)
        pred_xy = (gate * pred_xy_abs + (1.0 - gate) * pred_xy_delta).clamp(0.0, 1.0)

        pred_dur_raw = self.dur_head(search_state).squeeze(-1)
        pred_dur = self.apply_duration_output(pred_dur_raw)
        target_rel = self.compute_target_rel_features(pred_xy, target_bbox_norm)
        candidate_rel = self.compute_candidate_rel_features(
            pred_xy=pred_xy,
            candidate_xy=candidate_xy,
            candidate_wh=candidate_wh,
            candidate_area=candidate_area,
            confidence=candidate_confidence,
            entropy_norm=candidate_entropy,
        )
        if self.use_candidate_stop:
            stop_rel = candidate_rel
        else:
            stop_rel = target_rel
        if not self.use_target_aware_stop and not self.use_candidate_stop:
            stop_rel = torch.zeros_like(stop_rel)
        found_logit = self.found_head(torch.cat([search_state, candidate_rel], dim=-1)).squeeze(-1)
        pred_stop_logit = self.stop_head(torch.cat([search_state, stop_rel, found_logit.unsqueeze(-1)], dim=-1)).squeeze(-1)

        aux = {
            "pred_xy_abs": pred_xy_abs,
            "pred_xy_delta": pred_xy_delta,
            "pred_delta": pred_xy - last_xy,
            "raw_delta_head": pred_delta,
            "coord_gate": gate.squeeze(-1),
            "pred_dur_raw": pred_dur_raw,
            "pred_dur": pred_dur,
            "last_xy": last_xy,
            "target_token": target_token,
            "temporal_token": temporal_token,
            "target_rel_features": target_rel,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "candidate_xy": candidate_xy,
            "candidate_wh": candidate_wh,
            "candidate_area": candidate_area,
            "candidate_confidence": candidate_confidence,
            "candidate_entropy": candidate_entropy,
            "candidate_token": candidate_token,
            "candidate_rel_features": candidate_rel,
            "found_logit": found_logit,
        }
        if return_aux:
            return pred_xy, pred_dur, pred_stop_logit, aux
        return pred_xy, pred_dur, pred_stop_logit


# Backward-compatible class aliases for old import names.
HybridTemporalTargetDecoderModel = HybridTemporalTargetDurationDecoderModel
HybridPlainDecoderModel = HybridTemporalTargetDurationDecoderModel
