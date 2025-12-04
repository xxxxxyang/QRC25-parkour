import torch
import torch.nn as nn

class GatedRecurrentBelief(nn.Module):
    def __init__(self, base_backbone, env_cfg, policy_cfg, 
                 belief_dim=32, rnn_hidden=512, extero_dim=None) -> None:
        super().__init__()

        self.base_backbone = base_backbone   # depth encoder
        activation = nn.ELU()
        last_activation = nn.Tanh()
        # get extero output dim from depth backbone
        if extero_dim is None:
            if env_cfg is None:
                extero_dim = 32
            else:
                extero_dim = policy_cfg["scan_encoder_dims"][-1]
        proprio_dim = env_cfg.env.n_proprio if env_cfg else 53
        # combination MLP = g_e  
        # output_dim：belief_dim
        self.combination_mlp = nn.Sequential(
            nn.Linear(extero_dim + proprio_dim, 128),
            activation,
            nn.Linear(128, belief_dim)
        )
        # RNN Core
        self.rnn = nn.GRU(input_size=belief_dim,
                          hidden_size=rnn_hidden,
                          batch_first=True)
        # Encoder Gate Networks (g_a, g_b)
        self.g_a = nn.Sequential(
            nn.Linear(rnn_hidden, 128),
            activation,
            nn.Linear(128, extero_dim)       # output gate vector α size = extero_dim
        )
        self.g_b = nn.Sequential(
            nn.Linear(rnn_hidden, belief_dim),
            activation
        )
        # learnable projections only if extero_dim != belief_dim
        if extero_dim != belief_dim:
            self.extero_to_belief_proj = nn.Linear(extero_dim, belief_dim)
            self.gate_proj = nn.Linear(extero_dim, belief_dim)  # projects gate logits to belief_dim
        else:
            self.extero_to_belief_proj = None
            self.gate_proj = None
        # Decoder Modules (for training)
        self.belief_decoder = BeliefDecoder(rnn_hidden, extero_dim, env_cfg.env.n_scan)

        self.last_activation = last_activation
        self.hidden_states = None
        self.extero_dim = extero_dim
        self.belief_dim = belief_dim

    def forward(self, depth_image, proprioception):
        """
        Args:
            depth_image (Tensor): extero input (e.g. depth image) with shape [B, H, W].
            proprioception (Tensor): proprioceptive input with shape [B, proprio_dim].
        Returns:
            dict: {
                "belief": Tensor [B, belief_dim],         # belief state b_t
                "recon_extero": Tensor [B, extero_dim],   # reconstructed extero l_hat (for training)
                "gate": Tensor [B, extero_dim],           # gate α
                "raw_extero": Tensor [B, extero_dim],     # raw extero latent
            }
        Note:
            recon_extero is primarily used for reconstruction loss during training and can
            be ignored during inference.
        """
        # Encode extero (CNN)
        extero_latent = self.base_backbone(depth_image)     # [B, extero_dim]
        # Combine with proprio
        fusion_input = torch.cat((extero_latent, proprioception), dim=-1)
        fused = self.combination_mlp(fusion_input)          # [B, belief_dim]
        # Pass through GRU
        rnn_in = fused.unsqueeze(1)                         # [B, 1, belief_dim]
        b_prime, self.hidden_states = self.rnn(rnn_in, self.hidden_states)
        b_prime = b_prime.squeeze(1)                        # [B, hidden]
        # gate α = sigmoid(g_a(b_prime))
        gate_alpha = torch.sigmoid(self.g_a(b_prime))       # [B, extero_dim]
        # belief b_t = g_b(b′_t) + α * extero_latent
        b_proj = self.g_b(b_prime)                          # [B, belief_dim]
        # expand extero_latent to belief_dim if needed
        if extero_latent.size(-1) != self.belief_dim:
            # project extero_latent to match belief_dim
            extero_to_belief = self.extero_to_belief_proj(extero_latent)
            gate_alpha = torch.sigmoid(self.gate_proj(gate_alpha))  # [B, belief_dim]
        else:
            extero_to_belief = extero_latent
            gate_alpha = gate_alpha
        belief = b_proj + gate_alpha[:, :self.belief_dim] * extero_to_belief
        belief = self.last_activation(belief)
        # Decoder (reconstruction height map for training)
        # alpha * extero_latent + mlp(h_t)
        if self.belief_decoder is not None:
            recon_extero, dec_alpha = self.belief_decoder(b_prime, extero_latent)  # recon_extero: [B, scandot_dim] reconstuct scandot
        else:
            recon_extero, dec_alpha = None, None

        return {
            "belief": belief,   # [B, belief_dim]
            "recon_extero": recon_extero,
            "gate": gate_alpha,
            "dec_alpha": dec_alpha,
            "raw_extero": extero_latent
        }

    def reset(self):
        """Reset RNN hidden states (call at episode start)."""
        self.hidden_states = None

    def detach_hidden_states(self):
        """Detach hidden states to avoid backprop through entire episode."""
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

class BeliefDecoder(nn.Module):
    """
    h_t -> MLP1 -> alpha, 
    h_t -> MLP2 -> l_2, 
    recon: alpha * extero_latent + l_2
    """
    def __init__(self, hidden_dim, extero_dim, scandot_dim, mlp_hidden=128):
        super().__init__()
        self.extero_dim = extero_dim
        
        # MLP1: h_t -> alpha (sigmoid)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, extero_dim),
            nn.Sigmoid()  # [0,1] weight
        )
        
        # MLP2: h_t -> l_2 
        self.l2_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ELU(),
            nn.Linear(mlp_hidden, extero_dim)
        )
        
        # final_mlp
        self.final_mlp = nn.Sequential(
            nn.Linear(extero_dim, scandot_dim),
            nn.ELU()
        )

    def forward(self, h_t, extero_latent):
        """
        Args:
            h_t: [B, hidden_dim]
            extero_latent: [B, extero_dim]
        Returns:
            est_extero: [B, extero_dim] (recon)
            alpha: [B, extero_dim] (dec_alpha)
        """
        alpha = self.alpha_mlp(h_t)  # [B, extero_dim]
        l_2 = self.l2_mlp(h_t)      # [B, extero_dim]
        
        # alpha * extero_latent + l_2
        est_extero = alpha * extero_latent + l_2  # [B, extero_dim]
        
        # MLP Head
        est_extero = self.final_mlp(est_extero)
        
        return est_extero, alpha
    
# class BeliefDecoder(nn.Module):
#     """
#     Decoder that reconstructs exteroceptive info (and optionally privileged info).
#     It uses:
#       - cross-modal attention to align h_t with raw_extero
#       - an MLP / conv head to produce extero reconstruction
#       - gate_alpha to fuse raw_extero and decoded estimate (same gate as encoder)
#     """
#     def __init__(self, hidden_dim, belief_dim, extero_dim, 
#                  use_spatial_decoder=False, spatial_shape=None):
#         """
#         Args:
#             hidden_dim: rnn hidden dim (ht size)
#             belief_dim: final belief dim
#             extero_dim: extero latent dim (or flattened patch dim)
#             use_spatial_decoder: if True, decoder outputs a 2D height map via conv transpose
#             spatial_shape: tuple (H, W) required if use_spatial_decoder=True
#         """
#         super().__init__()
#         self.extero_dim = extero_dim
#         self.hidden_dim = hidden_dim
#         self.use_spatial_decoder = use_spatial_decoder
#         self.spatial_shape = spatial_shape

#         # cross-modal attention aligns h_t -> extero space
#         self.xattn = CrossModalAttention(hidden_dim, extero_dim, n_heads=4)

#         # MLP head to produce extero estimate (vector)
#         self.mlp_head = nn.Sequential(
#             nn.Linear(extero_dim, extero_dim),
#             nn.ELU(),
#             nn.Linear(extero_dim, extero_dim)
#         )

#         # Optionally build a conv decoder to produce spatial height map
#         if use_spatial_decoder:
#             H, W = spatial_shape
#             # project belief -> channels*H'*W' then upsample to HxW
#             self.to_feature = nn.Linear(belief_dim, 64 * (H//4) * (W//4))
#             self.conv_decoder = nn.Sequential(
#                 nn.Unflatten(1, (64, H//4, W//4)),
#                 nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # upsample x2
#                 nn.ELU(),
#                 nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # upsample x2
#                 nn.ELU(),
#                 nn.Conv2d(16, 1, kernel_size=3, padding=1)
#             )

#         # final small MLP to refine aligned vector -> extero vector
#         self.refine = nn.Sequential(
#             nn.Linear(extero_dim, extero_dim),
#             nn.ELU(),
#             nn.Linear(extero_dim, extero_dim)
#         )

#     def forward(self, h_t, belief, raw_extero, gate_alpha):
#         """
#         Args:
#             h_t: [B, hidden_dim] (RNN hidden or b_prime)
#             belief: [B, belief_dim] (final belief, could be used for spatial decode)
#             raw_extero: [B, extero_dim] (encoder output)
#             gate_alpha: [B, extero_dim] (sigmoid gate from encoder)
#         Returns:
#             est_extero: [B, extero_dim]  (estimated extero vector)
#             est_spatial: [B, 1, H, W] if spatial decode enabled
#         """
#         # 1) Cross-modal attention: align hidden -> extero space
#         attn_out = self.xattn(h_t, raw_extero)   # [B, extero_dim]

#         # 2) MLP refinement (from attention output)
#         dec_vec = self.mlp_head(attn_out)        # [B, extero_dim]
#         dec_vec = self.refine(dec_vec)           # [B, extero_dim]

#         # 3) Fuse using gate α (same gate as encoder)
#         #    Use element-wise blending: (1-α)*dec_vec + α*raw_extero
#         est_extero = (1.0 - gate_alpha) * dec_vec + gate_alpha * raw_extero

#         # 4) Optional: spatial reconstruction from belief (if required)
#         est_spatial = None
#         if self.use_spatial_decoder:
#             feat = self.to_feature(belief)      # [B, C*(H//4)*(W//4)]
#             est_spatial = self.conv_decoder(feat)  # [B, 1, H, W]

#         return est_extero, est_spatial


# class CrossModalAttention(nn.Module):
#     """
#     简化的 cross-attention module：
#     - Query 从 hidden (h_t) 投影
#     - Key/Value 来自 raw_extero (向量或 flattened spatial)
#     - 输出与 extero_dim 对齐
#     """
#     def __init__(self, hidden_dim, extero_dim, n_heads=4):
#         super().__init__()
#         assert extero_dim % n_heads == 0
#         self.n_heads = n_heads 
#         self.head_dim = extero_dim // n_heads

#         self.q_proj = nn.Linear(hidden_dim, extero_dim) # Query
#         self.k_proj = nn.Linear(extero_dim, extero_dim) # Key
#         self.v_proj = nn.Linear(extero_dim, extero_dim) # Value
#         self.out_proj = nn.Linear(extero_dim, extero_dim)   # output
#         self.scale = self.head_dim ** -0.5

#     def forward(self, h, extero):
#         # h: [B, hidden_dim]
#         # extero: [B, extero_dim]  (可以是flatten的空间patch或latent vector)
#         Q = self.q_proj(h).view(h.size(0), self.n_heads, self.head_dim)  # [B, H, d]
#         K = self.k_proj(extero).view(h.size(0), self.n_heads, self.head_dim)
#         V = self.v_proj(extero).view(h.size(0), self.n_heads, self.head_dim)

#         # 点积注意力（per-head scalar）
#         attn_logits = (Q * K).sum(-1) * self.scale  # [B, H]
#         attn = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, H, 1]

#         weighted = (attn * V).view(h.size(0), -1)  # [B, extero_dim]
#         out = self.out_proj(weighted)  # [B, extero_dim]
#         return out

def update_belief_actor(self, scandots_batch, actions_student_batch=None, actions_teacher_batch=None, recon_batch=None):
    if not self.if_depth:
        return 0.0, 0.0, 0.0

    device = self.device

    # ============================
    # 1. Compute raw task losses
    # ============================

    # Reconstruction MSE loss
    if recon_batch is not None and scandots_batch is not None:
        recon_loss_raw = nn.functional.mse_loss(
            recon_batch,
            scandots_batch.detach()
        )
    else:
        recon_loss_raw = torch.tensor(0.0, device=device)

    # Action imitation L2 loss
    if actions_student_batch is not None and actions_teacher_batch is not None:
        l2_per_sample = torch.norm(
            actions_student_batch - actions_teacher_batch.detach(),
            p=2,
            dim=-1
        )
        action_loss_raw = l2_per_sample.mean()
    else:
        action_loss_raw = torch.tensor(0.0, device=device)

    # Vector: raw losses (these represent true task performance)
    L_raw = torch.stack([recon_loss_raw, action_loss_raw])

    # =======================================
    # 2. Initialize EMA for loss normalization
    # =======================================
    if not hasattr(self, "loss_ema"):
        self.ema_decay = 0.99
        self.loss_ema = {
            "recon": None,
            "action": None
        }

    # Update EMAs
    def update_ema(name, value):
        v = value.detach()
        if self.loss_ema[name] is None:
            self.loss_ema[name] = v.clone()
        else:
            self.loss_ema[name] = (
                self.ema_decay * self.loss_ema[name] +
                (1 - self.ema_decay) * v
            )

    update_ema("recon", recon_loss_raw)
    update_ema("action", action_loss_raw)

    # ===============================
    # 3. Normalized losses for GradNorm
    # ===============================

    recon_loss_norm = recon_loss_raw / (self.loss_ema["recon"] + 1e-8)
    action_loss_norm = action_loss_raw / (self.loss_ema["action"] + 1e-8)

    L = torch.stack([recon_loss_norm, action_loss_norm])  # normalized losses

    # =================================================
    # 4. Initialize reference losses (for GradNorm ratios)
    # =================================================
    if self.initial_losses is None:
        self.initial_losses = L.detach().clone()

    # ================================
    # 5. Compute gradient norms G_i
    # ================================
    shared_params = self.gradnorm_shared_params
    G_list = []

    for i in range(L.numel()):
        grads = torch.autograd.grad(
            outputs=(self.task_weights[i] * L[i]),
            inputs=shared_params,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )

        flat_grads = []
        for g, p in zip(grads, shared_params):
            if g is None:
                flat_grads.append(torch.zeros_like(p).view(-1))
            else:
                flat_grads.append(g.contiguous().view(-1))

        if len(flat_grads) == 0:
            g_norm = torch.tensor(0.0, device=device)
        else:
            g_norm = torch.norm(torch.cat(flat_grads), p=2)

        G_list.append(g_norm)

    G = torch.stack(G_list)
    G_avg = G.mean().detach()

    # ==========================================
    # 6. Compute target gradient magnitudes
    # ==========================================
    loss_ratio = L.detach() / (self.initial_losses + 1e-12)
    r_i = loss_ratio / (loss_ratio.mean() + 1e-12)

    target_G = G_avg * (r_i ** self.gradnorm_alpha)

    # ==========================================
    # 7. Compute GradNorm loss and update weights
    # ==========================================
    gradnorm_loss = torch.abs(G - target_G).sum()

    self.task_weight_optimizer.zero_grad()
    gradnorm_loss.backward(retain_graph=True)
    self.task_weight_optimizer.step()

    # Normalize weights to sum to K
    with torch.no_grad():
        w = self.task_weights
        K = float(L.numel())
        w[:] = K * w / (w.sum() + 1e-12)

    # ==========================================
    # 8. Update encoder + actor using weighted loss
    # ==========================================
    weighted_loss = (self.task_weights * L).sum()

    self.depth_actor_optimizer.zero_grad()
    weighted_loss.backward()

    nn.utils.clip_grad_norm_(
        list(self.depth_encoder.parameters()) +
        list(self.depth_actor.parameters()),
        self.max_grad_norm
    )
    self.depth_actor_optimizer.step()

    # =====================================================
    # Return raw losses (not normalized ones!)
    # =====================================================
    return (
        recon_loss_raw.item(),
        action_loss_raw.item(),
        weighted_loss.item()
    )
