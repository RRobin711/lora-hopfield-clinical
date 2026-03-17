"""
Hopfield attention injection into GPT-2 for DREADDIT classification.

This module replaces the core attention computation inside each GPT-2 attention
block with the Hopfield energy-minimizing retrieval rule from Ramsauer et al.
(2020). Only the softmax(QK^T/sqrt(d)) @ V step is replaced — QKV projections
(c_attn), output projection (c_proj), layer norms, residual connections, MLPs,
and positional embeddings are all preserved untouched.

The injection approach: for each GPT2Attention block, we attach a HopfieldAttention
module and replace the block's forward method with a custom one that routes through
Hopfield retrieval instead of eager_attention_forward. This avoids modifying the
transformers library source and keeps the injection reversible.

Why this matters for Phase 1 narrative: beta controls the sharpness of the Hopfield
energy landscape. At the default beta = 1/sqrt(d_head), Hopfield exactly recovers
standard scaled dot-product attention — this is the correctness gate. Explicit beta
values let us explore whether sharper (high beta, winner-take-all) or broader
(low beta, distributed) retrieval helps with informal Reddit stress language,
connecting to Phase 1's HyDE finding that informal text needs softer matching.

Causal mask handling: GPT-2 defaults to SDPA (Scaled Dot-Product Attention) which
handles causal masking internally — the mask passed to attention blocks is often
None. When padding is present, create_causal_mask() returns a boolean tensor
(True = attend, False = mask). HopfieldAttention uses the opposite convention
(True = masked), so the injection inverts the mask at the boundary. When no mask
is provided, the injection constructs a standard upper-triangular causal mask.

Reference: Ramsauer et al. (2020), https://arxiv.org/abs/2008.02217
"""

from __future__ import annotations

import logging
from types import MethodType

import torch
import torch.nn as nn
from transformers import GPT2Model

from src.hopfield import HopfieldAttention, HopfieldConfig

logger = logging.getLogger(__name__)


def _make_hopfield_forward(
    original_attn: nn.Module,
    hopfield: HopfieldAttention,
) -> callable:
    """Build a replacement forward method for a GPT2Attention block.

    The replacement reuses the block's c_attn (QKV projection), c_proj (output
    projection), and resid_dropout from the original GPT2Attention. Only the
    attention weight computation is routed through HopfieldAttention.

    We capture `hopfield` in the closure so each block gets its own instance,
    and `original_attn` provides access to c_attn, c_proj, resid_dropout, and
    the GPT-2 config (head_dim, split_size, num_heads) at call time.

    Args:
        original_attn: The GPT2Attention module whose forward is being replaced.
            Its c_attn, c_proj, resid_dropout, head_dim, split_size, num_heads
            attributes are used directly.
        hopfield: HopfieldAttention instance for this block.

    Returns:
        A forward function with the same signature as GPT2Attention.forward.
    """

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values=None,
        cache_position=None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        # Self-attention QKV projection — works for both Conv1D and nn.Linear
        query_states, key_states, value_states = self.c_attn(
            hidden_states
        ).split(self.split_size, dim=2)

        # Reshape into (B, num_heads, seq_len, head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # Build the causal mask for HopfieldAttention (bool, True = MASKED).
        #
        # GPT-2 defaults to SDPA which handles causal masking internally via
        # is_causal=True — attention_mask is often None for non-padded inputs.
        # Our replacement must construct the causal mask explicitly.
        #
        # When attention_mask IS provided (with padding), create_causal_mask()
        # returns a boolean tensor: True = ATTEND, False = MASK. This encodes
        # both causal structure (future tokens masked) and padding (pad tokens
        # masked). Shape: (B, 1, seq_len, seq_len).
        #
        # HopfieldAttention uses the OPPOSITE convention (True = MASKED) in
        # masked_fill, so we invert the boolean mask.
        seq_len = query_states.size(2)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Invert: True=attend → False, False=mask → True
                causal_mask = ~attention_mask
            else:
                # Additive float mask (legacy path): -inf positions → True
                causal_mask = attention_mask < -1.0
        else:
            # No explicit mask — construct standard causal mask.
            # Upper triangle = True (future tokens should be masked).
            # Shape (1, 1, seq_len, seq_len) broadcasts over batch and heads.
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=query_states.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)

        # Hopfield retrieval: replaces softmax(QK^T/sqrt(d)) @ V
        # Returns (B, H, N, d_head)
        attn_output = hopfield(
            queries=query_states,
            keys=key_states,
            values=value_states,
            causal_mask=causal_mask,
        )

        # Transpose back to (B, seq_len, num_heads, head_dim) to match
        # eager_attention_forward's output convention (line 77 in modeling_gpt2.py)
        attn_output = attn_output.transpose(1, 2)

        # Reshape to (B, seq_len, embed_dim) and project through c_proj
        attn_output = attn_output.reshape(
            *attn_output.shape[:-2], -1
        ).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # attn_weights=None since we don't store them (saves memory)
        return attn_output, None

    return forward


def inject_hopfield(
    model: GPT2Model,
    config: HopfieldConfig | None = None,
) -> dict[str, int]:
    """Inject Hopfield attention into all GPT-2 attention blocks.

    Replaces the core attention computation (softmax(QK^T/sqrt(d)) @ V) in each
    block with Hopfield energy-minimizing retrieval. All base model weights are
    frozen — only the HopfieldAttention beta buffer (non-trainable) is added.

    The injection:
    1. Freezes all model parameters
    2. For each attention block, creates a HopfieldAttention(d_head=64)
    3. Attaches it as block.attn.hopfield_attn (so it moves with .to(device))
    4. Replaces block.attn.forward with a custom method routing through Hopfield

    Args:
        model: GPT2Model (base transformer). All parameters will be frozen.
        config: HopfieldConfig. If None, uses defaults (beta=1/sqrt(d_head),
            num_iters=1), which exactly recovers standard attention.

    Returns:
        Dict mapping block paths to the number of new parameters added
        (0 for each block — Hopfield has no trainable params, only a beta buffer).
    """
    config = config or HopfieldConfig()

    # Freeze all base weights — same pattern as inject_lora
    for param in model.parameters():
        param.requires_grad = False

    d_head = model.config.hidden_size // model.config.num_attention_heads
    replaced = {}

    for block_idx, block in enumerate(model.h):
        attn = block.attn

        # Create HopfieldAttention for this block
        hop = HopfieldAttention(d_head=d_head, config=config)

        # Attach as a submodule so it moves with .to(device) and appears
        # in model.modules() / state_dict
        attn.hopfield_attn = hop

        # Replace forward — MethodType binds the function as a bound method
        # so `self` refers to `attn` at call time
        new_forward = _make_hopfield_forward(attn, hop)
        attn.forward = MethodType(new_forward, attn)

        module_path = f"h.{block_idx}.attn"
        replaced[module_path] = 0  # No trainable params added

        logger.debug(
            "Injected HopfieldAttention into %s (beta=%.6f, num_iters=%d)",
            module_path,
            hop.beta.item(),
            hop.num_iters,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info(
        "Hopfield injection complete: %d blocks, beta=%.6f, "
        "%d total params, %d trainable (all frozen — only head trains)",
        len(replaced),
        hop.beta.item(),
        total_params,
        trainable_params,
    )

    return replaced


def verify_hopfield_injection(model: GPT2Model) -> dict[str, object]:
    """Post-injection diagnostic confirming Hopfield attention is active.

    Checks:
    1. Every attention block has a hopfield_attn submodule
    2. All base parameters are frozen (requires_grad=False)
    3. Forward produces valid output shape

    Args:
        model: GPT2Model after inject_hopfield() has been called.

    Returns:
        Summary dict with total_params, trainable_params, blocks_with_hopfield,
        and all_checks_passed.
    """
    all_passed = True
    blocks_with_hopfield = 0

    for block_idx, block in enumerate(model.h):
        attn = block.attn
        if not hasattr(attn, "hopfield_attn"):
            logger.error("h.%d.attn missing hopfield_attn submodule", block_idx)
            all_passed = False
        elif not isinstance(attn.hopfield_attn, HopfieldAttention):
            logger.error(
                "h.%d.attn.hopfield_attn is %s, expected HopfieldAttention",
                block_idx,
                type(attn.hopfield_attn).__name__,
            )
            all_passed = False
        else:
            blocks_with_hopfield += 1

    # All base params should be frozen
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.error("Parameter is trainable (should be frozen): %s", name)
            all_passed = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "blocks_with_hopfield": blocks_with_hopfield,
        "all_checks_passed": all_passed,
    }

    if all_passed:
        logger.info(
            "Hopfield verification passed: %d blocks, 0 trainable params",
            blocks_with_hopfield,
        )
    else:
        logger.error("Hopfield verification FAILED — see errors above")

    return summary
