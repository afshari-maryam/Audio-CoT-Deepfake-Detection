"""COLMBO-DF: Feature-Guided Audio Language Model for Deepfake Detection.

Architecture (Section 2.1):
  - Frozen WavLM-base-plus audio encoder  E
  - 6-layer QFormer projector              P  (trainable)
  - Llama 3.2-1B-Instruct LLM             (frozen by default)

Two audio inputs (reference + unknown) are each independently encoded,
projected to m query tokens, and prepended to the text prompt that contains
serialised acoustic evidence.  The LLM then generates a chain-of-thought
followed by a structured final prediction.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WavLMModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

AUDIO_PLACEHOLDER = "<|audio|>"


class QFormerProjector(nn.Module):
    """6-layer QFormer projector (Section 2.1 / Figure 1).

    Learnable query tokens attend (via cross-attention) to the frame-level
    WavLM features and produce m tokens in the LLM embedding space.
    Implemented as a standard TransformerDecoder:
        tgt  = learnable queries (B, m, d_llm)
        mem  = projected audio frames (B, T', d_llm)
    giving: out = TransformerDecoder(tgt, mem)  shape (B, m, d_llm)
    """

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        num_query_tokens: int = 32,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.empty(1, num_query_tokens, llm_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Linear bridge from encoder dim to LLM dim
        self.audio_proj = nn.Linear(audio_dim, llm_dim)

        # Pre-norm TransformerDecoder for stable training
        layer = nn.TransformerDecoderLayer(
            d_model=llm_dim,
            nhead=nhead,
            dim_feedforward=llm_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(
        self,
        audio_hidden: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_hidden:      (B, T', audio_dim) encoder frame-level outputs
            key_padding_mask:  (B, T') True for padded positions in audio

        Returns:
            (B, num_query_tokens, llm_dim)
        """
        B = audio_hidden.shape[0]
        memory = self.audio_proj(audio_hidden)             # (B, T', llm_dim)
        queries = self.query_tokens.expand(B, -1, -1)      # (B, Q, llm_dim)
        out = self.transformer(
            queries, memory, memory_key_padding_mask=key_padding_mask
        )
        return self.out_norm(out)                          # (B, Q, llm_dim)


class ColmboDF(nn.Module):
    """End-to-end COLMBO-DF model.

    Call `forward` for training (returns CausalLMOutputWithPast with loss).
    Call `generate` for inference (returns generated token ids).

    Input text must contain exactly two `<|audio|>` placeholder tokens
    (one for the reference audio and one for the unknown audio).  The
    forward / generate methods replace them with the projected audio embeddings.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/wavlm-base-plus",
        llm_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        num_query_tokens: int = 32,
        qformer_layers: int = 6,
        qformer_heads: int = 8,
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
    ):
        super().__init__()

        # ── Audio encoder ────────────────────────────────────────────────────
        self.encoder = WavLMModel.from_pretrained(encoder_name)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

        audio_dim = self.encoder.config.hidden_size  # 768 for wavlm-base-plus

        # ── Tokenizer & LLM ──────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [AUDIO_PLACEHOLDER]}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.bfloat16
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        if freeze_llm:
            self.llm.requires_grad_(False)

        self.audio_token_id: int = self.tokenizer.convert_tokens_to_ids(
            AUDIO_PLACEHOLDER
        )

        # ── Projector (always trainable) ─────────────────────────────────────
        llm_dim = self.llm.config.hidden_size
        self.projector = QFormerProjector(
            audio_dim=audio_dim,
            llm_dim=llm_dim,
            num_query_tokens=num_query_tokens,
            num_layers=qformer_layers,
            nhead=qformer_heads,
        )
        self.num_query_tokens = num_query_tokens

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_audio(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run WavLM and build a key-padding mask for the QFormer cross-attn."""
        with torch.no_grad():
            out = self.encoder(waveform, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T', d_enc)

        if attention_mask is not None:
            # Downsample: repeat-nearest to match encoder output length
            T_prime = hidden.shape[1]
            # True where padding (QFormer expects True = ignore)
            kpm = ~attention_mask[:, :T_prime].bool()
        else:
            kpm = None
        return hidden, kpm

    def _project_audio(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden, kpm = self._encode_audio(waveform, attention_mask)
        return self.projector(hidden, key_padding_mask=kpm)  # (B, Q, d_llm)

    def _splice_audio_into_sequence(
        self,
        input_ids: torch.Tensor,
        audio_tokens1: torch.Tensor,
        audio_tokens2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Replace the two <|audio|> placeholders with real audio embeddings.

        Returns:
            inputs_embeds : (B, L', d_llm)   L' = L + 2*(Q-1)
            attention_mask: (B, L')
            expanded_labels: (B, L') or None  — audio positions set to -100
        """
        B = input_ids.shape[0]
        Q = self.num_query_tokens
        embed_fn = self.llm.get_input_embeddings()

        embeds_out, attn_out, label_out = [], [], []
        for b in range(B):
            ids = input_ids[b]
            pos = (ids == self.audio_token_id).nonzero(as_tuple=True)[0]
            assert len(pos) == 2, (
                f"Expected exactly 2 <|audio|> tokens, found {len(pos)}"
            )
            p1, p2 = int(pos[0]), int(pos[1])

            seg = [
                ids[:p1],
                None,           # audio1 placeholder
                ids[p1 + 1 : p2],
                None,           # audio2 placeholder
                ids[p2 + 1 :],
            ]

            # ---- inputs_embeds ----
            e_seg = [
                embed_fn(seg[0]),
                audio_tokens1[b],
                embed_fn(seg[2]),
                audio_tokens2[b],
                embed_fn(seg[4]),
            ]
            embeds_out.append(torch.cat(e_seg, dim=0))

            # ---- attention_mask (1 everywhere except padding at the end) ----
            pad_id = self.tokenizer.pad_token_id
            a_seg = [
                torch.ones(len(seg[0]), device=ids.device, dtype=torch.long),
                torch.ones(Q,           device=ids.device, dtype=torch.long),
                torch.ones(len(seg[2]), device=ids.device, dtype=torch.long),
                torch.ones(Q,           device=ids.device, dtype=torch.long),
                (seg[4] != pad_id).long(),
            ]
            attn_out.append(torch.cat(a_seg, dim=0))

            # ---- labels ----
            if labels is not None:
                lbl = labels[b]
                l_seg = [
                    lbl[:p1],
                    torch.full((Q,), -100, device=ids.device, dtype=torch.long),
                    lbl[p1 + 1 : p2],
                    torch.full((Q,), -100, device=ids.device, dtype=torch.long),
                    lbl[p2 + 1 :],
                ]
                label_out.append(torch.cat(l_seg, dim=0))

        return (
            torch.stack(embeds_out, dim=0),
            torch.stack(attn_out, dim=0),
            torch.stack(label_out, dim=0) if labels is not None else None,
        )

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        waveform1: torch.Tensor,
        waveform2: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        wav_attention_mask1: Optional[torch.Tensor] = None,
        wav_attention_mask2: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        audio_tokens1 = self._project_audio(waveform1, wav_attention_mask1)
        audio_tokens2 = self._project_audio(waveform2, wav_attention_mask2)

        inputs_embeds, attention_mask, expanded_labels = (
            self._splice_audio_into_sequence(
                input_ids, audio_tokens1, audio_tokens2, labels
            )
        )

        return self.llm(
            inputs_embeds=inputs_embeds.to(self.llm.dtype),
            attention_mask=attention_mask,
            labels=expanded_labels,
        )

    # ── Generate (inference) ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        waveform1: torch.Tensor,
        waveform2: torch.Tensor,
        input_ids: torch.Tensor,
        wav_attention_mask1: Optional[torch.Tensor] = None,
        wav_attention_mask2: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **gen_kwargs,
    ) -> torch.Tensor:
        audio_tokens1 = self._project_audio(waveform1, wav_attention_mask1)
        audio_tokens2 = self._project_audio(waveform2, wav_attention_mask2)

        inputs_embeds, attention_mask, _ = self._splice_audio_into_sequence(
            input_ids, audio_tokens1, audio_tokens2, labels=None
        )

        return self.llm.generate(
            inputs_embeds=inputs_embeds.to(self.llm.dtype),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
