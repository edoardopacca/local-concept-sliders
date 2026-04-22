# shop_concept: LoRAShop pipeline adattata ai Concept Sliders (Flux)
#
# Contiene tutto il necessario per generare immagini con Flux.1-dev
# applicando uno o piu' Concept Sliders alla maniera di LoRAShop
# (mask estratte dall'attenzione del block 19, blending per-token).
#
# Modifiche rispetto a LoRAShop-main documentate in CHANGES.md.

# ---------------------------------------------------------------------------
# Compat shim: diffusers>=0.36 chiama `F.scaled_dot_product_attention` con
# `enable_gqa=...`. Quel kwarg esiste solo in torch>=2.5; su torch 2.4 la
# SDPA e' una funzione C che alza `TypeError: unexpected keyword argument
# 'enable_gqa'`. Il venv `flux-sliders` su HPC Bocconi ha torch 2.4.1+cu124.
#
# Installiamo la shim in maniera INCONDIZIONATA: droppiamo `enable_gqa` dagli
# kwargs prima di delegare alla SDPA reale.
#
# Perche' e' safe anche su torch>=2.5 o quando e' chiamata con enable_gqa=True:
# - `enable_gqa` attiva Grouped Query Attention, usata quando il numero di
#   head di K/V e' diverso da quello di Q (es. Llama). Flux ha stesso head
#   count per q/k/v in tutti i blocks (double e single), quindi GQA non e'
#   mai attivabile semanticamente. Il kwarg viene passato per default=False
#   da diffusers, droppare e' identico a passarlo a False.
#
# Non usiamo `inspect.signature` per decidere se installare: in torch 2.4 la
# SDPA e' un builtin e la signature introspettabile e' inaffidabile.
# ---------------------------------------------------------------------------
def _install_sdpa_enable_gqa_shim() -> None:
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    if getattr(orig, "_shop_concept_sdpa_shim", False):
        return  # shim gia' installata, evita wrap doppio

    def _sdpa_shim(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return orig(*args, **kwargs)

    _sdpa_shim._shop_concept_sdpa_shim = True
    F.scaled_dot_product_attention = _sdpa_shim


_install_sdpa_enable_gqa_shim()
del _install_sdpa_enable_gqa_shim
