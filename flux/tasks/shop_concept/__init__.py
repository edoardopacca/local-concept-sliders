# shop_concept package init. See README.md for the task documentation.
#
# Importing this package installs a compatibility shim on
# `torch.nn.functional.scaled_dot_product_attention` that silently drops the
# `enable_gqa` keyword argument. diffusers >= 0.36 passes this kwarg
# unconditionally to SDPA, but it only exists in torch >= 2.5; on torch 2.4
# the underlying SDPA implementation raises
# `TypeError: unexpected keyword argument 'enable_gqa'`.
#
# Dropping the kwarg is semantically a no-op for Flux: `enable_gqa` enables
# Grouped Query Attention, which requires K/V and Q to have a different
# number of heads (e.g. Llama). Flux uses the same head count for q/k/v on
# every attention block, so GQA is never active regardless. diffusers also
# defaults the kwarg to False, so removing it is equivalent to passing it
# as False.
#
# The shim is installed unconditionally rather than gated on
# `inspect.signature`, because the SDPA implementation on torch 2.4 is a
# C-level builtin whose signature is not reliably introspectable.
def _install_sdpa_enable_gqa_shim() -> None:
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    if getattr(orig, "_shop_concept_sdpa_shim", False):
        return  # shim already installed, avoid double-wrap

    def _sdpa_shim(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return orig(*args, **kwargs)

    _sdpa_shim._shop_concept_sdpa_shim = True
    F.scaled_dot_product_attention = _sdpa_shim


_install_sdpa_enable_gqa_shim()
del _install_sdpa_enable_gqa_shim
