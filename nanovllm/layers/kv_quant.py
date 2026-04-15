import torch  

import triton
import triton.language as tl

@triton.jit
def store_kvcache_int8_kernel(
    key_ptr, 
    key_stride,
    value_ptr, 
    value_stride,
    k_cache_ptr,           # INT8  [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache_ptr,           # INT8  [num_blocks, block_size, num_kv_heads, head_dim]
    k_scale_ptr,           # FP32  [num_blocks, block_size, num_kv_heads]
    v_scale_ptr,           # FP32  [num_blocks, block_size, num_kv_heads]
    slot_mapping_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
):
    idx = tl.program_id(0)
    token_idx = idx // num_heads
    head_idx = idx % num_heads
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot == -1: return

    key_offsets = token_idx * key_stride + head_idx * head_dim + tl.arange(0, head_dim)
    value_offsets = token_idx * value_stride + head_idx * head_dim + tl.arange(0, head_dim)

    key_fp = tl.load(key_ptr + key_offsets).to(tl.float32)
    value_fp = tl.load(value_ptr + value_offsets).to(tl.float32)

    k_scale = tl.max(tl.abs(key_fp))   / 127.0 + 1e-8
    v_scale = tl.max(tl.abs(value_fp)) / 127.0 + 1e-8

    scaled_key = key_fp / k_scale
    scaled_value = value_fp / v_scale

    key_int8 = tl.where(scaled_key >= 0, scaled_key + 0.5, scaled_key - 0.5).to(tl.int8)
    value_int8 = tl.where(scaled_value >= 0, scaled_value + 0.5, scaled_value - 0.5).to(tl.int8)

    cache_offsets = slot * num_heads * head_dim + head_idx * head_dim + tl.arange(0, head_dim)
    tl.store(k_cache_ptr + cache_offsets, key_int8) 
    tl.store(v_cache_ptr + cache_offsets, value_int8)

    scale_offsets = slot * num_heads + head_idx
    tl.store(k_scale_ptr + scale_offsets, k_scale)
    tl.store(v_scale_ptr + scale_offsets, v_scale)


def store_kvcache_int8(
        key: torch.Tensor,      # [N, num_heads, head_dim]
        value: torch.Tensor,    # [N, num_heads, head_dim]
        k_cache: torch.Tensor,  # INT8 [num_blocks, block_size, num_heads, head_dim]
        v_cache: torch.Tensor,  # INT8 [num_blocks, block_size, num_heads, head_dim]
        k_scale: torch.Tensor,  # FP32 [num_blocks, block_size, num_heads]
        v_scale: torch.Tensor,  # FP32 [num_blocks, block_size, num_heads]
        slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    
    assert k_cache.is_contiguous(), "k_cache must be strictly contiguous"
    assert v_cache.is_contiguous(), "v_cache must be strictly contiguous"
    assert k_scale.is_contiguous(), "k_scale must be strictly contiguous"
    assert v_scale.is_contiguous(), "v_scale must be strictly contiguous"

    assert slot_mapping.numel() == N
    assert triton.next_power_of_2(head_dim) == head_dim, "head_dim must be a power of 2"
    store_kvcache_int8_kernel[(N * num_heads,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, k_scale, v_scale, slot_mapping, num_heads=num_heads, head_dim=head_dim)


@triton.jit
def dequant_kvcache_kernel(
    cache_ptr,          # INT8  [num_blocks, block_size, num_kv_heads, head_dim]
    scale_ptr,          # FP32  [num_blocks, block_size, num_kv_heads]
    out_ptr,            # FP16  [num_blocks, block_size, num_kv_heads, head_dim]
    head_dim: tl.constexpr,
):
    idx = tl.program_id(0)

    cache_offsets = idx * head_dim + tl.arange(0, head_dim)
    scale_offset = idx

    cache_int8 = tl.load(cache_ptr + cache_offsets).to(tl.float32)
    scale = tl.load(scale_ptr + scale_offset)

    cache_out = (cache_int8 * scale).to(out_ptr.dtype.element_ty)

    out_offsets = idx * head_dim + tl.arange(0, head_dim)
    tl.store(out_ptr + out_offsets, cache_out)

def dequant_kvcache(
    cache: torch.Tensor,   # INT8  [num_blocks, block_size, num_kv_heads, head_dim]
    scale: torch.Tensor,   # FP32  [num_blocks, block_size, num_kv_heads]
    num_heads: int,
    head_dim: int,
    target_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    num_blocks, block_size = cache.shape[0], cache.shape[1]
    assert cache.is_contiguous(), "cache must be strictly contiguous"
    assert scale.is_contiguous(), "scale must be strictly contiguous"
    out = torch.empty(num_blocks, block_size, num_heads, head_dim, dtype=target_dtype, device=cache.device)
    dequant_kvcache_kernel[(num_blocks * block_size * num_heads,)](cache, scale, out, head_dim=head_dim)
    return out.view(num_blocks, block_size, num_heads, head_dim)


