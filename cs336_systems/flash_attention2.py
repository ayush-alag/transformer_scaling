import torch
from math import ceil, sqrt
from einops import einsum
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    kt_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # now let's load Q, O, L
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    original_q_tile_dtype = q_tile.dtype
    # store q as float16
    q_tile = (q_tile * scale).to(original_q_tile_dtype)

    # our temporary variables stored in memory
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # we want to loop over all the k tiles up to the current query position if causal
    max_tiles = (query_tile_index + 1) * Q_TILE_SIZE if is_causal else N_KEYS
    for k_tile_index in range(tl.cdiv(max_tiles, K_TILE_SIZE)):
        # load the K tile
        kt_tile = tl.load(kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (D, K_TILE_SIZE)
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # accumulate matrix multiplication
        s_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)
        s_ij = tl.dot(q_tile, kt_tile, acc=s_ij)

        if is_causal:
            q_range = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_range = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_range[:, None] < k_range[None, :]
            s_ij = tl.where(causal_mask, -1e6, s_ij)

        # update m_i
        new_m_i = tl.maximum(m_i, tl.max(s_ij, axis=-1))
        p_tilde = tl.math.exp(s_ij - new_m_i[:, None])
        exp_mi_diff = tl.math.exp(m_i - new_m_i)
        l_i = l_i * exp_mi_diff + tl.sum(p_tilde, -1)
        o_i = exp_mi_diff[:, None] * o_i
        o_i = tl.dot(p_tilde.to(v_tile.dtype), v_tile, acc=o_i)

        m_i = new_m_i

        kt_block_ptr = kt_block_ptr.advance((0, K_TILE_SIZE))
        v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))

    o_i /= l_i[:, None]
    l_i = tl.log(l_i) + m_i

    # todo: verify the boundary check
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    tl.store(O_block_ptr, o_i, boundary_check=(0,1))
    tl.store(L_block_ptr, l_i, boundary_check=(0,))


class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None):
        b, n_q, d = q.shape
        n_k = k.shape[1]

        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Our pointer arithmetic will assume contiguous q,k,v"

        ctx.is_causal = mask

        # b_q and b_k are chosen to be 16
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        o = torch.empty(b, n_q, d, device=q.device)
        l = torch.empty(b, n_q, device=q.device)

        # Launch our kernel with an instance per batch element per tile
        flash_fwd_kernel[(triton.cdiv(n_q, ctx.Q_TILE_SIZE), b)](
            q, k, v,
            o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            N_QUERIES=n_q, N_KEYS=n_k,
            D=d, scale=1 / sqrt(d),
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=ctx.is_causal,
        )

        ctx.save_for_backward(o, l, q, k, v)

        return o

    @staticmethod
    def backward(ctx, dO):
        return FlashAttention2.backward(ctx, dO)


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None):
        b_q, b_k = 16, 16
        b, n_q, d = q.shape
        b, n_k, d = k.shape
        # save O, L, K, V
        t_q_range = ceil(n_q / b_q)
        t_k_range = ceil(n_k / b_k)

        O = torch.zeros(b, n_q, d, device=q.device, dtype=q.dtype)
        L = torch.zeros(b, n_q, device=q.device, dtype=q.dtype)

        ctx.is_causal = mask

        for i in range(t_q_range):
            # load the tiles
            start_i = i * b_q
            end_i = min(start_i + b_q, n_q)
            start_k = i * b_k
            end_k = min(start_k + b_k, n_k)

            q_i = q[:, start_i:end_i, :]

            # need to store prev_m_i, prev_l_i, prev_o_i as copies
            # m initialized to minus infinity with size q.shape[0]
            prev_m_i = torch.full((b, b_q), -float('inf'), dtype=q.dtype, device=q.device)
            prev_l_i = torch.zeros((b, b_q), dtype=q.dtype, device=q.device)
            prev_o_i = torch.zeros((b, b_q, d), dtype=q.dtype, device=q.device)

            for j in range(t_k_range):
                start_k = j * b_k
                end_k = min(start_k + b_k, n_k)

                k_j = k[:, start_k:end_k, :]
                v_j = v[:, start_k:end_k, :]

                # tile of pre-softmax attention scores
                s_ij = einsum(q_i, k_j, '... q d, ... k d -> ... q k') / sqrt(d)

                # update m_i
                m_i = torch.max(prev_m_i, s_ij.amax(dim=-1))
                p_tilde = torch.exp(s_ij - m_i.unsqueeze(-1))
                exp_mi_diff = torch.exp(prev_m_i - m_i)
                l_i = exp_mi_diff * prev_l_i + torch.sum(p_tilde, dim=-1)
                o_i = einsum(p_tilde, v_j, '... q k, ... k d -> ... q d') + exp_mi_diff.unsqueeze(-1) * prev_o_i

                # update prev_m_i, prev_l_i, prev_o_i
                prev_m_i = m_i
                prev_l_i = l_i
                prev_o_i = o_i

            # update O, L, m
            O[:, start_i:end_i, :] = einsum(l_i.reciprocal(), o_i, '... k, ... k d -> ... k d')
            L[:, start_i:end_i] = torch.log(l_i) + m_i

        ctx.save_for_backward(O, L, q, k, v)
        return O

    @staticmethod
    @torch.compile(fullgraph=True)
    def backward(ctx, dO):
        O, L, Q, K, V = ctx.saved_tensors
        d = K.shape[-1]
        # return dQ, dK, dV
        S = einsum(Q, K, '... q d, ... k d -> ... q k') / sqrt(d)
        if ctx.is_causal:
            mask = torch.triu(torch.ones(S.shape[-2], S.shape[-1], device=S.device, dtype=torch.bool), diagonal=1)
            S = S.masked_fill(mask, float('-inf'))

        P = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P, dO, '... q k, ... q d -> ... k d')
        dP = einsum(dO, V, '... q d, ... k d -> ... q k')

        D = (O * dO).sum(dim=-1)
        dS = P * (dP - D.unsqueeze(-1))

        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') / sqrt(d)
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') / sqrt(d)
        return dQ, dK, dV, None
