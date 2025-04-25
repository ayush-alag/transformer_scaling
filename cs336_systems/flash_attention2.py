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
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qb, stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        tile_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_ob, stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        tile_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lb, stride_lq),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        tile_shape=(Q_TILE_SIZE,),
        order=(1, 0),
    )

    # now let's load Q, O, L
    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l_tile = tl.load(L_block_ptr, boundary_check=(0, 1), padding_option="zero")

    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    s_ij = tl.zeros((Q_TILE_SIZE, K_TILE_SIZE), dtype=tl.float32)

    kt_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        tile_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_QUERIES, D),
        strides=(stride_vb, stride_vk),
        offsets=(0, 0),
        tile_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # we want to loop over all the k tiles
    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load the K tile
        kt_tile = tl.load(kt_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_TILE_SIZE, D)

        # do some stuff with k, q, v
        tl.dot(q_tile, kt_tile, acc=s_ij) * scale
        new_m_i = tl.max(m_i, s_ij.amax(dim=-1))
        p_tilde = tl.exp(s_ij - new_m_i.unsqueeze(-1))
        exp_mi_diff = tl.exp(m_i - new_m_i)
        l_i = exp_mi_diff * l_tile + tl.sum(p_tilde, dim=-1)
        o_i = tl.dot(p_tilde.to(v_tile.dtype), v_tile, o_i) + exp_mi_diff.unsqueeze(-1) * o_tile

        m_i = new_m_i
        l_tile = l_i

        kt_block_ptr = kt_block_ptr.advance((0, K_TILE_SIZE))
        v_block_ptr = v_block_ptr.advance((0, K_TILE_SIZE))

    o_i = l_i.reciprocal() * o_i
    l_i = tl.log(l_i) + m_i

    # todo: verify the boundary check
    tl.store(O_block_ptr, o_i.to(o_tile.dtype), boundary_check=(0,))
    tl.store(L_block_ptr, l_i.to(l_tile.dtype), boundary_check=(0,))


class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, mask=None):
        b, n_q, d = q.shape
        n_k = k.shape[1]

        assert q.is_cuda and k.is_cuda and v.is_cuda, "Expected CUDA tensors"
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "Our pointer arithmetic will assume contiguous q,k,v"

        # b_q and b_k are chosen to be 16
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 16

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        o = torch.empty(b, n_q, d, device=q.device)
        l = torch.empty(b, n_q, device=q.device)

        # Launch our kernel with n instances in our 1D grid.
        flash_fwd_kernel[(tl.cdiv(n_q, ctx.Q_TILE_SIZE),)](
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
        )

        ctx.save_for_backward(o, l, k, v)

        return o

    @staticmethod
    def backward(ctx, dO, dL, dk, dv):
        return NotImplementedError


class FlashAttention2(torch.autograd.Function):
    # TODO: Handle the batch size dimension
    @staticmethod
    def forward(ctx, q, k, v, mask=None):
        b_q, b_k = 16, 16
        b, n_q, d = q.shape
        b, n_k, d = k.shape
        # save O, L, K, V
        t_q_range = ceil(n_q / b_q)
        t_k_range = ceil(n_k / b_k)

        O = torch.zeros(b, n_q, d)
        L = torch.zeros(b, n_q)

        for i in range(t_q_range):
            # load the tiles
            start_i = i * b_q
            end_i = min(start_i + b_q, n_q)
            start_k = i * b_k
            end_k = min(start_k + b_k, n_k)

            q_i = q[:, start_i:end_i, :]

            # need to store prev_m_i, prev_l_i, prev_o_i as copies
            # m initialized to minus infinity with size q.shape[0]
            prev_m_i = torch.full((b, b_q), -float('inf'))
            prev_l_i = torch.zeros((b, b_q))
            prev_o_i = torch.zeros((b, b_q, d))

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

        ctx.save_for_backward(O, L, k, v)
        return O

    @staticmethod
    def backward(ctx, dO, dL, dk, dv):
        return NotImplementedError