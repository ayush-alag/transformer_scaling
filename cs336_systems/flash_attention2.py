import torch
from math import ceil, sqrt
from einops import einsum

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