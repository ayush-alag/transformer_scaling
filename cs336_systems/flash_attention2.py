class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, b_q, b_k, mask=None):
        # save O, L, K, V
        t_q_range = ceil(q.shape[0] / b_q)
        t_k_range = ceil(k.shape[0] / b_k)

        O = torch.zeros(q.shape[0], q.shape[1])
        L = torch.zeros(q.shape[0])

        for i in range(t_q_range):
            # load the tiles
            start_i = i * b_q
            end_i = min(start_i + b_q, q.shape[0])
            start_k = i * b_k
            end_k = min(start_k + b_k, k.shape[0])

            q_i = q[start_i:end_i]

            # need to store prev_m_i, prev_l_i, prev_o_i as copies
            # m initialized to minus infinity with size q.shape[0]
            prev_m_i = torch.full((b_q,), -float('inf'))
            prev_l_i = torch.zeros(b_q)
            prev_o_i = torch.zeros(b_q, q.shape[1])

            for j in range(t_k_range):
                start_k = j * b_k
                end_k = min(start_k + b_k, k.shape[0])

                k_j = k[start_k:end_k]
                v_j = v[start_k:end_k]

                # tile of pre-softmax attention scores
                s_ij = einsum('... b_q d, .. b_k d -> b_q b_k', q_i, k_j) / sqrt(q_i.shape[1])

                # update m_i
                m_i = torch.max(prev_m_i, torch.max(s_ij, dim=1)[0])
                p_tilde = torch.exp(s_ij - m_i)
                exp_mi_diff = torch.exp(prev_m_i - m_i)
                l_i = exp_mi_diff * prev_l_i + torch.sum(p_tilde, dim=1)
                o_i = torch.einsum('... b_q b_k, ... b_k d -> ... b_q d', p_tilde, v_j) + exp_mi_diff.unsqueeze(-1) * prev_o_i

                # update prev_m_i, prev_l_i, prev_o_i
                prev_m_i = m_i
                prev_l_i = l_i
                prev_o_i = o_i

            # update O, L, m
            O[start_i:end_i] = torch.einsum('...k,...kd->...kd', l_i.reciprocal(), o_i)
            L[start_i:end_i] = torch.log(l_i) + m_i

        ctx.save_for_backward(O, L, K, V)
        return O, L

    @staticmethod
    def backward(ctx, dO, dL, dK, dV):
        return NotImplementedError