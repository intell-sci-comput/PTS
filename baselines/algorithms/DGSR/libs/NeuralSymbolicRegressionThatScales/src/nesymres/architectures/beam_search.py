import torch


def generate_beam(env, dec, decoder_args, beam_size, length_penalty, early_stopping, max_len=100):
    """
    Decode a sentence given initial start.
    `x`:
        - LongTensor(bs, slen)
            <EOS> W1 W2 W3 <EOS> <PAD>
            <EOS> W1 W2 W3   W4  <EOS>
    `lengths`:
        - LongTensor(bs) [5, 6]
    `positions`:
        - False, for regular "arange" positions (LM)
        - True, to reset positions from the new generation (MT)
    """

    # check inputs
    trg, enc_src, trg_mask, src_mask = decoder_args
    src_enc = enc_src
    src_len = enc_src

    #assert src_enc.size(0) == src_len.size(0)
    assert beam_size >= 1

    # batch size / number of words
    bs = len(src_enc)
    n_words = env.n_words
    breakpoint()

    # expand to beam size the source latent representations / source lengths
    src_enc = src_enc.unsqueeze(1).expand(
        (bs, beam_size) + src_enc.shape[1:]).contiguous().view((bs * beam_size,) + src_enc.shape[1:])
    #src_len = src_len.unsqueeze(1).expand(bs, beam_size).contiguous().view(-1)

    # generated sentences (batch with beam current hypotheses)
    # generated = src_len.new(max_len, bs * beam_size)  # upcoming output
    # generated.fill_(env.pad_index)                   # fill upcoming ouput with <PAD>
    # generated[0].fill_(env.eos_index)                # we use <EOS> for <BOS> everywhere

    # generated hypotheses
    generated_hyps = [BeamHypotheses(
        beam_size, max_len, length_penalty, early_stopping) for _ in range(bs)]

    # positions
    positions = src_len.new(max_len).long()
    #positions = torch.arange(max_len, out=positions).unsqueeze(1).expand_as(generated)

    # scores for each sentence in the beam
    beam_scores = src_enc.new(bs, beam_size).fill_(0)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    # current position
    cur_len = 1

    # cache compute states
    cache = {'slen': 0}

    # done sentences
    done = [False for _ in range(bs)]
    breakpoint()
    while cur_len < max_len:
        dec(trg[:, :-1], enc_src, trg_mask, src_mask)
        # compute word scores
        tensor = decoder(
            x=generated[:cur_len],
            lengths=src_len.new(bs * beam_size).fill_(cur_len),
            positions=positions[:cur_len],
            causal=True,
            src_enc=src_enc,
            src_len=src_len,
            cache=cache
        )
        assert tensor.size() == (1, bs * beam_size, env.dim)
        tensor = tensor.data[-1, :, :]          # (bs * beam_size, dim)
        scores = env.proj(tensor)              # (bs * beam_size, n_words)
        scores = F.log_softmax(scores, dim=-1)  # (bs * beam_size, n_words)
        assert scores.size() == (bs * beam_size, n_words)

        # select next words with scores
        # (bs * beam_size, n_words)
        _scores = scores + beam_scores[:, None].expand_as(scores)
        # (bs, beam_size * n_words)
        _scores = _scores.view(bs, beam_size * n_words)

        next_scores, next_words = torch.topk(
            _scores, 2 * beam_size, dim=1, largest=True, sorted=True)
        assert next_scores.size() == next_words.size() == (bs, 2 * beam_size)

        # next batch beam content
        # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for sent_id in range(bs):

            # if we are done with this sentence
            done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                next_scores[sent_id].max().item())
            if done[sent_id]:
                next_batch_beam.extend(
                    [(0, env.pad_index, 0)] * beam_size)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, value in zip(next_words[sent_id], next_scores[sent_id]):

                # get beam and word IDs
                beam_id = idx // n_words
                word_id = idx % n_words

                # end of sentence, or next word
                if word_id == env.eos_index or cur_len + 1 == max_len:
                    generated_hyps[sent_id].add(
                        generated[:cur_len, sent_id * beam_size + beam_id].clone().cpu(), value.item())
                else:
                    next_sent_beam.append(
                        (value, word_id, sent_id * beam_size + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == beam_size:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + \
                1 == max_len else beam_size
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, env.pad_index, 0)] * \
                    beam_size  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == beam_size * (sent_id + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == bs * beam_size
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = generated.new([x[1] for x in next_batch_beam])
        beam_idx = src_len.new([x[2] for x in next_batch_beam])

        # re-order batch and internal states
        generated = generated[:, beam_idx]
        generated[cur_len] = beam_words
        for k in cache.keys():
            if k != 'slen':
                cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    # def get_coeffs(s):
    #     roots = [int(s[i + 2]) for i, c in enumerate(s) if c == 'x']
    #     poly = np.poly1d(roots, r=True)
    #     coeffs = list(poly.coefficients.astype(np.int64))
    #     return [c % 10 for c in coeffs], coeffs

    # visualize hypotheses
    # print([len(x) for x in generated_hyps], cur_len)
    # globals().update( locals() );
    # !import code; code.interact(local=vars())
    # for ii in range(bs):
    #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
    #         hh = " ".join(self.id2word[x] for x in ww.tolist())
    #         print(f"{ss:+.4f} {hh}")
    #         # cc = get_coeffs(hh[4:])
    #         # print(f"{ss:+.4f} {hh} || {cc[0]} || {cc[1]}")
    #     print("")

    # select the best hypotheses
    tgt_len = src_len.new(bs)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = src_len.new(tgt_len.max().item(), bs).fill_(env.pad_index)
    for i, hypo in enumerate(best):
        decoded[:tgt_len[i] - 1, i] = hypo
        decoded[tgt_len[i] - 1, i] = env.eos_index

    # sanity check
    assert (decoded == env.eos_index).sum() == 2 * bs

    return decoded, tgt_len, generated_hyps


class BeamHypotheses(object):

    # https://code.ihub.org.cn/projects/763/repository/commit_diff?changeset=7e3070ae4f0a1f57a974de9b2b54fd7ad3f08fce
    # https://github.com/huggingface/transformers
    # https://github.com/huggingface/transformers/blob/4975002df50c472cbb6f8ac3580e475f570606ab/src/transformers/generation_beam_search.py#L808

    def __init__(self, n_hyp, max_len, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx)
                                       for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_len ** self.length_penalty
