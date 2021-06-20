from tokenizers import models
from typing import List, Tuple


def fuse_subwords(tokens_enum: List[Tuple], atts: List, tokenizer, strategy=None) -> (List, List):
    assert strategy in ['average', 'salient']
    tokenizer_model = tokenizer.backend_tokenizer.model

    fuse_index, fuse_token, fuse_att = [], '', []
    cleaned_tokens, cleaned_atts = [], []

    def append_cleaned(fuse_i, fuse_t, fuse_a, replace=None):
        """ Append to results, replace tokenizer artifacts if necessary """
        if len(fuse_i) == 1:
            fuse_i = fuse_i[0]
        if replace:
            fuse_t = fuse_t.replace(*replace)
        cleaned_fuse_token = fuse_t
        cleaned_tokens.append((fuse_i, cleaned_fuse_token))
        cleaned_atts.append(apply_fuse_strategy(fuse_a, strategy))

    if type(tokenizer_model) == models.Unigram:  # ALBERT, XLNet
        for i, (token_enum, a) in enumerate(zip(tokens_enum, atts)):
            tidx, token = token_enum
            if len(fuse_att) > 0:
                if token.startswith('▁') and token != '▁':
                    append_cleaned(fuse_index, fuse_token, fuse_att, replace=['▁', ''])
                    fuse_index, fuse_token, fuse_att = [], '', []  # Reset
            fuse_index.append(tidx)
            fuse_token += token
            fuse_att.append(a)
        if fuse_att:
            append_cleaned(fuse_index, fuse_token, fuse_att, replace=['▁', ''])

    elif type(tokenizer_model) == models.WordPiece:  # BERT, ELECTRA
        for i, (token_enum, a) in enumerate(zip(tokens_enum, atts)):
            tidx, token = token_enum
            if token.startswith('##'):
                fuse_index.append(tidx)
                # Append all subsequent '##' subword tokens
                fuse_token += token.replace('##', '')
                fuse_att.append(a)
                if i < len(tokens_enum) - 1:
                    if not tokens_enum[i + 1][1].startswith('##'):
                        append_cleaned(fuse_index, fuse_token, fuse_att)
                        fuse_index, fuse_token, fuse_att = [], '', []  # Reset
            else:
                if i < len(tokens_enum) - 1:
                    if tokens_enum[i + 1][1].startswith('##'):
                        fuse_index.append(tidx)
                        # Add the one word before the first '##' token
                        fuse_token += token
                        fuse_att.append(a)
                        continue
                # Append to results ("nothing happens" case)
                append_cleaned([tidx], token, [a])
        if fuse_att:
            append_cleaned(fuse_index, fuse_token, fuse_att, replace=['##', ''])

    elif type(tokenizer_model) == models.BPE:  # RoBERTa
        for i, (token_enum, a) in enumerate(zip(tokens_enum, atts)):
            tidx, token = token_enum
            if len(fuse_att) > 0:
                if token != 'Ġ':
                    append_cleaned(fuse_index, fuse_token, fuse_att)
                    fuse_index, fuse_token, fuse_att = [], '', []  # Reset
            fuse_index.append(tidx)
            fuse_token += token.replace('Ġ', '')
            fuse_att.append(a)
        if fuse_att:
            append_cleaned(fuse_index, fuse_token, fuse_att, replace=['Ġ', ''])

    else:
        raise NotImplementedError('Not a valid backend tokenizer model (Unigram, WordPiece, BPE).')

    tokens = cleaned_tokens
    atts = cleaned_atts

    return tokens, atts


def apply_fuse_strategy(fuse_att, strategy):
    if strategy == 'average':
        return sum(fuse_att) / len(fuse_att)
    elif strategy == 'salient':
        return max(fuse_att, key=abs)
