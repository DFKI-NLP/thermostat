from tokenizers import models
from typing import List


def fuse_subwords(tokens: List, atts: List, tokenizer, strategy=None) -> (List, List):
    assert strategy in ['average', 'salient']
    tokenizer_model = tokenizer.backend_tokenizer.model

    fuse_token = ''
    fuse_att = []
    cleaned_tokens = []
    cleaned_atts = []

    if type(tokenizer_model) == models.Unigram:  # ALBERT, XLNet
        for i, (t, a) in enumerate(zip(tokens, atts)):
            if len(fuse_att) > 0:
                if t.startswith('▁') and t != '▁':
                    cleaned_tokens.append(fuse_token.replace('▁', ''))
                    cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))
                    fuse_token = ''
                    fuse_att = []
            fuse_token += t
            fuse_att.append(a)
        if fuse_att:
            cleaned_tokens.append(fuse_token.replace('▁', ''))
            cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))

    elif type(tokenizer_model) == models.WordPiece:  # BERT, ELECTRA
        for i, (t, a) in enumerate(zip(tokens, atts)):
            if t.startswith('##'):
                # Append all subsequent '##' subword tokens
                fuse_token += t.replace('##', '')
                fuse_att.append(a)
                if i < len(tokens) - 1:
                    if not tokens[i + 1].startswith('##'):
                        # Append to results
                        cleaned_tokens.append(fuse_token)
                        cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))
                        # Reset
                        fuse_token = ''
                        fuse_att = []
            else:
                if i < len(tokens) - 1:
                    if tokens[i + 1].startswith('##'):
                        # Add the one word before the first '##' token
                        fuse_token += t
                        fuse_att.append(a)
                        continue
                # Append to results ("nothing happens" case)
                cleaned_tokens.append(t)
                cleaned_atts.append(a)
        if fuse_att:
            cleaned_tokens.append(fuse_token.replace('##', ''))
            cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))

    elif type(tokenizer_model) == models.BPE:  # RoBERTa
        for i, (t, a) in enumerate(zip(tokens, atts)):
            if len(fuse_att) > 0:
                if t != 'Ġ':
                    cleaned_tokens.append(fuse_token)
                    cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))
                    fuse_token = ''
                    fuse_att = []
            fuse_token += t.replace('Ġ', '')
            fuse_att.append(a)
        if fuse_att:
            cleaned_tokens.append(fuse_token.replace('Ġ', ''))
            cleaned_atts.append(apply_fuse_strategy(fuse_att, strategy))

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
