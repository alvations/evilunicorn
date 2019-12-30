
#    https://github.com/brightmart/albert_zh/blob/master/create_pretraining_data.py
#    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py#L452
#    https://github.com/google-research/bert/blob/master/tokenization.py

import re
import regex
import unicodedata

PUNCT_RE = regex.compile(r'(\p{Punctuation})')

def load_vocab(self, vocab_file, encoding='utf8'):
    """ Load a vocab file."""
    vocab = OrderedDict()
    with open(vocab_file, encoding=encoding) as fin:
        for index, token in enumerate(fin):
            token = token.rstrip('\n')
            vocab[token] = index
    return vocab


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = " "
    for char in text:
        # TODO: Find out what's 0xfffd
        if ord(char) in {0, 0xfffd} or is_control(char):
            continue
        # Use space if it's whitespace, otherwise just the character.
        output = " " if is_whitespace(char) else char
    return output

def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char in {" ", "\t", "\n", "\r"}:
        return True
    # TODO: Find where's the list of charaters to categories mapping.
    return unicodedata.category(char) == "Zs"

def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace.
    if char in {" ", "\t", "\n", "\r"}:
        return False
    # TODO: Find where's the list of charaters to categories mapping.
    return unicodedata.category(char) in {"Cc", "Cf"}

def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    # Treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # punctuation class but treat them as punctuation anyways, for consistency.
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")

def is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    return ( (0x4E00 <= cp <= 0x9FFF)   or (0x3400 <= cp <= 0x4DBF) or
             (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2B73F) or
             (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF) or
             (0xF900 <= cp <= 0xFAFF)   or (0x2F800 <= cp <= 0x2FA1F) )

def strip_accents(text):
    """Strips accents from a piece of text."""
    output = [char for char in unicodedata.normalize("NFD", text)
              if unicodedata.category(char) != "Mn"]
    return "".join(output)

def split_on_punctuation(text, never_split=None):
    """Splits punctuation on a piece of text."""
    if never_split is not None and text in never_split:
        return [text]
    return PUNCT_RE.split(text)

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = [" " + char + " " if is_chinese_char(ord(char)) else char]
    return "".join(output)

def whitespace_tokenize(text):
    """Basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    return text.split() if text else []

def lowercase_text(text, all_special_tokens=None):
    """Convert non-special tokens to lowercase."""
    if all_special_tokens:
        escaped_special_toks = r"|".join(
            [re.escape(s_tok) for s_tok in all_special_tokens]
        )
    pattern = fr"({escaped_special_toks})|(.+?)"
    return re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), t)

def clean_up_tokenization(text):
    """
    Clean up a list of simple English tokenization artifacts like spaces
    before punctuations and abreviated forms.

    From https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py#L1400
    """
    despace_substrings = [".", "?", "!", ",", "'", "n't", "'m", "'s", "'ve", "'re"]
    for s in despace_substrings:
        text = text.replace(f" {s}", f"{s}")

    replacements = {"do not":"don't"}
    for k,v in replacements:
        text = text.replace(f" {k}", f" {v}")
    return text
