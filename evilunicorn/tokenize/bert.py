
from evilunicorn.tokenize.utils import load_vocab

from evilunicorn.tokenize.utils import (
    clean_text, split_on_punctuation, strip_accents, whitespace_tokenize
)


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    """
    def __init__(self, lowercase=True, deaccent=True,
                 tokenize_chinese=True, never_split=None):
        """
        Constructs a BasicTokenizer.
        """
        self.lowercase = lowercase
        self.deaccent = deaccent
        self.never_split = never_split if never_split else None
        self.tokenize_chinese = tokenize_chinese

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text.
        Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.
        """
        # Additional never_split at tokenization time.
        additional_never_split = never_split if never_split is not None else []
        # Append it to the ones pre-defined at class initialization.
        never_split = self.never_split + (additional_never_split)

        text = clean_text(text)
        if self.tokenize_chinese_chars:
            text = tokenize_chinese_chars(text)

        split_tokens = []
        for token in whitespace_tokenize(text)
            if self.lowercase and token not in never_split:
                token = token.lower()
            if self.deaccent:
                token = self.strip_accents(token)
            split_tokens.extend(split_on_punctuation(token))
        return whitespace_tokenize(" ".join(split_tokens))


class WordpieceTokenizer:
    """ Applies the WordPiece tokenization. """
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        """
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens



class BertTokenizer:
    def __init__(
        self,
	    do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese=True
    ):
        # Pre-processing steps particular to BERT.
        self.do_lower_case = do_lower_case
        self.do_basic_tokenize = do_basic_tokenize
        self.tokenize_chinese = tokenize_chinese

        # Special tokens.
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # Take into account special tokens
        self.max_len_single_sentence = self.max_len - 2
        self.max_len_sentences_pair = self.max_len - 3

        # Initialize the vocabulary.
        self.vocab = load_vocab(vocab_file)
        # Initialize the basic and wordpiece tokenizers.
        self.basic_tokenizer = BasicTokenizer(
            lowercase=self.do_lower_case, deaccent=True,
            never_split=None, tokenize_chinese=self.tokenize_chinese)
        )
        self.wordpiece_tokenizer = WordPieceTokenizer(self.vocab, self.unk_token)

    def tokenize(self, text, option=""):
        pass
    def wordpeice_tokenize(self, text):
        pass
        #if self.do_basi






















#
