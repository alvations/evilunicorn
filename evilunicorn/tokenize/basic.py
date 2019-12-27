from evilunicorn.tokenize.bert.utils import *

class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    """
    def __init__(self, lowercase=True, deaccent=True,
                 never_split=None, tokenize_chinese=True):
        """
        Constructs a BasicTokenizer.
        """
        self.lowercase = lowercase
        self.deaccent = deaccent
        self.never_split = never_split if never_split else None
        self.tokenize_chinese = tokenize_chinese

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
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
