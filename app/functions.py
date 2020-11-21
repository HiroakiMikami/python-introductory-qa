from typing import List
from mlprogram.languages import Token
from nltk.tokenize import WhitespaceTokenizer


class TokenizeQuery:
    tokenizer = WhitespaceTokenizer()

    def __call__(self, text: str) -> List[Token]:
        tokens = TokenizeQuery.tokenizer.tokenize(text)
        int_to_idx = {}
        float_to_idx = {}
        str_to_idx = {}
        retval = []
        for v in tokens:
            try:
                int(v)
                if v not in int_to_idx:
                    int_to_idx[v] = len(int_to_idx)
                retval.append(Token(None, f"___int@{int_to_idx[v]}___", v))
            except:  # noqa
                try:
                    float(v)
                    if v not in float_to_idx:
                        float_to_idx[v] = len(float_to_idx)
                    retval.append(Token(None, f"___float@{float_to_idx[v]}___", v))
                except:  # noqa
                    if v.startswith("\"") and v.endswith("\""):
                        if v not in str_to_idx:
                            str_to_idx[v] = len(str_to_idx)
                        retval.append(Token(None, f"___str@{str_to_idx[v]}___", v))
                    else:
                        retval.append(Token(None, v, v))
        return retval


class SplitValue:
    def __call__(self, token: str) -> List[str]:
        return [token]
