from typing import List
from mlprogram.languages import Token
from tokenize import generate_tokens, NUMBER, STRING, ENCODING, ENDMARKER, NEWLINE
import io
import re


class TokenizeQuery:
    def __call__(self, text: str) -> List[Token]:
        int_to_idx = {}
        float_to_idx = {}
        str_to_idx = {}
        var_to_idx = {}
        retval = []

        textio = io.StringIO(text)
        tokens = [(toknum, v)
                  for toknum, v, _, _, _ in generate_tokens(textio.readline)]

        var_state = 0
        prev_token = None

        for toknum, v in tokens:
            if toknum in set([ENCODING, ENDMARKER, NEWLINE]):
                continue
            if re.match("\s+", v):
                continue
            if toknum == NUMBER:
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
                        retval.append(Token(None, v, v))
            elif toknum == STRING:
                v = eval(v)
                if v not in str_to_idx:
                    str_to_idx[v] = len(str_to_idx)
                retval.append(Token(None, f"___str@{str_to_idx[v]}___", v))
            else:
                if var_state == 0:
                    if v == "`":
                        var_state = 1
                    else:
                        retval.append(Token(None, v, v))
                elif var_state == 1:
                    if v == "`":
                        retval.append(Token(None, v, v))
                        retval.append(Token(None, v, v))
                        var_state = 0
                    else:
                        var_state = 2
                else:
                    # `[token]<cursor>
                    if v == "`":
                        if prev_token not in var_to_idx:
                            var_to_idx[prev_token] = len(var_to_idx)
                        retval.append(Token(None, f"___var@{var_to_idx[prev_token]}___",
                                            prev_token))
                    else:
                        retval.append(Token(None, "`", "`"))
                        retval.append(Token(None, prev_token, prev_token))
                        retval.append(Token(None, v, v))

                    var_state = 0
            prev_token = v
        if var_state == 1:
            retval.append(Token(None, "`", "`"))
        elif var_state == 2:
            retval.append(Token(None, "`", "`"))
            retval.append(Token(None, prev_token, prev_token))

        return retval


class SplitValue:
    def __call__(self, token: str) -> List[str]:
        return [token]
