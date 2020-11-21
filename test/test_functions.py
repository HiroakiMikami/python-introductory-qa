from app.functions import TokenizeQuery
from mlprogram.languages import Token


def test_tokenize_query():
    assert TokenizeQuery()("foo bar") == [Token(None, "foo", "foo"),
                                          Token(None, "bar", "bar")]
    assert TokenizeQuery()("1 1 2") == [Token(None, "___int@0___", "1"),
                                        Token(None, "___int@0___", "1"),
                                        Token(None, "___int@1___", "2")]
    assert TokenizeQuery()("1.0 1.0 2.0") == [Token(None, "___float@0___", "1.0"),
                                              Token(None, "___float@0___", "1.0"),
                                              Token(None, "___float@1___", "2.0")]
    assert TokenizeQuery()("\"x\" \"y\" \"x\"") == [Token(None, "___str@0___", "\"x\""),
                                                    Token(None, "___str@1___", "\"y\""),
                                                    Token(None, "___str@0___", "\"x\"")]
