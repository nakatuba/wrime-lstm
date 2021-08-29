import MeCab


class MeCabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger(
            "-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
        )

    def tokenize(self, text):
        text = self.tagger.parse(text)
        return text.split()
