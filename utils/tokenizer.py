import MeCab


class MeCabTokenizer:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")

    def tokenize(self, text):
        text = self.tagger.parse(text)
        return text.split()
