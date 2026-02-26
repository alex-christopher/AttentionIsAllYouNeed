import os
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import decoders
from tokenizers.processors import TemplateProcessing


special_tokens_dict = {
    "unknown_token" : "[UNK]",
    "pad_token": "[PAD]",
    "start_token": "[BOS]",
    "end_token": "[EOS]"
}

def train_tokenizer(path_to_data_root):

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    french_files = glob.glob(os.path.join(path_to_data_root, "**/*.fr"))
    
    trainer = WordPieceTrainer(vocab_size=32000, special_tokens=list(special_tokens_dict.values()))
    tokenizer.train(french_files, trainer)
    tokenizer.save("trained_tokenizer/french_wp.json")

class FrenchTokenizer:

    def __init__(self, path_to_vocab, truncate=True, max_length=512):

        self.path_to_vocab = path_to_vocab
        self.tokenizer = self.prepare_tokenizer()
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.special_tokens_dict = {
            "[UNK]": self.tokenizer.token_to_id("[UNK]"),
            "[PAD]": self.tokenizer.token_to_id("[PAD]"),
            "[BOS]": self.tokenizer.token_to_id("[BOS]"),
            "[EOS]": self.tokenizer.token_to_id("[EOS]")
        }

        self.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", self.special_tokens_dict["[BOS]"]),
                ("[EOS]", self.special_tokens_dict["[EOS]"])
            ]
        )

        self.truncate = truncate
        if self.truncate:
            self.max_length = max_length - self.post_processor.num_special_tokens_to_add(is_pair=False)
            
    def prepare_tokenizer(self):

        tokenizer = Tokenizer.from_file(self.path_to_vocab)
        tokenizer.decoder = decoders.WordPiece()
        
        return tokenizer        
    
    def encode(self, input):

        def _process_tokens(tokenized):

            if self.truncate:
                tokenized.truncate(self.max_length, direction="right")

            tokenized = self.post_processor.process(tokenized)

            return tokenized.ids

        if isinstance(input, str):
            tokenized = self.tokenizer.encode(input)
            tokenized = _process_tokens(tokenized)
            print(tokenized)

        if isinstance(input, list):
            tokenized = self.tokenizer.encode_batch(input)
            tokenized = [_process_tokens(t) for t in tokenized]

        return tokenized

    def decode(self, input, skip_special_tokens=True):
        
        if isinstance(input, list):
        
            if all(isinstance(item, list) for item in input):

                decoded = self.tokenizer.decode_batch(input, skip_special_tokens=skip_special_tokens)

            elif all(isinstance(item, int) for item in input):
                decoded = self.tokenizer.decode(input, skip_special_tokens=skip_special_tokens)

        
        return decoded


if __name__ == "__main__":

    path_to_data_root = "E:/datasets/english2french"
    # train_tokenizer(path_to_data_root=path_to_data_root)
    tok = FrenchTokenizer(path_to_vocab="trained_tokenizer/french_wp.json")

    sentence = "Hello world!"
    enc = tok.encode([sentence, sentence])
    print(tok.decode(enc, False))