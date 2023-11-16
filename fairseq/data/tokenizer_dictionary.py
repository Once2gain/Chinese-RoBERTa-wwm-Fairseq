# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .dictionary import Dictionary
from transformers import BertTokenizer
from fairseq.file_io import PathManager
import os

class TokenizerDictionary(Dictionary):
    """for bert like vocab that has contained special tokens such as [CLS]、[SEP]"""

    def __init__(
            self,
            tokenizer: BertTokenizer,
    ):
        # do not call to __init__ of super class
        self.tokenizer = tokenizer

        self.symbols = list(self.tokenizer.get_vocab().keys())
        self.indices = self.tokenizer.get_vocab()

        self.bos_index = tokenizer.cls_token_id
        self.pad_index = tokenizer.pad_token_id
        self.eos_index = tokenizer.sep_token_id
        self.unk_index = tokenizer.unk_token_id
        self.mask_index = tokenizer.mask_token_id

        self.bos_word = tokenizer.cls_token
        self.unk_word = tokenizer.unk_token
        self.pad_word = tokenizer.pad_token
        self.eos_word = tokenizer.sep_token
        self.mask_word = tokenizer.mask_token

        self.nspecial = 0

    def string(
            self,
            tensor,
            bpe_symbol=None,
            escape_unk=False,
            extra_symbols_to_ignore=None,
            unk_string=None,
            include_eos=False,  # is not used
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
    ):
        assert unk_string is None, "unk_string is not supported"
        assert bpe_symbol is None, "bpe_symbols is not supported"
        assert extra_symbols_to_ignore is None, "extra_symbols_to_ignore is not supported"
        assert escape_unk is False, "escape_unk is not supported, use skip_special_tokens=True"
        assert include_eos is False, "include_eos is not used"

        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces)
                for t in tensor
            )
        return self.tokenizer.decode(tensor, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        raise NotImplementedError("add_symbols should not be supported")

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        raise NotImplementedError("update should not be supported")

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        raise NotImplementedError("finalize should not be supported")

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        raise NotImplementedError("pad_to_multiple should not be supported")

    @classmethod
    def load(cls, model_path, add_special_symbols=True):
        """Loads the dictionary from the pretrained models' directory"""
        # add_special_symbols is not use
        tokenizer = BertTokenizer.from_pretrained(model_path)
        d = cls(tokenizer=tokenizer)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        raise NotImplementedError("add_from_file should not be supported")

    def _save(self, f, vocab):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for v in vocab:
            print(str(v), file=f)

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            ex_keys + self.symbols,
        )

    def dummy_sentence(self, length):
        raise NotImplementedError("dummy_sentence is not be supported yet")

    @staticmethod
    def _add_file_to_dictionary_single_worker(
            filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        raise NotImplementedError("_add_file_to_dictionary_single_worker is not supported")

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        raise NotImplementedError("add_file_to_dictionary_single_worker is not supported")