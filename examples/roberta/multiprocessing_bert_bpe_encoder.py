#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
from fairseq.data.encoders.hf_bert_bpe import BertBPE, BertBPEConfig
import unicodedata
import jieba


def main():
    """
    Helper script to encode raw text with the Bert BPE using multiple processes.
    Focus on address mlm task finetune on /chinese_roberta_wwm_ext/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="path to vocab.txt",
    )
    parser.add_argument(
        "--do-wwm",
        action="store_true",
        help="Whole Word Masking",
    )
    parser.add_argument(
        "--user-dict",
        type=str,
        default=None,
        help="path to user dict of jieba",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--refs",
        nargs="+",
        default=["-"],
        help="path to save ref outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"

    if args.user_dict is not None:
        jieba.load_userdict(args.user_dict)

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in args.outputs
        ]
        refs = [
            stack.enter_context(open(ref, "w", encoding="utf-8"))
            if ref != "-"
            else sys.stdout
            for ref in args.refs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        stats = Counter()
        for i, (filt, enc_lines, enc_masks) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, enc_mask, output_h, ref_h in zip(enc_lines, enc_masks, outputs, refs):
                    print(enc_line, file=output_h)
                    print(enc_mask, file=ref_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        config = BertBPEConfig(bpe_vocab_file=self.args.vocab_file)
        bpe = BertBPE(config)

    def encode(self, line):
        global bpe
        tokens = bpe.tokenize(line)

        if not self.args.do_wwm:
            return list(map(str, tokens))

        def is_control(char):
            """
            Checks whether `char` is a control character.
            Copied from transformers.models.bert.tokenization_bert.
            """
            # These are technically control characters but we count them as whitespace
            # characters.
            if char == "\t" or char == "\n" or char == "\r":
                return False
            cat = unicodedata.category(char)
            if cat.startswith("C"):
                return True
            return False

        def is_whitespace(char):
            """
            Checks whether `char` is a whitespace character.
            Copied from transformers.models.bert.tokenization_bert.
            """
            # \t, \n, and \r are technically control characters but we treat them
            # as whitespace since they are generally considered as such.
            if char == " " or char == "\t" or char == "\n" or char == "\r":
                return True
            cat = unicodedata.category(char)
            if cat == "Zs":
                return True
            return False

        def clean_text(text):
            """
            Performs invalid character removal and whitespace cleanup on text.
            Copied from transformers.models.bert.tokenization_bert.
            """
            output = []
            for char in text:
                cp = ord(char)
                if cp == 0 or cp == 0xFFFD or is_control(char):
                    continue
                if is_whitespace(char):
                    output.append(" ")
                else:
                    output.append(char)
            return "".join(output)

        words = list(jieba.cut(clean_text(line)))
        word_dic = {}
        words = set(words)

        for word in words:
            word_dic[word[0]] = word_dic.get(word[0], [])
            word_dic[word[0]].append(word)

        for words in word_dic.values():
            words.sort(key=lambda wd: len(wd), reverse=True)

        wwm_mask = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('##'):
                wwm_mask.append(0)
                i = i + 1
                continue

            if token == 'UNK' or token[0] not in word_dic:
                wwm_mask.append(1)
                i = i + 1
                continue

            words = word_dic[token[0]]
            assert isinstance(words, list)
            for word in words:
                assert isinstance(word, str)
                if ''.join(list(map(lambda x: x.split('##')[-1], tokens[i:]))).startswith(word):
                    if word in token and word != token:
                        wwm_mask.append(0)
                        break
                    word = word[word.index(token) + len(token):]
                    wwm_mask.append(1)
                    while word:
                        i = i + 1
                        token = tokens[i]
                        if token.startswith('##'):
                            token = token[2:]

                        # in case: a word is part of a token
                        if word in token and word != token:
                            wwm_mask.append(0)
                            break

                        word = word[word.index(token) + len(token):]
                        wwm_mask.append(0)
                    break
            i = i + 1

        return tokens, list(map(str, wwm_mask))

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        def has_chinese(ss):
            for s in ss:
                if '\u4e00' <= s <= '\u9fff':
                    return True
            return False

        enc_lines = []
        enc_masks = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None, None]
            if not has_chinese(line):
                return ["EMPTY", None, None]
            tokens, wwm_mask = self.encode(line)
            try:
                assert len(tokens) == len(wwm_mask), str(line)
            except AssertionError:
                wwm_mask = ['1'] * len(tokens)
            enc_lines.append(" ".join(tokens))
            enc_masks.append(" ".join(wwm_mask))
        return ["PASS", enc_lines, enc_masks]


if __name__ == "__main__":
    main()
