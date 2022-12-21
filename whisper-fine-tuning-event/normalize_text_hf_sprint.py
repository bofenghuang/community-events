#! /usr/bin/env python3
# coding=utf-8
# Copyright 2022 Bofeng Huang

"""Normalize french text data for ASR."""

import re
import unicodedata

from num2words import num2words


ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str, keep=""):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        c if c in keep else " " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s)
    )


def escape_string(s):
    return r"".join([rf"\{s_}" for s_ in s])


class FrenchNumberNormalizer:
    def __init__(self, lang: str = "fr", converter: str = "cardinal"):
        self.lang = lang
        self.converter = converter

    def __call__(self, s: str):
        length_diff = 0
        for match in re.finditer(r"[1-9][0-9]*|(?:(?<=[^0-9])|(?<=^))0", s):
            num_word = num2words(match.group(), lang=self.lang, to=self.converter)
            start, end = match.start() + length_diff, match.end() + length_diff
            s = f"{s[:start]} {num_word} {s[end:]}"
            # +2 espaces
            length_diff += len(num_word) - (end - start) + 2
        return s


class FrenchTextNormalizer:
    def __init__(self, remove_diacritics: bool = False):
        # self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        # todo: add French fillers
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um|ah|bah|beh|ben|eh|euh|hein|hum|mmh|oh|pff)\b"
        # self.ignore_patterns = None

        self.replacers = {
            # standarize symbols
            r"’|´|′|ʼ|‘|ʻ|`": "'",  # replace special quote
            r"−|‐": "-",  # replace special dash
            # standarize characters (for french)
            r"æ": "ae",
            r"œ": "oe",
            # ordinal
            # r"1er": "premier",
            # r"2ème": "deuxième",
            # numbers
            # r"(\d),(\d)": r"\1 virgule \2",
            # r"(\d).(\d)": r"\1 point \2",
            # r"(\d)\s?\%": r"\1 pour cent ",
            # r"(?<=\d)\s(?=000)": "",  # 1 000 -> 1000
            # others
            # r"(?<=\d)h(?=\d|\s|$)": r"\1 heures \2",
            r"€": " euro ",
            r"\$": " dollar ",
            # r"&": " et ",
            # r"m\.": "monsieur",
        }

        self.numbers_before_dash = [
            "dix",
            "vingt",
            "trente",
            "quarante",
            "cinquante",
            "soixante",
            "quatre",
        ]

        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

        self.standardize_numbers = FrenchNumberNormalizer()
        # self.standardize_spellings = EnglishSpellingNormalizer()

        # latin chars
        # bh: speechbrain version for "en", "fr", "it", "rw"
        self.allowed_chars = "’'A-Za-z0-9À-ÖØ-öø-ÿЀ-ӿéæœâçèàûî"
        # bh: lowercased
        # self.allowed_chars = "a-zàâäéèêëîïôöùûüÿçñ"
        # bh: all
        # check https://en.wikipedia.org/wiki/List_of_Unicode_characters
        french_chars_lower = "a-zàâäéèêëîïôöùûüÿçñ"
        french_chars_upper = "A-ZÀÂÄÇÈÉÊËÎÏÔÖÙÛÜŸ"
        number_chars = "0-9"
        self.allowed_chars = french_chars_lower + french_chars_upper + number_chars

    def __call__(self, s: str, do_lowercase=True, symbols_to_keep="'", do_standardize_numbers=True):
        if do_lowercase:
            s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        # s = re.sub(r"^\s*#\d{3}\s", "", s)  # remove beginning http response code

        if self.ignore_patterns is not None:
            s = re.sub(self.ignore_patterns, "", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        if do_standardize_numbers:
            s = self.standardize_numbers(s)  # convert number to words
        # s = self.standardize_spellings(s)

        # s = self.clean(s, keep="-'")
        # s = re.sub(rf"[^{self.allowed_chars}\'\- ]", "", s)
        # don't keep dash for hf event
        s = self.clean(s, keep=symbols_to_keep)  # remove any other markers, symbols, punctuations with a space
        # s = re.sub(rf"[^{self.allowed_chars}\' ]", "", s)  # remove unnecessary alphabet characters
        s = re.sub(rf"[^{self.allowed_chars}{escape_string(symbols_to_keep)}\s]", " ", s)  # remove unnecessary alphabet characters

        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe
        s = re.sub(r"'\s+", "'", s)  # standardize when there's a space after an apostrophe
        # s = re.sub(rf"([{self.allowed_chars}])\s+'", r"\1'", s)  # standardize when there's a space before an apostrophe
        # s = re.sub(rf"([{self.allowed_chars}])'([{self.allowed_chars}])", r"\1' \2", s)  # add an espace after an apostrophe
        # s = re.sub(rf"(?<!aujourd)(?<=[{self.allowed_chars}])'(?=[{self.allowed_chars}])", "' ", s)  # add an espace after an apostrophe (except)

        # s = re.sub(r"(?<!\b{})-(?=\S)".format(r")(?<!\b".join(self.numbers_before_dash)), " ", s)  # remove dash not after numbers
        # s = re.sub(r"(?:(?<=\s)|(?<=^))\-+(?=\w)", " ", s)  # remove beginning dash in words
        # s = re.sub(r"(?<=\w)\-+(?=\s|$)", " ", s)  # remove trailing dash in words
        # s = re.sub(r"(?:^|\s)\-+\w*", " ", s)  # remove words with beginning dash
        # s = re.sub(r"\w*\-+(?=\s|$)", " ", s)  # remove words with trailing dash

        # s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        # s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        # s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        # s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s).strip()  # replace any successive whitespace characters with a space

        return s


if __name__ == "__main__":
    import sys
    raw_text = " ".join(sys.argv[1:])
    print(raw_text)
    print(FrenchTextNormalizer()(raw_text))
