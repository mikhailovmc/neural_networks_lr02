import re
import demoji
from deep_translator import GoogleTranslator


progress = 0
emoji_vocab = {}


def translate_text(text):
    global emoji_vocab

    if text in emoji_vocab:
        return emoji_vocab[text]

    translated_text = GoogleTranslator(source='en', target='ru').translate(text)

    emoji_vocab[text] = translated_text

    return translated_text


def replace_between_colons(text):
    pattern = r':([^:]*):'
    return re.sub(pattern, lambda match: f":{translate_text(match.group(1))}:", text)


def translate_text_with_emoji(text):
    global progress
    progress += 1
    if progress % 10 == 0:
        print(progress)
    if demoji.findall(text):
        cleaned_text = demoji.replace_with_desc(text)
        cleaned_text = cleaned_text.replace("flag:", "flag")
        modified_text = replace_between_colons(cleaned_text)
        modified_text = modified_text.replace(':', ' ')
        return modified_text
    return text
