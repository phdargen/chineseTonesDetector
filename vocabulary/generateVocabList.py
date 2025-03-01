import os
import argparse
import torch
from transformers import (
    BertTokenizer, BertModel, 
    AutoModelForSeq2SeqLM, AutoTokenizer,
    TextGenerationPipeline,
    VitsModel, VitsTokenizer
)
from pypinyin import pinyin, Style
import genanki
import random
import soundfile as sf
import io
import requests
from openai import OpenAI
client = OpenAI()
    
def get_common_chinese_words(num_words=1000):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    vocab = tokenizer.get_vocab()
    
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'
    
    chinese_words = [
        word for word, index in vocab.items() 
        if len(word) >= 1 and any(is_chinese(char) for char in word)
    ]
    sorted_words = sorted(chinese_words, key=lambda word: vocab[word])
    if not sorted_words:
        raise ValueError("No Chinese words found in the vocabulary")
    return sorted_words[:num_words]

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_text_chatgpt(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the following Chinese text to English. The text is intended for flash cards. "},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

def generate_sentence(word):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("uer/t5-base-chinese-cluecorpussmall")
        tokenizer = AutoTokenizer.from_pretrained("uer/t5-base-chinese-cluecorpussmall")
        
        prompt = f"使用\"{word}\"造一个日常对话中的句子："
        inputs = tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
        
        # Remove token_type_ids if present
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True, top_p=0.95, top_k=50)
        sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure the target word is in the sentence
        if word not in sentence:
            sentence = f"{sentence} {word}。"
        
        return sentence.strip()
    except Exception as e:
        print(f"Error generating sentence: {str(e)}")
        return f"这是一个使用\"{word}\"的例句。" 

def generate_sentence_chatgpt(word):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Chinese language assistant. Create a simple sentence using the given word."},
            {"role": "user", "content": f"Create a simple Chinese sentence using the word '{word}'."}
        ]
    )
    return response.choices[0].message.content

def generate_audio(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        audio = model(**inputs).waveform
    return audio.squeeze().numpy()

def generate_audio_chatgpt(text,filename):
    response = client.audio.speech.create(
        model="tts-1-hd", #"tts-1-hd"
        voice="nova",
        input=text
    ) 
    response.stream_to_file(filename)
    return filename

def add_info_and_generate_audio(words, useChatGPT = False):

    if not useChatGPT:
        print("Loading models...")
        translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
        translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

        #tts_model = VitsModel.from_pretrained("espnet/kan-bayashi_ljspeech_vits")
        #tts_tokenizer = VitsTokenizer.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

    word_info = []
    for i, word in enumerate(words, 1):
        print(f"Processing word {i}/{len(words)}: {word}")
        py = ' '.join([p[0] for p in pinyin(word)])

        if useChatGPT:
            en = translate_text_chatgpt(word)
            sentence = generate_sentence_chatgpt(word)
            word_audio = generate_audio_chatgpt(word,f"audio/word_{i}.wav")
            sentence_audio = generate_audio_chatgpt(sentence,f"audio/sentence_{i}.wav")
        else:
            en = translate_text(word, translation_model, translation_tokenizer)
            sentence = generate_sentence(word)
            word_audio = None
            sentence_audio = None
        
        py_sentence = ' '.join([p[0] for p in pinyin(sentence)])

        word_info.append((word, py, en, sentence, py_sentence, word_audio, sentence_audio))
    
    return word_info

def create_anki_deck(word_info):
    model_id = random.randrange(1 << 30, 1 << 31)
    model = genanki.Model(
        model_id,
        'Chinese Word Model',
        fields=[
            {'name': 'Chinese'},
            {'name': 'Pinyin'},
            {'name': 'English'},
            {'name': 'Sentence'},
            {'name': 'WordAudio'},
            {'name': 'SentenceAudio'},
        ],
        templates=[
            {
                'name': 'Word to English',
                'qfmt': '{{Chinese}}<br>{{Pinyin}}<br>{{WordAudio}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{English}}<br>{{Sentence}}<br>{{SentenceAudio}}',
            },
            {
                'name': 'English to Word',
                'qfmt': '{{English}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Chinese}}<br>{{Pinyin}}<br>{{WordAudio}}<br>{{Sentence}}<br>{{SentenceAudio}}',
            },
        ])

    deck_id = random.randrange(1 << 30, 1 << 31)
    deck = genanki.Deck(deck_id, 'Common Chinese Words')

    for word, pinyin, english, sentence, py_sentence, word_audio, sentence_audio in word_info:
        note = genanki.Note(
            model=model,
            fields=[word, pinyin, english, sentence, 
                    f'[sound:{word_audio}]',
                    f'[sound:{sentence_audio}]']
        )
        deck.add_note(note)

    package = genanki.Package(deck)
    #package.media_files = [f"{word}_audio.wav" for word, _, _, _, _, _ in word_info] + \
     #                     [f"{word}_sentence_audio.wav" for word, _, _, _, _, _ in word_info]
    package.write_to_file('common_chinese_words.apkg')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--useChatGPT', action='store_true', help="Use ChatGPT API")
    args = parser.parse_args()
    print("Settings: ")
    print(f"Use ChatGPT API: {args.useChatGPT}")

    print("Fetching common Chinese words...")
    common_words = get_common_chinese_words(10) 
    print(common_words)
    for word in common_words:
        print(f"{word}")

    print("Adding translations, generating sentences and audio...")
    word_info = add_info_and_generate_audio(common_words, args.useChatGPT)
    
    print("Creating Anki deck...")
    create_anki_deck(word_info)
    print("Anki deck 'common_chinese_words.apkg' has been created.")
    
    print("\nSample of the word list:")
    for word, py, en, sentence, py_sentence, _, _ in word_info:
         print(f"{word} ({py}): {en}")
         print(f"Example: {sentence}\n")
         print(f"{py_sentence}\n")

if __name__ == "__main__":
    main()