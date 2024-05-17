from gtts import gTTS

sentence = "今天的天气非常好，我们一起去公园散步吧。"
tts = gTTS(text=sentence, lang='zh')

tts.save("sentence.mp3")
print("The audio file has been saved as 'sentence.mp3'.")
