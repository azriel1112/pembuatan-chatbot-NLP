#import library
import re
import emoji
from util import JSONParser
import string
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#jika harus melakukan steaming normalisation poststaging disarankan menuliskan di dalam folder util
def load_kamus_txt(filepath):
    kamus = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                key, value = line.strip().split('\t', 1)
                kamus[key.lower()] = value.lower()
    return kamus
# preproces text input
def preprocess_text(text, kamus_dict): 
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Hapus URL, mention, hashtag, angka
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+", "", text)

    # Hapus emoji
    text = emoji.replace_emoji(text, replace="")

    # Hapus tanda baca
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenisasi dan normalisasi
    words = text.split()
    normalized = [kamus_dict.get(w, w) for w in words]
    text= " ".join(normalized)

    return text

def preprocess(chat):
    #konversi ke non kapital
    chat=chat.lower()
    #hilangkan tanda baca
    tandabaca=tuple(string.punctuation)
    chat= ''.join(ch for ch in chat if ch not in tandabaca)
    return chat


def bot_response(chat, pipeline, jp):
    # Load kamus dari file normalization.txt
    kamus_path = "data/normalization.txt"
    kamus_singkatan = load_kamus_txt(kamus_path)
    chat=preprocess_text(chat, kamus_singkatan)
    res=pipeline.predict_proba([chat])
    max_prob=max(res[0])
    if max_prob < 0.1:
        return "maaf kak, aku ga ngerti :(", None, max_prob
    else:
        max_prob=max(res[0])
        max_id= np.argmax(res[0])
        pred_text=pipeline.classes_[max_id]
        return jp.get_response(pred_text), pred_text, max_prob

    jp.get_response()

# Load kamus dari file normalization.txt
kamus_path = "data/normalization.txt"
#load data
path='data/intents.json'
jp= JSONParser()
jp.parse(path)
df=jp.get_dataframe()

#praproses data
#case folding ->transform kapital ke non kapital, hilangkan tanda baca
df['text_input_prep']=df.text_input.apply(preprocess)

#pemodelan
pipeline=make_pipeline(CountVectorizer(),MultinomialNB())

#train 
print("[INFO] Training Data ...")
pipeline.fit(df.text_input_prep, df.intents)

#interaction with bot
print("[INFO] Anda Sudah Terhubung dengan Bot Kami")
while True:
    chat=input("anda >> ")
    res, tag, max_prob=bot_response(chat, pipeline, jp)
    print(f"Bot >> {res} \t (probabilitas= {max_prob})")
    if tag=='bye':
        break

