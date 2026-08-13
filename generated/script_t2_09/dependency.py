import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
CLASSES = ["joy", "anger", "sadness", "surprise"]
MAX_LEN = 32
BATCH_SIZE = 16
EPOCHS = 15


print("====================================================")
print("Project 60: Emotion Classification from Text (BERT)")
print("====================================================")

# 1. Load Bert Tokenizer
print("Step 1: Loading pre-trained BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def generate_synthetic_dataset():
    raw_dataset = {
        "joy": [
            "I am so happy and excited today!",
            "This is the best news I have ever heard!",
            "We won the championship, this is wonderful!",
            "What a beautiful and sunny day!",
            "I feel so grateful and cheerful.",
            "Success brings so much happiness and delight.",
            "I love spending time with my family and friends.",
            "That joke was hilarious, I cannot stop laughing!",
            "I am absolutely thrilled with the results.",
            "I feel highly motivated and full of life today.",
            "It is a pleasure to meet such wonderful people.",
            "We celebrated our victory with joy and dancing.",
            "My heart is full of peace, love and content.",
            "This achievement makes me feel proud and happy.",
            "I am enjoying this beautiful melody.",
            "Today was a perfect day filled with smiles.",
            "I received a lovely gift from my friend.",
            "I am so glad everything worked out so well.",
            "It's a wonderful feeling to help someone in need.",
            "Laughter and happiness filled the entire room.",
            "I feel blessed and very fortunate.",
            "What a pleasant surprise and lovely evening!",
            "We had a fantastic holiday at the beach.",
            "This success exceeds all my expectations!",
            "I feel so warm and cozy inside.",
            "She has a bright and beautiful smile that lifts everyone.",
            "It is a magnificent day to go for a run.",
            "I am so proud of your hard work and achievements.",
            "Everything is going perfectly, I am overjoyed.",
            "Seeing you succeed makes my heart glow with pride."
        ] * 2,
        "anger": [
            "I am absolutely furious about this situation!",
            "Get out of my face, I hate this!",
            "This is completely unfair and makes me mad.",
            "I cannot believe they did this to me, it's annoying.",
            "Stop irritating me, I am so pissed off.",
            "This bad service is extremely frustrating and unacceptable.",
            "I am so angry that I want to scream!",
            "They lied to my face, and I am disgusted.",
            "His rude behavior is completely intolerable.",
            "I am sick and tired of these constant interruptions.",
            "This is a total disaster, and it's all your fault!",
            "He broke his promise, which makes me rage.",
            "I cannot stand his arrogant and selfish attitude.",
            "They completely ignored my request, I am offended.",
            "This delay is wasting my time and making me mad.",
            "Don't speak to me like that, it's highly disrespectful.",
            "I am extremely annoyed by this stupid mistake.",
            "Their negligence caused a massive failure, I am furious.",
            "Stop making excuses, it makes me even angrier.",
            "This is the worst experience of my life, I am raging.",
            "I am fed up with their terrible attitude.",
            "You have crossed the line, and I won't tolerate it.",
            "He ruined my project, I am absolutely livid.",
            "I feel so much resentment toward their actions.",
            "Stop pushing my buttons, I am about to lose my temper."
        ] * 2 + [
            "I am absolutely furious about this situation!",
            "Get out of my face, I hate this!",
            "This is completely unfair and makes me mad.",
            "I cannot believe they did this to me, it's annoying.",
            "Stop irritating me, I am so pissed off."
        ],
        "sadness": [
            "I feel so lonely and depressed lately.",
            "It is heartbreaking to see them leave.",
            "I am crying because of this terrible loss.",
            "Everything feels gloomy and hopeless today.",
            "I am deeply disappointed and sad.",
            "I miss my old friends and the good times we shared.",
            "The tragedy left everyone in deep sorrow and tears.",
            "I feel completely isolated and abandoned.",
            "My heart is heavy with grief and pain.",
            "It is hard to smile when everything is going wrong.",
            "I am feeling down and just want to be alone.",
            "This failure makes me feel worthless and unhappy.",
            "The cold rain matches the sadness in my heart.",
            "She is going through a very painful divorce.",
            "I feel so sorry for their loss, it's devastating.",
            "Life feels empty and directionless right now.",
            "The memories of that day still bring tears to my eyes.",
            "I am mourning the passing of my beloved pet.",
            "It is a dark and lonely night, full of regret.",
            "I feel rejected and unloved by everyone around me.",
            "This constant struggle is making me lose hope.",
            "I feel so blue and tired of everything.",
            "A deep sense of melancholia settled over the room.",
            "He spoke with a voice full of grief and despair.",
            "I am struggling to find any reason to be happy."
        ] * 2 + [
            "I feel so lonely and depressed lately.",
            "It is heartbreaking to see them leave.",
            "I am crying because of this terrible loss.",
            "Everything feels gloomy and hopeless today.",
            "I am deeply disappointed and sad."
        ],
        "surprise": [
            "Oh my god, I cannot believe my eyes!",
            "Wow! That was completely unexpected!",
            "What a shocking and amazing twist!",
            "I am astonished by this sudden event!",
            "This is an absolute shock to me!",
            "I never expected to see you here today!",
            "He suddenly jumped out of the box, startling me!",
            "The magician's trick left the audience amazed.",
            "I was speechless when they announced my name.",
            "What a surprise! I did not see that coming.",
            "She gasped in shock when she opened the letter.",
            "This sudden change of plans caught me off guard.",
            "I am totally stunned by this beautiful gift!",
            "We stared in disbelief at the sudden turn of events.",
            "It was an astonishing revelation that changed everything.",
            "I am absolutely amazed by this incredible performance!",
            "Who would have thought this could happen?",
            "He won the lottery, it was a mind-blowing shock.",
            "The sudden alarm startled everyone in the building.",
            "I am surprised by your sudden change of heart.",
            "What a bizarre and unexpected coincidence!",
            "The box was empty, which was quite a surprise.",
            "I stood frozen in astonishment at the news.",
            "She threw a surprise birthday party for him.",
            "The sudden thunderclaps startled the quiet neighborhood."
        ] * 2 + [
            "Oh my god, I cannot believe my eyes!",
            "Wow! That was completely unexpected!",
            "What a shocking and amazing twist!",
            "I am astonished by this sudden event!",
            "This is an absolute shock to me!"
        ]
    }

    texts, labels = [], []
    for class_name in CLASSES:
        c_idx = CLASSES.index(class_name)
        for phrase in raw_dataset[class_name]:
            texts.append(phrase)
            labels.append(c_idx)

    return texts, np.array(labels, dtype=np.int32)
