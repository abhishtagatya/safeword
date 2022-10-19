import json
import random
from typing import List
import logging

import numpy as np
import pandas as pd

import whisper

import streamlit as st

from model.ser_model import SER

model = whisper.load_model("base")
ser_model = SER.load('pretrained/cnn_attention_lstm_model.pt')

with open('safewords.json', 'r') as swfile:
    swfile_json = json.load(swfile)


def detect_safe_words(full_text: str, safe_words: List[str]):
    tokens = full_text.lower().split()

    flag = False
    for token in tokens:
        for word in safe_words:
            if token == word.lower():
                flag = True

    if flag:
        st.session_state.on_alert = True
        return
    st.session_state.on_alert = False


def detect_emotion(model, audio_file, match_emotion):
    if st.session_state.on_alert:
        return

    prediction_bool = model.match_prediction(audio_file, match_emotion)

    if prediction_bool:
        st.session_state.on_alert = True
        return
    st.session_state.on_alert = False


def spacing(units=1):
    for _ in range(units):
        st.write("")


st.set_page_config(layout="wide", page_icon="🚨", page_title="SafeWord - Powered by OpenAI Whisper")

if "working_text" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.working_text = ""

if "on_alert" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.on_alert = False

if "audio_file_uploaded" not in st.session_state:
    st.session_state.audio_file_uploaded = False

st.title("🚨 SafeWord")

st.markdown("""
        Utilizing OpenAI's Whisper model and a CNN-Based Speech Emotion Recognition (SER)
        model to determine whether to call the authorities based on sentiment. 
        """)
spacing(1)

safeword_choices = st.multiselect(
    "Choose a Rarely Used Word",
    swfile_json['wordlist'],
    swfile_json['wordlist'][0:2]
)

uploaded_file = st.file_uploader("Put Audio File", type=['mp3', 'm4a', 'wav'])

if uploaded_file is not None:
    logging.info('Uploaded File : ' + uploaded_file.name)

    decoding_progress = 0
    decoding_progress_bar = st.progress(decoding_progress)

    audio_file = uploaded_file.read()

    with open('temp.mp3', 'wb') as temp_audio:
        temp_audio.write(audio_file)

    decoding_progress += 10
    decoding_progress_bar.progress(decoding_progress)

    whisper_audio = whisper.load_audio('temp.mp3')
    whisper_audio = whisper.pad_or_trim(whisper_audio)

    decoding_progress += 20
    decoding_progress_bar.progress(decoding_progress)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(whisper_audio).to(model.device)

    decoding_progress += 30
    decoding_progress_bar.progress(decoding_progress)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    logging.info(f'Whisper Language : {max(probs, key=probs.get)}')

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    decoding_progress += 40
    decoding_progress_bar.progress(decoding_progress)

    logging.info('Whisper Text : ' + result.text)
    detect_safe_words(full_text=result.text, safe_words=safeword_choices)
    detect_emotion(ser_model, 'temp.mp3', [5, 6, 7, 0])

    st.audio(audio_file, format='audio/ogg')
    decoding_progress_bar.empty()

if st.session_state.on_alert:
    st.error("The Police has been Alerted, Please Stay Safe until they Arrive", icon="🚨")
