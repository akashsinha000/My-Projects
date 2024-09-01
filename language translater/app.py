from flask import Flask, render_template, request, send_file
from transformers import MarianMTModel, MarianTokenizer
import logging
from gtts import gTTS
import base64
from io import BytesIO

app = Flask(__name__)

model_name_hi = "Helsinki-NLP/opus-mt-en-hi"
model_hi = MarianMTModel.from_pretrained(model_name_hi)
tokenizer_hi = MarianTokenizer.from_pretrained(model_name_hi)

logging.basicConfig(level=logging.INFO)

def translate_to_hindi(text):
    if tokenizer_hi and model_hi:
        try:
            inputs = tokenizer_hi(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model_hi.generate(**inputs)
            translated_text = tokenizer_hi.decode(outputs[0], skip_special_tokens=True)
            return translated_text
        except Exception as e:
            logging.error(f"Error occurred during translation to Hindi: {str(e)}")
    return "Translation to Hindi failed. Please try again later."

def generate_audio(text):
    tts = gTTS(text=text, lang='hi')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    audio_data_base64 = base64.b64encode(audio_data.read()).decode('utf-8')
    return audio_data_base64

@app.route('/')
def home():
    return render_template('index.htm')

@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == 'POST':
        input_text = request.form['input_text']

        translated_text_hi = translate_to_hindi(input_text)
        
        
        audio_data = generate_audio(translated_text_hi)

        return render_template('index.htm', input_text=input_text, translated_text_hi=translated_text_hi, audio_data=audio_data)

if __name__ == '__main__':
    app.run(debug=True, port=8000)  
