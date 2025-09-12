# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pickle
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from keras.utils import custom_object_scope

# # Load the tokenizers and models (assuming they are in the same directory)
# with open("C:\\Users\\lenovo\\OneDrive\\Bureau\\ML_Projects\\Deep_Learning\\LSTM\\Patient_Educational_materials\\tokenizer_english.pkl", "rb") as f:
#     tokenizer_english = pickle.load(f)

# with open('C:\\Users\\lenovo\\OneDrive\\Bureau\ML_Projects\\Deep_Learning\\LSTM\\Patient_Educational_materials\\tokenizer_arabic.pkl', 'rb') as f:
#     tokenizer_arabic = pickle.load(f)

# encoder_model = tf.keras.models.load_model(
#     "C:\\Users\\lenovo\\OneDrive\\Bureau\\ML_Projects\\Deep_Learning\\LSTM\\Patient_Educational_materials\\encoder_model.keras"
# )

# # Load decoder model
# decoder_model = tf.keras.models.load_model(
#     "C:\\Users\\lenovo\\OneDrive\\Bureau\\ML_Projects\\Deep_Learning\\LSTM\\Patient_Educational_materials\\model_decoder.keras"
# )

# # Define max lengths (replace with your actual values from training)
# max_len_eng = 241  # Example value; adjust based on your training data
# max_len_ar = 219   # Example value; adjust based on your training data

# def decode_sequence(input_seq):
#     # Encode input sentence
#     states_value = encoder_model.predict(input_seq, verbose=0)
#     # Start with <start> token (assuming 'start' is in your Arabic tokenizer)
#     target_seq = np.array([[tokenizer_arabic.word_index.get('start', 1)]])  # Default to 1 if not found
#     decoded_sentence = []
#     for _ in range(max_len_ar):
#         output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
#         # Pick the word with max probability
#         sampled_token_index = np.argmax(output_tokens[0, -1, :])
#         sampled_word = tokenizer_arabic.index_word.get(sampled_token_index, '')
#         if sampled_word == 'end' or sampled_word == '':
#             break
#         decoded_sentence.append(sampled_word)
#         # Update target sequence and states
#         target_seq = np.array([[sampled_token_index]])
#         states_value = [h, c]
#     return ' '.join(decoded_sentence)

# # Flask app setup
# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend requests

# @app.route('/translate', methods=['POST'])
# def translate():
#     data = request.json
#     if not data or 'sentence' not in data:
#         return jsonify({'error': 'No sentence provided'}), 400
    
#     test_sentence = data['sentence']
#     # Preprocess: lowercase and remove non-alphabetic (basic cleanup)
#     test_sentence = test_sentence.lower()
#     test_sentence = ' '.join(test_sentence.split())
    
#     # Tokenize and pad
#     test_seq = tokenizer_english.texts_to_sequences([test_sentence])
#     test_seq = pad_sequences(test_seq, maxlen=max_len_eng, padding='post')
    
#     # Decode
#     translation = decode_sequence(test_seq)
    
#     return jsonify({'translation': translation})

# if __name__ == '__main__':
#     app.run(debug=True)



import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


with open(r"C:\Users\lenovo\OneDrive\Bureau\ML_Projects\Deep_Learning\LSTM\Patient_Educational_materials\tokenizer_english.pkl", "rb") as f:
    tokenizer_english = pickle.load(f)

with open(r"C:\Users\lenovo\OneDrive\Bureau\ML_Projects\Deep_Learning\LSTM\Patient_Educational_materials\tokenizer_arabic.pkl", "rb") as f:
    tokenizer_arabic = pickle.load(f)


encoder_model = tf.keras.models.load_model(
    r"C:\Users\lenovo\OneDrive\Bureau\ML_Projects\Deep_Learning\LSTM\Patient_Educational_materials\encoder_model.keras"
)

decoder_model = tf.keras.models.load_model(
    r"C:\Users\lenovo\OneDrive\Bureau\ML_Projects\Deep_Learning\LSTM\Patient_Educational_materials\model_decoder.keras"
)


max_len_eng = 241
max_len_ar = 219


def decode_sequence(input_seq):
    # Encode input sentence
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Start token
    start_token_index = tokenizer_arabic.word_index.get('start', 1)
    target_seq = np.array([[start_token_index]])
    decoded_sentence = []

    for _ in range(max_len_ar):
        # Predict next token
        output_tokens = decoder_model.predict([target_seq] + states_value, verbose=0)

        # If decoder outputs states separately
        if isinstance(output_tokens, list) and len(output_tokens) == 3:
            output_tokens_seq, h, c = output_tokens
            states_value = [h, c]
        else:
            output_tokens_seq = output_tokens

        sampled_token_index = np.argmax(output_tokens_seq[0, -1, :])
        sampled_word = tokenizer_arabic.index_word.get(sampled_token_index, '')

        if sampled_word in ['end', ''] or sampled_word is None:
            break

        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])

    return ' '.join(decoded_sentence)


app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    if not data or 'sentence' not in data:
        return jsonify({'error': 'No sentence provided'}), 400

    test_sentence = data['sentence'].lower().strip()
    test_seq = tokenizer_english.texts_to_sequences([test_sentence])
    test_seq = pad_sequences(test_seq, maxlen=max_len_eng, padding='post')

    translation = decode_sequence(test_seq)
    return jsonify({'translation': translation})


if __name__ == '__main__':
    app.run(debug=True)
