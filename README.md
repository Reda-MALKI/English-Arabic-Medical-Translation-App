English → Arabic Medical Translation App 🧠💉
I recently built a deep learning-based translation model specifically for medical and educational sentences from English to Arabic.

💡 What I built:
A seq2seq LSTM model with an encoder-decoder architecture:
Encoder: Processes the English sentence and encodes it into a fixed-length context vector.
Decoder: Generates the Arabic translation word by word using the context vector and previously generated words.
Preprocessed datasets using tokenizers for English and Arabic, handling special <start> and <end> tokens.

A Flask web application where users can input an English sentence and receive its Arabic translation in real time.

⚡ Technical highlights:
Embedding layers: Both English and Arabic words are converted into dense vector representations, allowing the model to capture semantic relationships between words.
Encoder LSTM: Processes the embedded English sentence and encodes it into a fixed-length context vector that summarizes the sentence meaning.
Decoder LSTM: Takes the context vector and predicts each Arabic word sequentially, conditioning on the previous words.
Dense layer (output layer): Applies a softmax activation to the decoder LSTM outputs to produce probabilities over the Arabic vocabulary in the dataset selecting the most likely next word.
The system uses teacher forcing during training and greedy decoding for inference.

🛠 Challenges & Learnings:
The model performs well for short and medium-length sentences, but struggles with long sentences due to LSTM limitations like vanishing gradients and fixed-length context vectors.
This highlighted the importance of exploring attention mechanisms or transformers in future iterations for better long-sequence translation.

📈 Next Steps:
Incorporate attention layers or switch to a Transformer-based architecture for more accurate and fluent translations when required hardware is available.
Expand the dataset to cover more medical terminology and sentence variatio


Dataset , tokenizers , models and a picture of model architecture are also provided in this repository for later use if needed .
