from flask import Flask, render_template, request
import pickle, string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates")

# model = load_model("model_colab.h5")
with open("french_tokenizer.pickle", "rb") as handle:
    french_tokenizer_rec = pickle.load(handle)
with open("english_tokenizer.pickle", "rb") as handle:
    eng_tokenizer_rec = pickle.load(handle)


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding="post")
    return seq


def get_english_sentence(pred_french_sentence):
    temp = []
    for j in range(len(pred_french_sentence)):
        t = get_word(pred_french_sentence[j], eng_tokenizer_rec)
        if j > 0:  # If it is not the first word
            if (t == get_word(pred_french_sentence[j - 1], eng_tokenizer_rec)) or (
                t == None
            ):  # if the next word is same as the previous
                temp.append("")
            else:
                temp.append(t)

        else:  # if it's not the first word
            if t == None:  # if we didn't get a valid code from dictionary
                temp.append("")
            else:
                temp.append(t)
    return " ".join(temp)


fra_length = 14


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    french_sentence = request.form["french_text"]
    french_sentence = french_sentence.translate(
        str.maketrans("", "", string.punctuation)
    )
    french_sentence = french_sentence.lower()
    encode_french_sentence = encode_sequences(
        french_tokenizer_rec, fra_length, [french_sentence]
    )
    pred_french_sentence = model.predict_classes(
        encode_french_sentence.reshape(
            (encode_french_sentence.shape[0], encode_french_sentence.shape[1])
        )
    )
    eng_sentence = get_english_sentence(pred_french_sentence[0])
    return render_template("predict.html", predicted_eng_sentence=eng_sentence)


if __name__ == "__main__":
    app.run(debug=True)
