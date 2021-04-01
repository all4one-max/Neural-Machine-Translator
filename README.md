# Neural Machine Translator

This web app convert french sentences to english sentences.

# Built with

![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Library](https://img.shields.io/badge/Library-keras-red)
![Library](https://img.shields.io/badge/Library-tensorflow-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-blueviolet)

# Overview

-   The model has been built on top of keras Sequential api and uses LSTM for training.

-   One thing to note is that the words in the sentence has been vectorized, so similar words may have similar vectors, which is a better option than using bag of words or Tf-Idf, which do not take into account the semantics of sentences.

-   A training a deep learning model on local system takes a lot of time, the model has been trained on google collab with GPU session.

-   Finally, to make the web app Flask has been used in the Backend and HTML, CSS and Bootstrap on the frontend.

## Directory Tree

```
├── static
│    ├── css
|    |   ├── bootstrap-grid.min.css
|    |   ├── bootstrap.min.css
|    |   ├── style.css
|    ├── js
|    |   ├── bootstap.min.js
|    ├── english.jpg
|    ├── french.jpg
├── template
|    |── home.html
│    ├── predict.html
├── app.py
├── english_tokenizer.pickle
├── french_tokenizer.pickle
├── model_colab.h5
├── model.py
├── Procfile
├── translator_gif.gif
├── README.md
├── requirements.txt
├── .gitattributes
```

# Note

In the repo you can see Procfile, it is there because I have had made an attempt to deploy the app on heroku. The maximum allowed slug size on heroku is 500 Mb and tensorflow 2.3.0 is itself around 300 Mb and model_colab.h5 is around 130 Mb, therefore i was unable to deploy it on heroku.
If someone figures out a way to deploy Deep learning model on heroku for free, do let me know. My final slug size was around 547 Mb.

# Demo

![GIF](./translator_gif.gif)

## Future Scope

-   Deploy the web app on heroku(for free Obviously)
-   The Bleu score of the model is aroung 66.7% so try to optimize the model
-   Optimize Flask app.py
-   Front-End
