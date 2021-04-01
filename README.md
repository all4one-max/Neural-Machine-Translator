# Neural Machine Translator

This web app convert french sentences to english sentences.

# Overview

-   The model has been built on top of keras Sequential api and uses LSTM for training.

-   One thing to note is that the words in the sentence has been vectorized, so similar words may have similar vectors, which is a better option than using bag of words or Tf-Idf, which do not take into account the semantics of sentences.

-   AS training a deep learning model on local system takes a lot of time, the model has been trained on google collab with GPU session.

# Built with

![Python](https://img.shields.io/badge/Python-3.8-blueviolet)
![Library](https://img.shields.io/badge/tensorflow-Library-blue)
![Framework](https://img.shields.io/badge/Framework-Flask-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS/JS-green)

# Note

In the repo you can see Procfile, it is there because I have had made an attempt to deploy the app on heroku. The maximum allowed slug size on heroku is 500 Mb and tensorflow 2.3.0 is itself around 300 Mb and model_colab.h5 is around 130 Mb, therefore i was unable to deploy it on heroku.
If someone figures out a way to deploy Deep learning model on heroku for free, do let me know. My final slug size was around 547 Mb.

# Demo

![GIF](./translator_gif.gif)
