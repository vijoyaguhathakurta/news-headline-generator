# News Headline Generator

**BSc Final Year Project – Computer Science**

This project implements a **News Headline Generator** using **Natural Language Processing (NLP)** and a **Transformer-based sequence-to-sequence model**.
Given a news article, the system generates a concise and relevant headline.



## Project Motivation

Manually writing headlines for large volumes of news articles is time-consuming.
This project explores how **deep learning and NLP** can be used to automatically generate meaningful headlines from article text.

The focus of this project is **conceptual understanding and academic learning**, rather than production-level deployment.



## Approach & Model

* Problem Type: **Text Summarization (Abstractive)**
* Model Architecture: **Transformer (Sequence-to-Sequence)**
* Learning Type: **Supervised Learning**
* Input: Full news article text
* Output: Generated headline

The model is trained on article–headline pairs to learn semantic relationships between long-form text and short summaries.



## Technologies Used

* Python
* Natural Language Processing (NLP)
* Transformer Models
* Flask (for web interface)
* Jupyter Notebook


## Workflow

1. News articles are preprocessed and tokenized
2. Transformer model is trained on article–headline pairs
3. Tokenizer and trained model is saved for inference
4. Flask app loads the tokenizer and model logic
5. User inputs article text and receives generated headline



## Dataset

* The original public dataset contained a large number of news articles.
* **Only a small sample (first 100 rows)** is included in this repository due to GitHub file size limits.
* The dataset is used strictly for **academic and learning purposes**.


## Limitations & Future Work

* Model trained on a limited dataset
* Performance can improve with larger datasets
* Can be extended using pre-trained models like **BERT**, **T5**, or **BART**
* Deployment scalability and optimization not addressed



## Academic Note

This project was developed as part of a **BSc Computer Science final-year project**.
The emphasis is on understanding **NLP concepts, Transformer models, and end-to-end ML workflow**.


## Privacy Notice

All personal, institutional, and academic identifiers have been **intentionally excluded** from this repository.
