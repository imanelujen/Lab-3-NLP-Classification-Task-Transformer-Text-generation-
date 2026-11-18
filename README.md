# Lab-3-NLP-Classification-Task-Transformer-Text-generation-

The main purpose of this lab is to become familiar with deep learning techniques for Natural Language Processing (NLP), using sequence models and Transformer architectures for text classification and text generation.

## part 1:Train Models Using RNN Architectures

The dataset contains Arabic paragraphs related to environmental topics, classified into three categories:

- **Class 0:** Non pertinent
- **Class 1:** General environment
- **Class 2:** Pollution

Four recurrent neural network architectures were implemented and evaluated: SimpleRNN, Bidirectional RNN, GRU, and LSTM.

### 1. SimpleRNN

The SimpleRNN model achieved **50% test accuracy**. It performed better on the "Non pertinent" class but failed to identify the "General environment" class.

This experiment showed that a basic recurrent network can learn sequential patterns, but its performance is limited by the small dataset and class imbalance.

### 2. Bidirectional RNN

The Bidirectional RNN also achieved **50% test accuracy**. It processes text in both forward and backward directions, allowing it to capture contextual information from both sides of a sequence.

However, it did not significantly improve the overall classification performance.

### 3. GRU (Gated Recurrent Unit)

The GRU achieved the best test accuracy among the four models, with **70.83%**.

Its update and reset gates help control information flow and preserve relevant context throughout the sequence. It performed particularly well on the "Non pertinent" and "Pollution" classes.

### 4. LSTM (Long Short-Term Memory)

The LSTM achieved **66.67% test accuracy** and the highest validation accuracy, reaching **78.95%**.

Its memory cell and three gates (forget, input, and output) help it learn long-term dependencies. Unlike the other models, it was able to identify some examples from the "General environment" class.

### Model Comparison

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| SimpleRNN | 68.42% | 50.00% |
| Bidirectional RNN | 52.63% | 50.00% |
| GRU | 73.68% | **70.83%** |
| LSTM | **78.95%** | 66.67% |

**Best test result:** GRU, with 70.83% accuracy.

These results should be interpreted carefully because the dataset is small and imbalanced, which limits the models' generalization ability.

## What I Learned

Through this part of the lab, I learned how to:

- Prepare and preprocess Arabic text data.
- Perform text normalization, tokenization, stop-word removal, stemming, and lemmatization.
- Transform textual data into numerical sequences using a Keras Tokenizer and padding.
- Split data into training, validation, and testing sets.
- Build and train recurrent neural network architectures using Embedding, SimpleRNN, Bidirectional, GRU, and LSTM layers.
- Understand how recurrent networks process sequential information and how gates help manage information flow.
- Apply regularization techniques such as Dropout and EarlyStopping to reduce overfitting.
- Evaluate classification models using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- Compare different architectures and analyze their strengths and limitations.

**Conclusion:** This experiment demonstrated that GRU and LSTM architectures can capture sequential information more effectively than a basic RNN in some cases. However, model performance also depends heavily on dataset size, quality, and class distribution.

## part 2:Fine-tuning GPT-2 on Generatin a new paragraph according to a given sentence
