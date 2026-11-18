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


## Part 2: Fine-Tuning GPT-2 for Arabic Text Generation

### 1. Objective

The objective of this part is to fine-tune a pretrained Arabic GPT-2 model to generate a new paragraph based on a given input sentence.

Unlike Part 1, which focuses on text classification, this task is a **text generation task**. The model learns to predict the next token based on the preceding tokens, allowing it to generate a continuation of the input sentence.

### 2. Dataset Preparation

We reused the Arabic environmental texts collected in Part 1.

- **Total paragraphs:** 117
- **Input feature:** Original Arabic text
- **Preprocessing:** AraBERT preprocessing
- **Missing or empty paragraphs:** Removed
- **Training set:** 105 paragraphs (90%)
- **Validation set:** 12 paragraphs (10%)

Only the original Arabic paragraphs were used. Classification labels, stems, and lemmas were not included because GPT-2 needs natural text to learn language patterns and generate meaningful continuations.

### 3. Loading the Pretrained GPT-2 Model

We selected the pretrained Arabic GPT-2 model:

`aubmindlab/aragpt2-base`

The model was loaded using the Hugging Face Transformers library.

**Model characteristics:**

- Architecture: GPT-2 (Transformer decoder)
- Number of parameters: Approximately 135 million
- Vocabulary size after adding the padding token: 64,001
- Execution device: GPU (CUDA)

A padding token was added to handle sequences of different lengths during batch processing.

### 4. Tokenization and Dataset Encoding

Before training, the Arabic paragraphs were converted into token IDs using the model's tokenizer.

The preprocessing and tokenization steps were:

1. Apply AraBERT preprocessing to each paragraph.
2. Convert the text into token IDs.
3. Truncate sequences to a maximum of 255 tokens.
4. Append an End-of-Sequence (EOS) token to each paragraph.

We used `DataCollatorForLanguageModeling` with `mlm=False`, because GPT-2 uses **Causal Language Modeling (CLM)**. This means that the model learns to predict the next token using the preceding context, rather than predicting masked words.

### 5. Fine-Tuning Configuration

We fine-tuned the pretrained model using the Hugging Face `Trainer` API.

| Parameter | Value |
|---|---:|
| Number of epochs | 3 |
| Training batch size | 2 |
| Evaluation batch size | 2 |
| Gradient accumulation steps | 4 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Maximum sequence length | 256 |
| Precision | FP16 |
| Evaluation strategy | After each epoch |
| Best model selection | Lowest validation loss |

We enabled `load_best_model_at_end=True` to automatically retain the checkpoint with the lowest validation loss.

### 6. Training Results

The following table presents the training and validation losses obtained during fine-tuning.

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 8.7011 | **3.8676** |
| 2 | 7.4467 | 3.8949 |
| 3 | 6.4467 | 3.8977 |

**Observations:**

- Training loss decreased throughout the three epochs, indicating that the model was learning from the training data.
- Validation loss was lowest after the first epoch.
- The slight increase in validation loss during subsequent epochs suggests possible overfitting.

Therefore, the best checkpoint was selected from **Epoch 1**, as it achieved the lowest validation loss.

### 7. Text Generation

After fine-tuning, we tested the model by providing Arabic sentences as input prompts.

The generation process used the following parameters:

| Parameter | Value | Purpose |
|---|---:|---|
| `max_new_tokens` | 150 | Maximum number of generated tokens |
| `temperature` | 0.8 | Controls randomness |
| `top_k` | 50 | Limits token selection to the top 50 candidates |
| `top_p` | 0.9 | Uses nucleus sampling |
| `repetition_penalty` | 1.2 | Discourages repetitive text |
| `no_repeat_ngram_size` | 3 | Prevents repeated sequences of three tokens |



The model  generated longer continuations for other prompts, sometimes producing multiple sentences related to environmental topics.

### 8. Evaluation and Discussion

The generated results demonstrate that the fine-tuned model can:

- Continue an Arabic sentence based on its context.
- Generate text related to environmental topics.
- Produce grammatically plausible Arabic expressions.
- Reuse linguistic patterns learned from the training dataset.

However, some limitations were observed:

- Some generated paragraphs may shift to a different topic.
- Some statements may contain unsupported or inaccurate information.
- The small dataset limits the model's ability to generalize.
- The model may reproduce linguistic patterns from its training examples.

These limitations are expected given the small size of the dataset (117 paragraphs). Therefore, generated text should be considered a demonstration of language generation rather than a source of verified environmental information.

### 9. Conclusion

In this part, we successfully fine-tuned the pretrained Arabic GPT-2 model on a custom environmental dataset containing 117 paragraphs.

The model learned to generate Arabic text based on a given input sentence. Although the dataset size and possible overfitting limit its generalization capabilities, the results demonstrate how transfer learning can adapt a pretrained language model to a specific domain.

This experiment also illustrates the difference between the two tasks explored in this project:

- **Part 1 – Text Classification:** The model receives a paragraph and predicts its environmental category.
- **Part 2 – Text Generation:** The model receives a sentence and predicts subsequent tokens to generate a continuation.
