Project Report on Text Summarization Techniques

Introduction

With increasingly busy human lifestyles, staying informed quickly and efficiently has become crucial. Automated text summarization techniques address this by generating concise summaries from lengthy text sources. This report focuses on two prominent methods: Extractive and Abstractive Text Summarization.

Text Summarization Techniques

1. Extractive Text Summarization

Overview:

Identifies the most significant sentences from the original text and combines them to create a concise summary.

Utilizes sentence-ranking algorithms.

Advantages:

Time and resource-efficient, suitable for CPU environments.

Easier to implement and interpret.

Less data-intensive.

Disadvantages:

The summary consists of verbatim sentences from the source text, which may lack smooth narrative flow.

Potential redundancy or less coherence compared to human-written summaries.

2. Abstractive Text Summarization

Overview:

Generates summaries containing new words and phrases different from the source material.

Requires context understanding and complex language modeling.

Typically relies on deep learning techniques such as transformer models.

Advantages:

Creates fluent, natural, and contextually coherent summaries.

Captures the essence of the original text more accurately and fluently.

Disadvantages:

Computationally intensive; requires GPUs or TPUs.

Resource and time-consuming, especially during training.

Performance Benchmarks and Evaluation

Dataset Used: CNN/Dailymail dataset comprising 287K training articles, 13K validation, and 11K test articles.

Evaluation Metrics: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum).

Results:

Extractive Model Performance:

ROUGE-1: 28.91

ROUGE-2: 12.04

ROUGE-L: 17.76

SOTA T5 Abstractive Summarization Performance:

ROUGE-1: 43.52

ROUGE-2: 21.55

ROUGE-L: 40.69

ROUGE-Lsum: 32.71

The results indicate that abstractive summarization models, particularly the T5 transformer model fine-tuned using Huggingface, significantly outperform extractive summarization in terms of ROUGE scores, suggesting better overall fluency and contextual understanding.

Comprehensive Documentation

Data Curation:

Detailed exploration and cleaning of CNN/Dailymail dataset.

Median article length: ~700 words; summary length: ~60 words.

Model Fine‑tuning and Optimization:

T5 Transformer from Huggingface was fine-tuned.

Data tokenized using T5 tokenizer and trained using Huggingface Trainer API.

Evaluation Results:

ROUGE evaluation for extractive and abstractive summarization clearly shows abstractive methods outperforming extractive summarization in accuracy but are computationally more demanding.

Conclusion

The project successfully demonstrates the advantages and limitations of extractive and abstractive summarization methods. Abstractive summarization, despite its resource-intensive nature, outperforms extractive summarization significantly in terms of quality and readability.

Practical Implementation

Streamlit Application:

Simple, user-friendly interface.

Users can select preferred summary length and paste article content.

Automatically generates both extractive and abstractive summaries.

Recommendations for Future Enhancements

Enhanced Data Accuracy: Improve summaries by training models on larger datasets.

Newsletter Feature: Weekly digest based on summaries for user convenience.

User Customization: Personalized summaries based on user preference and engagement history.

<img width="839" alt="image" src="https://github.com/user-attachments/assets/0b96bed9-bd5b-45b6-b93b-1b6fc1ff07c3" />

The data is divided into training , validation and tests sets. Each article has a human generated summary associated with it
Training set: 287K articles
● Validation set: 13K articles
● Test set: 11K articles
Fields:
● ID : Unique ID of article
● Article: News article body
● Highlights: News article summary

Distribution of Article and Summary Length
Doing analysis of the article and summary length will help in deciding model
parameters. For this purpose, the distribution of article and summary length can be
examined.

<img width="840" alt="image" src="https://github.com/user-attachments/assets/dd343a0f-482f-40b3-848b-90adf7e50397" />
From the length distributions we can see that articles have a median length of 700
words and summaries have a median length of 60.
Both the distributions are right skewed, indicating that most of the articles have length
less than or equal to 700, with a few articles having length greater than 1600 words.

Box Plot of Summary and Article Length
Summary lengths and article lengths both have a few outliers, indicating few articles
have longer length and corresponding longer summaries. This can be verified with the
below box plots. Both indicate that outliers are towards the higher word length side.

<img width="838" alt="image" src="https://github.com/user-attachments/assets/620af9ef-51bb-4c59-a5c2-fe7ea7bbcfc8" />

Word cloud of Summaries and Articles
A word cloud helps in analyzing the most important topics and keywords that are
covered in a text. Generating word clouds of articles and summaries will help in seeing
the keywords of the data and seeing patterns.
Following word clouds are plotted from a sample of news and highlights (100k). Both
word clouds show similar words like year, police, oﬃcial, say, new etc, so both the news
articles and the summaries cover similar topics. They are also the topics that mostly
appear in any news articles.

<img width="800" alt="image" src="https://github.com/user-attachments/assets/fc5f9331-ab7a-4980-95a8-ab6b4caf582a" />

Article Word Cloud

<img width="834" alt="image" src="https://github.com/user-attachments/assets/e9ff6839-309e-4463-b0a4-3f830184a18b" />

Summary Word Cloud

Methods Used
Text summarization can be done using two ways: Extractive and Abstractive
summarization.

<img width="815" alt="image" src="https://github.com/user-attachments/assets/72c9bbe5-c643-48c6-b5a8-47b944295747" />

In extractive summarization, we identify important sentences from the article and make
a summary by selecting the most important sentences.
Whereas, for abstractive summarization the model understands the context and
generates a summary with the important points with new phrases and language.
Abstractive summarization is more similar to the way a human summarizes any
content. A person might read the entire document, remember a few key points and while
writing the summary, will make new sentences that include these points. Abstractive
summarization follows the same concept.
Evaluation Metric
Since the target in summarization is a text summary, it becomes complex to measure
model performance. The widely used metric for text summarization is ROUGE (Recall
Oriented Understudy for Gisting Evaluation). It generates a precision, recall and f1 score
of the overlap between actual summary and generated summary.
ROUGE makes use of n-grams for calculating the overlap. As such, this gives rise to
different ROUGE metrics like ROUGE-1, ROUGE-2, which consider 1-gram (basically word
overlap) and bigrams. Another ROUGE metric is the ROUGE-L, which considers the
Longest Common Subsequence (LCS) of matching words. (Code Snippets in Appendix)
Following images explain the ROUGE metric with an example of “I really loved reading th
Hunger Games” as the generated summary and “I loved reading the Hunger Games” as
a reference summary.
This has a ROUGE-1 recall of 100%, ROUGE-1 precision 85%, ROUGE-1 F1 Score 91%
ROUGE-L recall 100% and ROUGE-L precision 85% as explained below.

<img width="575" alt="image" src="https://github.com/user-attachments/assets/173917e1-af60-489c-920c-ade8e9dcdc47" />

This project evaluates extractive and abstractive summarization on the basis of
different ROUGE scores achieved.
Following table states the SOTA T5 metrics on the CNN/Dailymail Dataset

<img width="574" alt="image" src="https://github.com/user-attachments/assets/e2e25294-136e-4580-b4b6-e7b37af9fbfb" />

Extractive Summarization Technique
One of the techniques of extractive summarization is text ranking. In this method, each
sentence of the document is assigned a score based on its importance. The top K
sentences from these are selected as a summary.
Gensim provides a summarizer that performs such summarization. But, for the purpose
of this project, I have implemented the extractive summarizer from scratch using python
in a News Summarizer class (Code Snippets in Appendix).
Algorithm
The idea is that sentences that have words that are more frequently used throughout
the article are more important. This works because we remove stopwords. For example,
if an article talks about NASA, it will have many references to space, stars etc, thus
these sentences will be more important.

<img width="587" alt="image" src="https://github.com/user-attachments/assets/0df603d2-6228-4e21-915d-7206abdc7f7a" />

The steps followed for extractive summarization are as follows
● Text Preprocessing (Code Snippets in Appendix)
○ Convert to lowercase
○ Remove punctuation
○ Remove stopwords
● Tokenization:
○ Tokenize into sentence and words
● Word Frequency:
○ Calculate word frequency for all words
○ Normalized using maximum frequency : Normalization helps to reduce the
effect of article length
● Sentence Tokenization and Ranking:
○ Rank each sentence by adding word frequencies
● Summary:
○ Select Top K Sentences by Rank. K can be a parameter that depends on
the user
Advantages of Extractive Summarization
● Time Eﬃcient : It is very quick to perform this summarization as it does not
require any complex calculations and operations
● Resource Eﬃcient: A normal CPU works just fine for this type of summarization.
We do not need heavy compute machines like GPU
● Easier to interpret summarization process
● Does not require large amount of data
Disadvantages of Extractive Summarization
● The summary generated is basically important sentences from the article. This
might not cover the entire context of the article and might miss important points
that are covered in discarded sentences
● Summary might look disconnected and less fluent as it is made of different
sentences
ROUGE Performance on Test Set
<img width="586" alt="image" src="https://github.com/user-attachments/assets/38fea011-874b-4b12-a24e-7eee01e1b90a" />

<img width="1401" alt="image" src="https://github.com/user-attachments/assets/8969c06a-90bd-4483-8723-b8410f95723b" />
<img width="1363" alt="image" src="https://github.com/user-attachments/assets/3f1d3725-7542-4640-ab15-6855d3ffbf98" />
<img width="1376" alt="image" src="https://github.com/user-attachments/assets/850c349b-5f17-475f-b6cc-583e37d2c8eb" />

1. Training Metrics
Mean Token Accuracy: This shows accuracy at each training step. The graph fluctuates initially but settles around 48%, indicating moderate accuracy improvement during training.
Loss: The loss plot shows significant fluctuations, indicating unstable model learning. It initially rises, then drops sharply before rising again. Ideally, this should consistently decrease.
Gradient Norm (grad_norm): Reflects the magnitude of gradients during training. A sharp peak followed by a decline suggests initial instability that stabilizes gradually.
2. GPU Performance Metrics
GPU Memory Errors: Both corrected and uncorrected GPU memory errors remain at zero, indicating no memory issues.
GPU Memory Clock Speed: Stable at 6250 MHz, showing consistent GPU memory performance.
GPU SM Clock Speed: Increases over time, suggesting rising GPU activity as training progresses.
GPU Power Usage: Initially stable, then sharply decreases, which could suggest less intense computation or completion of training towards the end.
GPU Utilization: Stays consistently high (~100%), indicating optimal GPU utilization during model training.
3. GPU Resource Allocation
GPU Enforced Power Limit: Constant at 72W, indicating a fixed power limitation for GPU usage.
GPU Memory Allocated: Quickly rises and stabilizes near full allocation (about 94%), indicating high memory usage.
GPU Temperature: Spikes initially (~80°C), then stabilizes around 75°C, indicating efficient thermal management under high load.
GPU Time Accessing Memory: Initially high, declining sharply towards the end, reflecting reduced memory access possibly as training nears completion.
Overall, the training metrics indicate a somewhat unstable training run with moderate accuracy gains, while GPU metrics suggest efficient and optimal hardware utilization with no errors or issues.












