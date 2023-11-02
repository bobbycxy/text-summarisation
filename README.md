# text-summarisation

## 1. Introduction
Text summarisation has its applications in almost all the areas of the World Wide Web. We have search engines that provide queries and summaries as part of users' search experiences, news agencies providing summaries for articles and even e-commerce platforms using summaries to improve buyers' purchasing and window shopping experiences. With the number of opinions and events in the World Wide Web growing daily, there is ample opportunities for us to use text summarisations to extract key information out of text documents.

## 2. Basic Literature
In the field of text summarisation, the techniques used can be broadly classified into two categories - extraction and abstraction. Extraction techniques take out the important sentences or phrases from the original text and joins them to form a summary. This involves a ranking algorithm to assign scores to sentences or phrases based on a certain relevance to the overall meaning of the document. Abstraction techniques may or may not pick out the same sentences from the document. Instead, it generates new texts that capture the meaning of the original document through an analyses of the original document. 

To give an analogy, extractive techniques are like a highlighter, while abstractive techniques are like a pen. Both have their respective strengths and uses. For greater measure of accuracies, the extractive method is more suitable as it selects sentences from the original text. But, it may not be as concise and succinct as abstraction. On the other hand, abstraction techniques are more concise and succinct. However, they are more prone to inaccuracies and is reliant on how well the model has been trained on a labelled data set.

## 3. Problem Statement
Event extraction from a document is an important component of various tasks such as summarisation and clustering. We are interested to identify and extract the main event from a news article. We define 'main event' as the sentence that contains the most information and captures the main subject matter of the news article. 

The main event may or may not be the first sentence of the article and may consist of more than one sentence. Write a program to extract the main event from a news article. Note that the program should be able to take a text file of the provided format as input and output the main event sentences.

## 4. Methodologies
In this assignment, I'll use extraction and abstraction techniques to extract the main events from news articles. 

### 4.1. Method 1 - Extraction with Weighted Frequency-Based Approach
The first technique I’ll implement is an extraction method known as weighted frequencies. This is where each sentence of the provided text is ranked against other sentences based on word frequencies. 

The use of word frequency as a ranking method means the importance of words in a text are measured by how often words appear. As such, the sentence that comprises of the most frequently occurring words will be scored high. However, this summation would typically favour longer sentences. So, to normalise the results, I will use the average word frequency score of the sentence, instead of the total (i.e. the total score is divided by the number of words). 

How I wrote this function? The function begins by removing the stopwords in the text, then creating a python dictionary that counts the occurrence of each non-stopword. Then, it will proceed to take the average sum of all the word occurrences for each sentence. I have withheld the model from ranking sentences longer than K non-stopwords because they are likely to be explanatory sentences, than conclusive statements. Finally, we take the top N sentences ranked.

This is all implemented with the summarise_weight_freq function.

```summarise_weight_freq(text, n, max_sentence_length)```

This solution takes in 3 parameters. The first parameter is a processed text, the second parameter is an integer called N, and the third parameter K determines which sentences qualify to be scored by the word frequency algorithm. This function will rank the sentences that have K or lesser words, and take the top N highest scoring sentences. 

Applying this function to the 3 articles, I was able to derive the following:

| **Document**     | **Summary**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Article1.txt** | _'A second presale will be held for KrisFlyer members from 10am on Oct 30 to 9.59am on Oct 31.','They will then receive a unique access code from KrisFlyer via email on Oct 27.','UOB cardholders can enjoy a presale from 10am on Oct 27 till 9.59 am on Oct 29.'_|
| **Article2.txt** | _'Dozens of Palestinians have been killed in the West Bank in the latest flare-up of Israeli-Palestinian violence.','Oct 19 (Reuters) - Three Palestinians, including two teenagers, were killed by Israeli forces in separate incidents in the occupied West Bank early on Thursday, Palestinian official news agency WAFA said.',‘Israeli forces have carried out their fiercest bombardment of Gaza in response, killing more than 3,000 Palestinians and imposing a total siege on the blockaded enclave that Hamas controls, fuelling anger among Palestinians in the West Bank.'_ |
| **Article3.txt** | _'He was driving along Sophia Road towards Upper Wilkie Road shortly before 11.30pm when he spotted a police roadblock.','The prosecutor said: “The accused requested his front-seat passenger to swop seats with him, so that he would not be presented as the driver of the vehicle at the roadblock.','SINGAPORE – In an attempt to evade arrest, a doctor who drove a car after drinking beer tried to change seats with his passenger when he spotted a police roadblock.'_|

### 4.2. Method 2 - Extraction with Term Frequency-Inverse Document Frequency (TF-IDF)
The second technique I’ll implement is an extraction method that uses TF-IDF. TF-IDF typically works by measuring the number of times a word appears in the document but is counterbalanced by the number of documents in which it is present. Hence, words that are commonly used in all documents are not given a very high rank. However, a word that is present too many times in a few of the documents will be given a higher rank as it might be indicative of the context of the document. In our case, the documents are at the level of sentences. Hence, a word that is present too many times in a few of the sentences is given a higher rank.

How I wrote this function? The function begins by using Scikit-Learn’s TfidVectorizer to remove the stopwords. Then, it creates a TF-IDF matrix out of the remaining non-stopwords. Then, it does a cosine similarity calculation of the sentence against the entire text. Finally, it  takes the top N sentences ranked using the nlargest method from heapq. This is all implemented with the summarise_tfidf function.

```summarise_tfidf(text, n)```

This function takes in 2 parameters. The first parameter is still a text, and the second parameter is still an integer called N. This integer determines the top N number of sentences based on the TF-IDF ranking algorithm. Applying this function to the 3 articles, I was able to derive the following:

| **Document**     | **Summary**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Article1.txt** | _'UOB cardholders can enjoy a presale from 10am on Oct 27 till 9.59 am on Oct 29.','A second presale will be held for KrisFlyer members from 10am on Oct 30 to 9.59am on Oct 31.','They will then receive a unique access code from KrisFlyer via email on Oct 27.'_|
| **Article2.txt** | _'Oct 19 (Reuters) - Three Palestinians, including two teenagers, were killed by Israeli forces in separate incidents in the occupied West Bank early on Thursday, Palestinian official news agency WAFA said.','Dozens of Palestinians have been killed in the West Bank in the latest flare-up of Israeli-Palestinian violence.','Israel is preparing a ground assault in the Gaza Strip in response to a deadly attack by Palestinian militant group Hamas that killed at least 1,400 Israelis, mostly civilians, on Oct. 7.'_ |
| **Article3.txt** | _'SINGAPORE – In an attempt to evade arrest, a doctor who drove a car after drinking beer tried to change seats with his passenger when he spotted a police roadblock.','The passenger refused to do so and Nah Kwang Meng, who practises at Dr Nah & Lee Family Clinic in Woodlands, initially failed a breathalyser test after he stepped out of the vehicle.','Assistant Public Prosecutor Chye Jer Yuan told the court that before going behind the wheel on July 14, 2022, Nah had dinner and consumed about three to four glasses of beer.'_|

### 4.3. Method 3 - Abstraction with Pre-Trained Model with Fine-Tuning
The third technique I’ll implement is an abstraction method that uses transformer models. Specifically, we will apply transfer learning from pre-trained models that will be fine-tuned on a downstream task. Transfer learning, in the context of transformers, is helpful at improving model performances. Fortunately, there are available pre-trained transformers we can use. Notable examples include the Bi-Directional Encoder Representations from Transformers (BERT), and Text-to-Text Transfer Transformer (T5). For our problem set, we will use the small T5 model. It is an encoder-decoder model that's been pre-trained on multiple types of tasks. As a result, it works well on a variety of tasks. 

Before implementing our pre-trained T5 model, we fine-tune it with a supervised task of reading news articles as inputs and training them against their respective authored news summaries. Fortunately, there are available data sets that offer such news summarisation, e.g. CNN+Dailynews, and XSum. By fine tuning the model, we can ensure that the model is better trained for the given task of extracting a succinct summary from news articles. 

How I wrote it? Firstly, the pre-trained model needs to be trained on a related dataset. In our case, the dataset I used is the XSum which stands for 'Extreme Summarisation'. It is a dataset for evaluating single-document summarisation systems. Each article summary follows the question of 'what is the article about?'. It comprises 226,711 news articles accompanied with one-sentence summary, and they are collected from BBC (from 2010 to 2017) which cover a wide variety of genres such as general news, politics, sports, weather, business, technology, science, health, family, education, entertainment and arts. With a wide span of genre, it is the ideal dataset to use for our pre-trained models fine tuning exercise.

I began by tokenizing the text, followed by padding it, and loading in the ‘small-T5’ model weights and architecture. After which, I fine-tuned the model by training it over 3 epochs of the training data. This training took over ~19 hours on my personal AWS EC2 using the Deep Learning AMI  GPU TensorFlow 2.7.3 (Ubuntu 20.04), with an instance type: c5.2xlarge. The approximate cost was ~$14. For information on how I had set up my Deep Learning workstation with AWS, please read my medium article here. After training was complete, the evaluation metric of ROUGE-L gave the model a score of 0.20. 

Subsequently, I proceeded to save the model’s weights and tokenizer. And, I wrote a python function called summariser_t5 that needs only 2 parameters to reload the pretrained and fine-tuned T5 transformer model. 

```summariser_t5(model_filepath, tokenizer_filepath)```

The first parameter is the saved model’s weights folder path, and the second parameter is the saved model’s tokenizer folder path. The following text summaries were abstracted out of the articles’ content.

| **Document**     | **Summary**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Article1.txt** | _"Tickets for KrisFlyer and SIA Group's Category 1 to 4 concert tickets are to be redeemed at the end of the year."_|
| **Article2.txt** | _'Israeli forces have killed three Palestinians, including two teenagers, in separate incidents in the West Bank, a news agency has said.'_ |
| **Article3.txt** | _'A doctor who drove a car after drinking beer has been fined $4,000 after he pleaded guilty to attempting to pervert the course of justice.'_|

### 4.4. Method 4 - Abstraction with Transformer Model from Scratch
The fourth technique I’ll implement is an abstraction method that also uses transformer models. However, it is a newly trained model that follows the transformer architecture used in the “Attention Is All You Need” paper which was originally where the ‘transformer’ was proposed. Similar to Method 3, I will use the XSum dataset to train the transformer model on.

How I wrote it? I performed similar preprocessing steps on the text data. However, since this model was built from scratch, I had to first write helper functions – e.g. scaled dot product, masking and positional encoding – that were subsequently used to build each major block of the transformer architecture – e.g. multi head attention, encoder block and the decoder block, followed by training the model over 15 epochs that took 5.5 hours. 

Due to a limited size of my compute instance on AWS EC2, I was only able to design the transformer model to encode article lengths up to 500 words and decode up to 100 words for headers, with a transformer model that has 4 heads (for the multi head attention) and has 5 layers of the encoder and decoder blocks. Unfortunately, this created limitations for my model as it was unable to get all the needed inputs from an article text to perform the needed training on summary targets. As a result, there were disagreements between the abstracted text summaries and the original summaries.

I could increase the size of my compute instance and compare the result. However, that will be costly. Hence, due to cost concerns, I have decided against training my model further at this point, and have excluded it from being used as a method of extraction.






