
# Predicting Irregular Sleeping Patterns from Tweets

## Overview
Predict whether a user is suffering from ISP based on their social media interactions on Twitter. We identify users' ISP from their psycholinguistic characteristics and social media word usage patterns.

 - **Domain**:  Data Science, Deep Learning, Natural Language Processing, Mental Health, Psycholinguistic Patterns, Social Media.
 - **Tools used**: Twitter API - Tweepy, LIWC2015, IBM SPSS Statistical Software.

## Authors

- **Corresponding author:** [Dr. Md. Saddam Hossain Mukta](https://cse.uiu.ac.bd/profiles/saddam/), Associate Professor and Undergraduate Coordinator, United International University (UIU), Bangladesh
- **Other authors:** [Mohammed Jawwadul Islam](https://www.linkedin.com/in/jawwadfida/), [Mohammad Fahad Al Rafi](https://www.linkedin.com/in/md-fahad-al-al-rafi-14b968111/), Nafisa Akhter, [Moumy Kabir](https://www.linkedin.com/in/pranto-podder-b78b97162/), [Pranto Podder](https://www.linkedin.com/in/aysha-siddika-577ba5224/), [Aysha Siddika](https://www.linkedin.com/in/moumy-kabir-156a0a232/), and [Dr. Mohammed Eunus Ali](https://cse.buet.ac.bd/faculty_list/detail/eunus), Professor, Bangladesh University of Engineering and Technology (BUET), Bangladesh. 

## Data Availability

When collecting data from Twitter for our experiments, we followed Twitter's rules and regulations for data privacy. We collect tweets after getting permission from Twitter. According to [Twitter's policy](https://developer.twitter.com/en/developer-terms/policy), the tweets collected are their public property, which should only be exposed partially publicly with their consent, a consent that is granted with the Twitter API. Therefore, we are sharing a modified form containing relevant tweets to understand users' psychological behavior.


## Overall steps for dataset creation. 

<img src="https://user-images.githubusercontent.com/64092765/199010521-7c2b2123-62db-4308-a6a6-bfb65931b9b6.png" width="75%">

## Data Collection
We used Twitter to collect 924 tweets from certain nations or continents, such as the United States, the United Kingdom, Canada, Australia, New Zealand, Bangladesh, and some countries in Africa. We looked at the times when their tweets were tweeted. We used random sampling on the user based on their tweeting time and word choices and labeled their file with "Irregular Sleeping Pattern Yes" and "Irregular Sleeping Pattern No" labels. For Irregular Sleeping Pattern Yes, a time interval of 1 am to 5 am, which indicated that these users suffered from irregular sleeping patterns. For Regular Sleeping Pattern users, we randomly selected users who did not tweet at night, in the time range from 1 am to 5 am. 

## Filtering tweets based on time

For time filtering, we first divide the users (Irregular and Regular Sleeping Pattern users) based on their geographical location. Then we convert the UTC to standard geographical UTC based on the user’s location. For example, for the USA, we have to subtract 4 hours from Bangladesh standard UTC to convert it into (UTC-4) standard geographical UTC. For this conversion, we needed to separate the user CSV files into six sections based on areas: USA, Europe, Asia (Bangladesh, India, Pakistan, Korea, Japan), New Zealand, Australia, and South Africa. For Irregular Sleep Users time range is from 1 am - 6 am (excluding). Regular sleep users are outside this time range.


## Data Preprocessing

For raw data pre-processing, we discarded the username and mentions. We translated tweets that were in other languages to English using langdetect and googletrans. We kept the users' retweets because retweeting shows the various personality traits of the users. We removed hashtags and converted them into text. We also removed URLs and HTTP links. We also converted contractions, e.g., "they're" to "they are." We tokenize the sentences and removed punctuation using the ntlk library. Finally, we removed emojis by using the Python demoji package.

## Filtering Tweets using words

After filtering our dataset based on time, we applied another filtering technique to find tweets that are relevant to irregular sleeping and regular sleeping patterns for our final dataset. The filtering technique is based on users’ way of using certain words as we mentioned in the data collection section.

### Finding out LIWC correlated categories

Linguistic Inquiry and Word Count (LIWC) is a text analysis program that counts words in psychologically significant categories. LIWC enables users to examine the inner workings of literary works. When a user uploads a text to LIWC, he or she will receive an output with more than 70 columns of data. Here, we have analyzed users’ time-filtered tweets using LIWC, which sorts different features from texts into more than 70 different categories. To uncover the correlated LIWC word categories, we used Fisher’s linear discriminant analysis with IBM Corp’s SPSS statistical program. We got 34 word categories in total with high correlation coefficient scores. Categories with higher scores (mostly greater than 2.0) are better predictors.  The scores are in the range from 2.686 to 42.778.

### Filtering using WordNet

WordNet provides a list of the alternatives from which different words can be taken. Words and word senses can have a far more comprehensive range of semantic links in WordNet. Nouns, verbs, adjectives, and adverbs are organized into synsets of cognitive synonyms, each expressing a separate notion. In our experiment, from the selected LIWC categories in the subsection, we created a word pool for each type. This wordpool contains example words from the LIWC dictionary. We will find synonyms of the words in the word pool by using WordNet. We then checked if a tweet had a word present in the wordpool. For a particular user, if a word in the wordpool is present in the tweet, we consider that tweet. Otherwise, that tweet was discarded. Finally, the user CSV files were updated with word-filtered tweets. 

## Contextualized Word Embeddings using BERT

We used Bidirectional Encoder Representations from Transformers (BERT) to generate contextualized word embeddings. We generated BERT vectors using a pre-trained model named [Bert-based-uncased](https://huggingface.co/bert-base-uncased) from Hugging Face sentence-transformers library. The tweet sentences were converted into a 768-dimensional vector space. Shape of the data after word embedding is 924x768. BERT word embeddings is our final dataset which contains both psycholinguistic and word embedding–based analyses regarding the users' tweets. These analysis are the independent variables, while the dependent variable is Irregular Sleep Yes and Irregular Sleep No. BERT word embeddings is our final dataset which is used as input to our classification models (deep learning architectures) . 


