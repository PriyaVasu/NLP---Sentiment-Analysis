# NLP---Sentiment-Analysis
## Suggestion Mining from Online Reviews and Forums using CNN
### Import Libraries :
tensorflow, numpy, nltk, string, pandas, gensim, from nltk.corpus import stopwords, and
from gensim import corpora, models, similarities

### Text cleaning
Remove numeric and empty texts
1.	Convert five classes into two classes (positive = 1 and negative = 0)
2.	Remove punctuation from texts
3.	Convert words to lower case
4.	Remove stop words
5.	Stemming

### Tokenizing for dataset
tok_corp= [nltk.word_tokenize(sent) for sent in df['sentence']] Genism model for word 2 vector :
model = gensim.models.Word2Vec(tok_corp, min_count=1, size = 100) 

### Train dataset
I convert sentence into [maxlength X 100] size matrix . Where each row represent words most similar probablity.

Label :{0,1} /where 1 for positive and 0 for negative 

### CNN Model 
W1- [[0…	100]
W2- [1…	100]
:
Wm- [m….100]]
filter size-[5,5,1,32] Convolution layer(32 outputs)(mX100)
Max pooling layer(32 )(ma/2 X 50)
Fully connected layer1(1024 nodes) H1
Hidden layer(2) 
Softmax[0,1]

### Work Flow: 
![image alt](https://github.com/PriyaVasu/NLP---Sentiment-Analysis/blob/91c4d12b9c927e70953bb05fd3831ae09f04b47c/Suggestion%20mining%20from%20online%20review%20forums_using%20CNN.png)

### Results:
![image alt](https://github.com/PriyaVasu/NLP---Sentiment-Analysis/blob/4c30bca82e050eacf8b1d1b2163e15d0544db0ba/full_res.png)


