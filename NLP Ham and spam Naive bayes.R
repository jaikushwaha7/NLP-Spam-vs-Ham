setwd("D:/Study/Great Lakes/machine learning/NLP")

sms_raw <- read.csv(file.choose(), stringsAsFactors = F)
str(sms_raw)

sms_raw$type<- as.factor(sms_raw$type)

prop.table(table(sms_raw$type))

# 86 % ham and 13 % spam

library(tm)  # for text mining

# VCorpus - > Volatile corpus ( Utilizes Ram Memory, Faster, not suitable for large datasets, Prototyping)
# PCorpus - > Permanent (Hard Disk, Slow, suitable for large data sets, Big Data)

sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:5], as.character) # viewing multiple rows

sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
lapply(sms_corpus_clean[6:9], as.character) # lower cases 

sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
lapply(sms_corpus_clean[6:9], as.character) # Numbers Removed

# Stopwords
stopwords()
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
lapply(sms_corpus_clean[6:9], as.character) 

mystopwords<- c("cine", "amore", "bugis")
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, mystopwords)
lapply(sms_corpus[1], as.character) 
lapply(sms_corpus_clean[1], as.character)

# Punctuations
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
lapply(sms_corpus[1], as.character) 
lapply(sms_corpus_clean[1], as.character)

# Stemming
library(SnowballC)

wordStem(c("learning", "learn", "learns", "learned")) # Outputlearn

wordStem(c("go", "going", "went", "gone"))

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
lapply(sms_corpus[1], as.character) 
lapply(sms_corpus_clean[1], as.character)

# Stripwhite spaces
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
lapply(sms_corpus[1], as.character) 
lapply(sms_corpus_clean[1], as.character)


# Wordcloud
library(wordcloud)

wordcloud(sms_corpus_clean, min.freq = 50, random.order = T)

spam<- subset(sms_raw, type=="spam")
ham<- subset(sms_raw, type== "ham")
par(mfcol= c(1,2))
wordcloud(spam$text, min.freq = 50, random.order = F)
wordcloud(ham$text, min.freq = 50, random.order = F)

sms_dtm<- DocumentTermMatrix(sms_corpus_clean)
dim(sms_dtm)

sms_dtm1<- DocumentTermMatrix(sms_corpus)
dim(sms_dtm1)
rm(sms_dtm1)
# step identification
sms_corpus_clean1 <- tm_map(sms_corpus, content_transformer(tolower))
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1)  #13175

sms_corpus_clean1 <- tm_map(sms_corpus_clean1, removePunctuation)
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1)  # 9346  3829  words reduced 29 % Punctuation

sms_corpus_clean1 <- tm_map(sms_corpus_clean1, removeNumbers)
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1)  #13175 to 12249  # 7%

sms_corpus_clean1 <- tm_map(sms_corpus_clean1, removeWords, stopwords())
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1)  # 13175 to 12505 670 5%

sms_corpus_clean1 <- tm_map(sms_corpus_clean1, stemDocument)
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1) ## 13175 to 11943  9.35% 

sms_corpus_clean1 <- tm_map(sms_corpus_clean1, stripWhitespace)
sms_dtm1<- DocumentTermMatrix(sms_corpus_clean1)
dim(sms_dtm1)



#####################

inspect(sms_dtm)
5574*.8
sms_dtm_train<- sms_dtm[1:4169,]
sms_dtm_test<- sms_dtm[4170:5574,]

sms_train_labels<- sms_raw[1:4169,]$type
sms_test_labels<- sms_raw[4170:5574,]$type
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

sms_frequent_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_frequent_words)           

sms_dtm_freq_train<- sms_dtm_train[, sms_frequent_words]
sms_dtm_freq_test<- sms_dtm_test[, sms_frequent_words]

convert_count <- function(x){
  x<- ifelse(x>0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_count)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_count)

head(sms_train)
dim(sms_train)
library(e1071)
sms_classifier<- naiveBayes(sms_train, sms_train_labels)

sms_test_pred<- predict(sms_classifier, sms_test)

conf <-table(predicted = sms_test_pred, Actual = sms_test_labels)
library(caret) 
f.conf <- confusionMatrix(conf)
f.conf  #  Test Accuracy : 0.9794  

sms_train_pred<- predict(sms_classifier, sms_train)
conf <-table(predicted = sms_train_pred, Actual = sms_train_labels)
library(caret) 
f.conf <- confusionMatrix(conf)
f.conf  #  Train Accuracy : 0.9844  


#install.packages("gmodels")
install.packages("gmodels")
library("gmodels")
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, 
           prop.r = FALSE, prop.c=FALSE, dnn=c('predicted', 'actual'))

