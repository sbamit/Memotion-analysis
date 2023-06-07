# datamining_proj
Term project for Data Mining which we selected - Memotion analysis prediction. That is we attempted to catagorize what emotion a meme is expressing through its text contents and also tried to quantify that emotion on a scale of 0-3; still we couldn't implement the quantification. But implemented the emotion prediction for a meme on the basis of text contents.

We selected the data set with 6000+ rows.

We designed to catagorize a meme as follows:

Humor:
not_funny   (0)
funny       (1)
very_funny  (2)
hilarious   (3)
need to assign some tags to the outliers

Sarcasm:
not_sarcastic   (0)
general         (1)
twisted_meaning (2)
very_twisted    (3)
need to assign some tags to the outliers

Offense:
not_offensive     (0)
slight            (1)
very_offensive    (2)
hateful_offensive (3)

Motivation:
not_motivational  (0)
motivational      (1)

Overall_sentiment:
very_negative   (0)
negative        (1)
neutral         (2)
positive        (3)
very_positive   (4)


Additional instructiosn on how to run and test the project.
1.	Must keep the files named ‘data_6512_new.csv’ and ‘meme_final.py’ in one folder. Here ‘data_6512_new.csv’ is the dataset for the project and ‘meme_final.py’ is the python script to run.
2.	We run our code in Spyder environment in Anaconda. 
3.	Run the python script ‘mem_final.py’. It will generate 4 accuracy graphs one after another in the console, two for overall sentiment (positive/negative/neutral) and two for all the labels of
    sentiments(humor/sarcasm/offensive/overall)
4.	The following libraries will be needed to run the program:
    numpy 
    pandas 
    sklearn
    os    
    matplotlib.pyplot
    nltk
    re
Thank you. 
