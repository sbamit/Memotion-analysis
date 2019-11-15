# datamining_proj
Term project for Data Mining
We select the data set with 6000+ rows.
header of rows:
image_name, Image_URL, OCR_extracted_text, Corrected_text, Humour, Sarcasm, Offense, Motivation, Overall_sentiment

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



We will remove any row which has null entry in one of the columns
