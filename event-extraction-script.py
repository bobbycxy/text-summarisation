## IMPORT LIBRARIES
from main_function import Summarizer

## LOCAL PARAMETERS
EXTRACTION_METHOD = 'finetuned-t5' # choose between ['finetuned-t5','weighted-frequency','tf-idf]
INPUT_ARTICLE = 'article2.txt' # choose between ['article1.txt','article2.txt','article3.txt'], or other new articles of the same format

## EVENT EXTRACTION
summariser = Summarizer(INPUT_ARTICLE,EXTRACTION_METHOD) # Instantiate the Summarizer object

## RESULTS
print("\nMain Event Sentences >>>", summariser.summarise()) # print event extraction