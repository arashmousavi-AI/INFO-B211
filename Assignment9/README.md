# Assignment 9 - NLTK - NLP Text Analysis

## Purpose 
The purpose of this project is to perform basic natural language processing (NLP) using NLTK to analyze and compare multiple texts.

The goal is to:

- Extract meaningful information from unstructured text
- Compare writing styles across different authors
- Determine whether an unknown text matches any known author

## Approach
For each text, I applied the following steps:

1. Load the text file
2. Tokenize into words
3. Remove stop words
4. Apply stemming and lemmatization
5 Compute word frequencies
6. Extract top 20 most common tokens
7. Identify named entities
8. Generate trigrams (n = 3)

This allowed both content analysis (what the text is actually about) and style analysis (how it is written).
     
## Text Files

Text 1-->	RJ_Lovecraft.txt
Text 2-->	RJ_Tolkein.txt
Text 3-->	RJ_Martin.txt
Text 4-->	Martin.txt 

## Implementation Details
- nltk --> tokenization, POS tagging, NER
- FreqDist --> word frequency analysis
- collections.Counter --> trigram counting

## Functions
The following methods and functions were used throughout the project:

load_text(filename)
- Opens and reads the file
- Uses utf-8-sig in order to avoid hidden characters
- Returns raw text

preprocess_text(text)
- Tokenizes text into words
- Converts words to lowercase
- Removes stop words and non-alphabetic tokens
- Applies:
  - Stemming
  - Lemmatization
- Returns cleaned tokens

count_named_entities(text)
- Uses POS tagging and ne_chunk()
- Extracts named entities (such as people, locations)
- Returns a list of entities

get_trigrams(tokens)
- Creates sequences of 3 words
- Counts frequency using Counter
- Returns top 10 trigrams

## Results Summary
Texts 1 - 3 contain:
- Similar vocabulary (such as love, fate, ancient)
- Repeated named entities like Romeo, Juliet, Verona
Text 4:
- Has significantly more tokens and unique words
- Uses very different vocabulary (such as Aldric, Toran, Merck)
- Contains many different named entities

N-Gram (Trigram) Analysis
- Trigrams indicate common phrasing patterns
- Texts 1 - 3 have similar structure and wording patterns
- Text 4 shows a noticeably different writing style

## Limitations 
- Lemmatization was done without POS tagging, so results may not always be accurate
- Stemming may produce non-dictionary words
- NLTK’s named entity recognition is limited and may miss entities
- No visualization (graphs) included

## Conclusion
Based on:
- Word frequency
- Named entities
- Trigram patterns

The Text 4 does not match the authors of Texts 1–3.

It differs in vocabulary, writing style ,and entity usage
