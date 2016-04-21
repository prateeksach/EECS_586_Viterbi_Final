# AUTHOR: Prateek Sachdeva
# EECS 586, Final Project
# Viterbi Algorithm - Part of Speech Tagging

# Import all libraries
import nltk, sys
from nltk.corpus import brown

print("Reading the NLTK Corpus.")

# Read in the corpus from NLTK
# Read it as a pair of tuples between (word, tag)
corpus_words = []
for sent in brown.tagged_sents():
  corpus_words.append(("START", "START"))
  corpus_words.extend([(tag[:2], word) for (word, tag) in sent])
  corpus_words.append(("END", "END"))

# Generate the tag set from the corpus and the unique ones from the dataset
corpus_tags = [tag for (tag, word) in corpus_words]
unique_tags = set(corpus_tags)

# Get conditional probability distribution for the vocabulary/words AND the tags
words_conditional_prob = nltk.ConditionalProbDist(nltk.ConditionalFreqDist(corpus_words), nltk.MLEProbDist)
tags_conditional_prob = nltk.ConditionalProbDist(nltk.ConditionalFreqDist(nltk.bigrams(corpus_tags)) , nltk.MLEProbDist)

print("Corpus read and probabilities calculated.")
while 1:
  # Get the sentence from the command line and make it an array
  sentence = input("\nEnter a sentence: ").split(" ")

  # Initialize variables for the actual viterbi iteration
  viterbi_arr, backpointer_arr = [], []
  viterbi_obj, backpointer_obj = {}, {}

  # Do the first iteration before we start the loop
  # This is to see probabilities of the first word as every part-of-speech tag
  for tag in unique_tags:
    if tag == "START": 
      continue
    
    viterbi_obj[tag] = tags_conditional_prob["START"].prob(tag) * words_conditional_prob[tag].prob(sentence[0])
    backpointer_obj[tag] = "START"

  # Add first iteration to array
  viterbi_arr.append(viterbi_obj)
  backpointer_arr.append(backpointer_obj)

  # Print potential tag sequence for the first word in the sentence
  # This will be START [tag]
  current_best_tags = max(viterbi_obj.keys(), key = lambda tag: viterbi_obj[tag])
  print("\nWord: '" + sentence[0] + "'. Best Tag Sequence: " + backpointer_obj[current_best_tags] + " " + current_best_tags)

  # Loop over and do viterbi iterations
  for index in range(1, len(sentence)):
    # Initialize temp variables for this iteration
    viterbi_tmp, backpointer_tmp = {}, {}
    previous_round = viterbi_arr[len(viterbi_arr) - 1]
    
    # Iterate over the tags and calculate the new probabilities
    for tag in unique_tags:
      if tag == "START":
        continue

      best_previous = max(previous_round.keys(), key = lambda tag_tmp: previous_round[tag_tmp] * tags_conditional_prob[tag_tmp].prob(tag) * words_conditional_prob[tag].prob(sentence[index]))

      # Update forward and backward data for each possible tag
      viterbi_tmp[tag] =  previous_round[best_previous] * tags_conditional_prob[best_previous].prob(tag) * \
                          words_conditional_prob[tag].prob(sentence[index])
      backpointer_tmp[tag] = best_previous

    # Add current round values to viterbi, backpointer arrays
    current_best_tags = max(viterbi_tmp.keys(), key = lambda tag: viterbi_tmp[tag])
    viterbi_arr.append(viterbi_tmp)
    backpointer_arr.append(backpointer_tmp)

    print("Word: '" + sentence[index] + "'. Best Tag Sequence: " + backpointer_tmp[current_best_tags] + " " + current_best_tags)

  # Get the final value to get probabilities of the last word
  previous_round = viterbi_arr[-1]
  best_previous = max(previous_round.keys(), key = lambda tag_tmp: previous_round[tag_tmp] * tags_conditional_prob[tag_tmp].prob("END"))

  # Map of tags to full names
  tag_map = {
    "AT": "Article",
    "NN": "Noun",
    "JJ": "Adjective",
    "BE": "Verb",
    "UH": "UH",
    "WR": "Adverb",
    "CS": "CS",
    "DT": "Definite Article",
    "VB": "Verb",
    "IN": "Preposition",
    "PP": "Pronoun"
  }

  # Calculate the probability of this tag sequence * END
  sequence_probabilities = previous_round[best_previous] * tags_conditional_prob[best_previous].prob("END")
  sequence_best = ["END", best_previous]

  # Start at the last best tag and backtrace from there
  backpointer_arr.reverse()
  current_best_tag = best_previous

  # Backtrace and get the final sequence of tags for the input
  for backpointer_tmp in backpointer_arr:
      sequence_best.append(backpointer_tmp[current_best_tag])
      current_best_tag = backpointer_tmp[current_best_tag]

  # Reverse it of course
  sequence_best.reverse()

  sequence_best_full = [tag_map[i] if i in tag_map else i for i in sequence_best]

  # Print out all the results
  print("\nOriginal Sentence: " + " ".join(sentence))
  print("Most Likely Tag Sequence: " + "-".join(sequence_best_full))
  print("Probability of Tag Sequence: " + str(sequence_probabilities) + "\n")
