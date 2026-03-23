
This is a supervised classification problem.

This is an entity resolution problem with asymmetric misclassification cost.

High precision > recall

Thus, it is better to have a missing a link over linking wrong legal entities


Candidate generation was performed using token‑based blocking to meet the runtime constraint, after which a supervised similarity model ranked candidate matches.
Final linking decisions were made using the threshold that minimised the expected matching cost on the training set.

# 21.03 TODO:
- Re-think punctuation, single character removal and legal name suffixes given names in G (single character company names present)
- Re-think how to match to entities in G which have the same name but different ids (potentially need to add a component to the threshold comparison part where I check the differences between top probabilities)
- Optimize the kNN process for TF-IDF for blocking