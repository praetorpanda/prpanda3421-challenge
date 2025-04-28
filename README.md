# prpanda3421-challenge

Competition Information
Competition Name: Multimedia Street Satellite Matching Challenge (ACMMM25)

Participant ID: rpanda3421

Code Link
(https://github.com/praetorpanda/prpanda3421-challenge)

Brief Method Description
In this challenge, I followed the official baseline approach for cross-view geo-localization. The process was as follows:

Training
I trained the model using the official train.py script with the University-1652 training set. Multiple settings were tested, but the baseline configuration provided the best results.

Feature Extraction
After training, I used test.py to extract features from the name-masked test set.

Matching and Submission Generation
Finally, I used the demo1.py script to perform retrieval. The script reads the query image list from query_street_name.txt, calculates feature similarities, retrieves the top-10 satellite images for each query, and saves the results in the answer.txt file following the required submission format.

Experimental Results
Among various settings tested, the baseline setting achieved the best performance on the masked test set:

Rank-1 Accuracy: 1.4300%

Rank-5 Accuracy: 5.0400%

Rank-10 Accuracy: 8.4900%

This setting was used for the final submission to the CodaLab leaderboard.
