# capstone_project

**Cross-Cultural Annotator Disagreement in AI Data**.

## Overview

Conversation AI systems are increasing used in real-world settings, yet their safety assessments often rely on majority voting or binary labels, ignoring how perceptions of harm, bias or misinformation can vary across different demographic groups.

Rather than treating disagreement as error, this project embraces annotator disagreement to uncover patterns across demographic groups (e.g., gender, race, education) and propose fairer, more inclusive AI evaluation models. Using a dataset of adversarial dialogues and rater responses, the goal is to answer these research questions:

RQ 1: How do annotator demographic attributes (e.g., race, gender, education, age) influence perceptions of
safety in AI-generated conversations?

RQ 2: Can demographics or behavior-based clustering re-
veal consistent patterns of agreement and disagree-
ment in safety annotations?

RQ 3: What types of conversational content trigger the
highest levels of inter-group disagreement, and how
can this information be used to design fairer AI
safety evaluations?

## Features
- **Clustering Raters**: Cluster annotators based on demographic information and rating behavior
- **Clustering Techniques**: KMeans, KModes, HDBSCAN, and Hierarchical clustering  
- **Disagreement Quantification**: Entropy-based metrics for variance  
- **Word Cloud Generation**: For top agreed/disagreed contexts and model responses  
- **PDF/CSV Outputs**: Plots and tables saved for reporting


## How to Run
- Clone this repository
- install required packages
- prepare the dataset (diverse_safety_adversarial_dialog_350.csv)
- run the script
