# Job Posting Fraud Detection 

## Overview 
**Detecting fraudulent job postings** using machine learning and NLP-based features.

---

## Dataset  
The dataset containes job postings. 
- **Target:** "fraudulent" 
- **Class imbalance:** 95% are real job posting, while 5% are fraudulent. 

## Evaluation 
The official test set had no labels, so model performance was evaluated using a stratified 70-30 split from the labeled training data. 

## Models Evaluated
- **Decision Tree**
- **MLP**
- **SVM**
- **Bagged Decision Trees ensemble** 
The best model was selected based on ROC-AUC using 10-fold stratified cross-validation. 

## The Selected Best Model 
The selected model was **Decision tree ensemble**, which achieved **0.91 ROC-AUC** on the 30% test split set. 

## how to Run
install dependencies: 
```bash
pip install -r requirements.txt
```

Then run the training pipeline:

```bash
python -m src.train
```




