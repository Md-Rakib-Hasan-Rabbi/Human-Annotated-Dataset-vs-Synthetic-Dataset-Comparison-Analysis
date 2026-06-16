import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.rank import get_label_quality_scores
import os

def main():
    csv_path = os.path.join("dataset", "synth2.csv")
    output_path = "synth2_top_100_suspects.csv"
    
    print(f"1. Loading dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return
        
    df = pd.read_csv(csv_path)
    texts = df['sentence'].values
    labels = df['label'].values
    print(f"   Loaded {len(texts)} examples.")
    
    print("\n2. Extracting text features (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    print("\n3. Training model using 5-fold cross-validation...")
    clf = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(clf, X, labels, cv=5, method="predict_proba")
    
    print("\n4. Calculating label quality scores with cleanlab...")
    # Lower score indicates higher likelihood of label error
    scores = get_label_quality_scores(labels=labels, pred_probs=pred_probs)
    
    # Add scores and predictions to dataframe
    df['Label_Quality_Score'] = scores
    df['Predicted_Label'] = pred_probs.argmax(axis=1)
    df['Prob_Negative'] = pred_probs[:, 0].round(4)
    df['Prob_Positive'] = pred_probs[:, 1].round(4)
    
    # Sort by score in ascending order (lowest quality score first)
    df_sorted = df.sort_values(by='Label_Quality_Score', ascending=True)
    
    print("\n--- Top 5 Most Suspicious Sentences (Lowest Quality Scores) ---")
    for i in range(5):
        row = df_sorted.iloc[i]
        print(f"\nRank #{i+1} (Index: {row.name})")
        print(f"Sentence: \"{row['sentence']}\"")
        print(f"Given Label: {row['label']} | Predicted Label: {row['Predicted_Label']}")
        print(f"Label Quality Score: {row['Label_Quality_Score']:.4f}")
        print(f"Probabilities [Neg, Pos]: [{row['Prob_Negative']}, {row['Prob_Positive']}]")
            
    print(f"\n5. Exporting top 100 suspects to {output_path}...")
    top_100_suspects = df_sorted.head(100).copy()
    
    # Add manual verification columns for the team
    top_100_suspects['Team_Corrected_Label'] = ""
    top_100_suspects['Error_Type'] = ""
    top_100_suspects['Notes'] = ""
    
    top_100_suspects.to_csv(output_path, index=True, index_label='Original_Index')
    print(f"   Export complete! '{output_path}' is ready for review.")

if __name__ == "__main__":
    main()
