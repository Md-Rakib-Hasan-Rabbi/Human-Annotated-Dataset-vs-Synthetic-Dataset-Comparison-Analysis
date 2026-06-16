import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

def main():
    print("1. Downloading the SST2 dataset...")
    dataset = load_dataset("glue", "sst2", split="train")
    
    df = pd.DataFrame(dataset)
    texts = df['sentence'].values
    labels = df['label'].values
    print(f"   Loaded {len(texts)} examples.")
    
    print("\n2. Extracting text features...")
    # Using 5000 features for a fast MVP run
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    print("\n3. Training model using 5-fold cross-validation...")
    clf = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(clf, X, labels, cv=5, method="predict_proba")
    
    print("\n4. Using cleanlab to find label errors...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by='self_confidence'
    )
    
    print(f"\n   Found {len(issue_indices)} potential label errors!")
    
    # --- TERMINAL PREVIEW (From your code) ---
    if len(issue_indices) > 0:
        print("\n--- Top 5 Potential Errors (Preview) ---")
        for i, idx in enumerate(issue_indices[:5]):
            print(f"\nError #{i+1} (Dataset Index: {idx})")
            print(f"Sentence: \"{texts[idx]}\"")
            print(f"Given Label: {labels[idx]} (0=Negative, 1=Positive)")
            print(f"Predicted Probability [Neg, Pos]: {pred_probs[idx].round(3)}")

    # --- TEAM CSV EXPORT (From my code) ---
    print("\n5. Exporting the Top 100 for Team Validation...")
    top_100_indices = issue_indices[:100]
    suspicious_df = df.iloc[top_100_indices].copy()
    
    # Add columns for the manual review phase
    suspicious_df['Team_Corrected_Label'] = ""
    suspicious_df['Error_Type'] = ""
    suspicious_df['Notes'] = ""

    suspicious_df.to_csv("SST2_Top_100_Suspects.csv", index=False)
    print("   Export complete! 'SST2_Top_100_Suspects.csv' is ready.")

if __name__ == "__main__":
    main()