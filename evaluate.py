"""
Evaluation script for SHL Recommendation System
Computes Mean Recall@K on train dataset and generates predictions on test set
"""
import json
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recommender import SHLRecommender

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shl_catalog.json")
DATASET_PATH = "/mnt/user-data/uploads/Gen_AI_Dataset.xlsx"


def recall_at_k(predicted_urls, relevant_urls, k=10):
    """Compute Recall@K for a single query."""
    predicted_top_k = predicted_urls[:k]
    # Normalize URLs for comparison
    pred_set = set(u.rstrip('/').lower() for u in predicted_top_k)
    rel_set = set(u.rstrip('/').lower() for u in relevant_urls)
    if not rel_set:
        return 0.0
    hits = len(pred_set & rel_set)
    return hits / len(rel_set)


def mean_recall_at_k(all_predicted, all_relevant, k=10):
    """Compute Mean Recall@K across all queries."""
    recalls = []
    for pred, rel in zip(all_predicted, all_relevant):
        recalls.append(recall_at_k(pred, rel, k))
    return sum(recalls) / len(recalls) if recalls else 0.0


def evaluate_train(recommender):
    """Evaluate on train set and print results."""
    train_df = pd.read_excel(DATASET_PATH, sheet_name='Train-Set')
    
    # Group by query
    grouped = train_df.groupby('Query')['Assessment_url'].apply(list).reset_index()
    
    print("=" * 70)
    print("TRAIN SET EVALUATION")
    print("=" * 70)
    
    all_predicted = []
    all_relevant = []
    
    for _, row in grouped.iterrows():
        query = row['Query']
        relevant_urls = row['Assessment_url']
        
        recs = recommender.get_recommendations(query, top_k=10)
        predicted_urls = [r['url'] for r in recs]
        
        recall = recall_at_k(predicted_urls, relevant_urls, k=10)
        all_predicted.append(predicted_urls)
        all_relevant.append(relevant_urls)
        
        print(f"\nQuery: {query[:80]}...")
        print(f"  Relevant ({len(relevant_urls)}): {[u.split('view/')[-1].rstrip('/') for u in relevant_urls]}")
        print(f"  Predicted ({len(predicted_urls)}): {[u.split('view/')[-1].rstrip('/') for u in predicted_urls[:5]]}...")
        print(f"  Recall@10: {recall:.3f}")
    
    mr10 = mean_recall_at_k(all_predicted, all_relevant, k=10)
    print("\n" + "=" * 70)
    print(f"Mean Recall@10 (Train): {mr10:.4f}")
    print("=" * 70)
    return mr10


def generate_test_predictions(recommender):
    """Generate predictions on test set and save as CSV."""
    test_df = pd.read_excel(DATASET_PATH, sheet_name='Test-Set')
    
    print("\n" + "=" * 70)
    print("GENERATING TEST SET PREDICTIONS")
    print("=" * 70)
    
    rows = []
    for i, row in test_df.iterrows():
        query = row['Query']
        recs = recommender.get_recommendations(query, top_k=10)
        
        print(f"\nQuery {i+1}: {str(query)[:80]}...")
        for j, rec in enumerate(recs):
            print(f"  {j+1}. {rec['name']}")
            rows.append({
                'Query': query,
                'Assessment_url': rec['url']
            })
    
    output_df = pd.DataFrame(rows)
    output_path = "/mnt/user-data/outputs/test_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total rows: {len(output_df)}")
    return output_df


if __name__ == "__main__":
    print("Loading SHL Recommender...")
    rec = SHLRecommender(CATALOG_PATH)
    
    print(f"Catalog size: {len(rec.catalog)} assessments\n")
    
    # Evaluate on train set
    evaluate_train(rec)
    
    # Generate test predictions
    generate_test_predictions(rec)
