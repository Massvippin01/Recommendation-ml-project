# 🎬 RECOMMENDATION SYSTEM PROJECT
## CODTECH Internship Task

---

## 📋 PROJECT OVERVIEW

Complete recommendation system implementing **User-Based CF**, **Item-Based CF**, and **Matrix Factorization (SVD)** to predict movie ratings and generate personalized recommendations.

**Dataset:** MovieLens-style synthetic dataset (200 users, 50 movies, 3000+ ratings)

**Objective:** Build and compare collaborative filtering algorithms

---

## 🚀 QUICK START

```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy

# Launch Jupyter
jupyter notebook

# Open Recommendation_System.ipynb and run all cells!
```

---

## 📦 WHAT'S INCLUDED

### **1. Data Analysis**
- Rating distribution
- User activity patterns
- Movie popularity
- Data sparsity analysis

### **2. Three CF Algorithms**

**User-Based CF:**
- Finds similar users
- Weighted rating prediction
- Cosine similarity

**Item-Based CF:**
- Finds similar movies
- More stable than user-based
- Precomputable similarities

**SVD Matrix Factorization:**
- Latent factor model
- Gradient descent training
- Best accuracy

### **3. Evaluation**
- RMSE, MAE
- Precision@K, Recall@K
- Coverage analysis
- Prediction visualization

### **4. Recommendations**
- Top-N movie recommendations
- Similar movie suggestions
- User rating history
- Personalized lists

---

## 📊 EXPECTED RESULTS

```
Model Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User-Based CF  RMSE: 0.65
Item-Based CF  RMSE: 0.62
SVD            RMSE: 0.58 ← Best
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recommendation Example:
User has rated: The Matrix, Inception
Recommendations:
1. Interstellar (predicted: 4.7)
2. The Dark Knight (predicted: 4.5)
3. The Prestige (predicted: 4.4)
```

---

## 🎯 KEY FEATURES

✅ Three collaborative filtering methods  
✅ Custom SVD implementation  
✅ Multiple evaluation metrics  
✅ Personalized recommendations  
✅ Similar item discovery  
✅ Coverage & diversity analysis  
✅ Comprehensive visualizations  



*"Good recommendations create great experiences!"*

---

**CODTECH Internship Task Complete! ✓**
