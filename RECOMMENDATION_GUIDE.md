# RECOMMENDATION SYSTEMS - LEARNING GUIDE
## CODTECH Internship Task

---

## 📚 TABLE OF CONTENTS

1. [What are Recommendation Systems?](#what-are-recommendation-systems)
2. [Types of Recommendation Systems](#types-of-recommendation-systems)
3. [Collaborative Filtering](#collaborative-filtering)
4. [Matrix Factorization](#matrix-factorization)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Common Challenges](#common-challenges)
7. [Real-World Applications](#real-world-applications)

---

## 🎯 WHAT ARE RECOMMENDATION SYSTEMS?

**Recommendation Systems** predict user preferences and suggest items users might like.

### Why Recommendation Systems?

**The Information Overload Problem:**
- Netflix: 15,000+ titles
- Amazon: 350+ million products
- YouTube: 500+ hours uploaded per minute

**Solution:** Help users discover relevant content

### Business Impact:

- **Netflix**: 80% of watched content comes from recommendations
- **Amazon**: 35% of revenue from recommendations
- **YouTube**: 70% of watch time from recommendations

---

## 📊 TYPES OF RECOMMENDATION SYSTEMS

### 1. **Collaborative Filtering (CF)**

**Idea:** "Users who liked similar items in the past will like similar items in the future"

**Types:**
- User-Based CF
- Item-Based CF
- Matrix Factorization

**Pros:**
- No domain knowledge needed
- Discovers unexpected patterns
- Improves over time

**Cons:**
- Cold start problem
- Scalability issues
- Data sparsity

### 2. **Content-Based Filtering**

**Idea:** Recommend items similar to what user liked before

**How:** Use item features (genre, director, actors)

**Pros:**
- No cold start for items
- Transparent recommendations
- Works with one user

**Cons:**
- Needs item features
- Over-specialization
- Limited diversity

### 3. **Hybrid Systems**

**Combine** collaborative + content-based

**Examples:**
- Netflix: CF + content + context
- YouTube: Multiple algorithms combined

---

## 🤝 COLLABORATIVE FILTERING

### User-Based Collaborative Filtering

**Algorithm:**
1. Find users similar to target user
2. Get items those users liked
3. Recommend highest-rated items

**Example:**
```
User A likes: [Movie1, Movie2, Movie3]
User B likes: [Movie1, Movie2, Movie4]

User A and B are similar!

Recommend Movie4 to User A
```

**Similarity Measures:**

**Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)

User A ratings: [5, 4, 0, 0]
User B ratings: [4, 5, 0, 0]

Cosine = (5×4 + 4×5) / (√41 × √41)
       = 40 / 41
       = 0.976 (very similar!)
```

**Pearson Correlation:**
```
Accounts for rating bias
Range: -1 to +1
```

**Steps:**
1. Build user-item matrix
2. Calculate user similarities
3. Find K nearest neighbors
4. Predict ratings using weighted average

**Prediction Formula:**
```
Predicted Rating = Σ(similarity × rating) / Σ(similarity)
```

### Item-Based Collaborative Filtering

**Algorithm:**
1. Find items similar to items user liked
2. Recommend those similar items

**Example:**
```
User likes "The Matrix"
Similar movies: "Inception", "Interstellar"
Recommend those!
```

**Why Item-Based > User-Based?**
- Items more stable than users
- Easier to explain ("Because you liked X")
- Better for large user bases
- Precompute similarities

**Implementation:**
```python
# Calculate item similarities
item_similarity = cosine_similarity(user_item_matrix.T)

# For each item user liked, find similar items
for item in user_liked_items:
    similar_items = item_similarity[item]
    recommendations.extend(similar_items)
```

---

## 🔢 MATRIX FACTORIZATION

### The Core Idea

**Problem:** User-item matrix is huge and sparse

**Solution:** Decompose into smaller matrices

```
Rating Matrix (sparse):
Users × Items = Ratings
1000 × 10000 = 10M entries (mostly empty)

Decompose:
User Matrix × Item Matrix = Predictions
1000 × 20   ×   20 × 10000

Much smaller! (20K + 200K = 220K parameters)
```

### SVD (Singular Value Decomposition)

**Formula:**
```
R ≈ U × Σ × V^T

R: Rating matrix (users × items)
U: User factors (users × k)
Σ: Singular values (k × k)
V: Item factors (k × items)
k: Number of latent factors (e.g., 20)
```

**What are Latent Factors?**

Hidden features that explain ratings:
- Factor 1: "Action intensity"
- Factor 2: "Romance level"
- Factor 3: "Complexity"
- ...

**Example:**
```
Movie: "Inception"
Latent factors: [0.8, 0.2, 0.9, ...]
              (high action, low romance, high complexity)

User: Alice
Latent factors: [0.7, 0.1, 0.8, ...]
              (likes action, not romance, likes complex)

Predicted rating = dot product
                 = 0.8×0.7 + 0.2×0.1 + 0.9×0.8 + ...
                 = High rating!
```

### Training SVD with Gradient Descent

**Prediction:**
```
predicted_rating = μ + b_u + b_i + (p_u · q_i)

μ: Global average rating
b_u: User bias (optimistic vs pessimistic)
b_i: Item bias (generally good vs bad)
p_u: User latent factors
q_i: Item latent factors
```

**Error:**
```
error = actual_rating - predicted_rating
```

**Update Rules:**
```
b_u += α × (error - λ × b_u)
b_i += α × (error - λ × b_i)
p_u += α × (error × q_i - λ × p_u)
q_i += α × (error × p_u - λ × q_i)

α: Learning rate
λ: Regularization (prevents overfitting)
```

**Training Process:**
```
For each epoch:
    For each rating:
        1. Predict rating
        2. Calculate error
        3. Update parameters
```

### Why Matrix Factorization Works

✅ **Handles sparsity**: Learns from similar patterns
✅ **Scalable**: Fewer parameters than full matrix
✅ **Accurate**: Often best performance
✅ **Flexible**: Can add biases, temporal effects

---

## 📈 EVALUATION METRICS

### 1. **Rating Prediction Metrics**

**RMSE (Root Mean Squared Error):**
```
RMSE = √(Σ(predicted - actual)² / N)

Lower is better
Penalizes large errors more
```

**MAE (Mean Absolute Error):**
```
MAE = Σ|predicted - actual| / N

Lower is better
More robust to outliers
```

**Example:**
```
Actual: [4, 5, 3, 4, 2]
Predicted: [4.2, 4.8, 3.1, 4.5, 2.3]

Errors: [0.2, 0.2, 0.1, 0.5, 0.3]
MAE = (0.2 + 0.2 + 0.1 + 0.5 + 0.3) / 5 = 0.26
RMSE = √((0.04 + 0.04 + 0.01 + 0.25 + 0.09) / 5) = 0.28
```

### 2. **Ranking Metrics**

**Precision@K:**
```
Precision@K = (Relevant items in top-K) / K

Example: Recommend 10 movies, 7 are good
Precision@10 = 7/10 = 0.7
```

**Recall@K:**
```
Recall@K = (Relevant items in top-K) / (Total relevant)

Example: User likes 20 movies, we recommend 7 of them
Recall@10 = 7/20 = 0.35
```

**F1@K:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 3. **Business Metrics**

**Click-Through Rate (CTR):**
```
CTR = Clicks / Impressions
```

**Conversion Rate:**
```
Conversion = Purchases / Recommendations
```

**Diversity:**
```
How different are recommended items?
```

**Coverage:**
```
Coverage = Unique items recommended / Total items
```

---

## 🚧 COMMON CHALLENGES

### 1. **Cold Start Problem**

**New User:**
- No rating history
- Can't find similar users

**Solutions:**
- Ask for initial preferences
- Use demographic info
- Recommend popular items
- Hybrid with content-based

**New Item:**
- No ratings yet
- Can't compare to other items

**Solutions:**
- Use item metadata
- Bootstrap with expert ratings
- Show to diverse users first

### 2. **Data Sparsity**

**Problem:** Most users rate very few items

```
Matrix: 1M users × 100K items = 100B possible ratings
Actual ratings: 100M (0.1% filled!)
```

**Solutions:**
- Matrix factorization
- Transfer learning
- Side information

### 3. **Scalability**

**User-based CF:**
```
For each prediction:
  - Compare with ALL users: O(N)
  - N = millions of users
  - Too slow!
```

**Solutions:**
- Use item-based (items more stable)
- Approximate nearest neighbors
- Precompute similarities
- Use sparse matrices

### 4. **Filter Bubble**

**Problem:** Only recommend similar items

**Example:**
```
User watches action movies
→ Only recommend action
→ Never discover other genres
```

**Solutions:**
- Add exploration/randomness
- Diversity-aware ranking
- Serendipity metrics
- Multi-objective optimization

### 5. **Popularity Bias**

**Problem:** Always recommend popular items

**Solutions:**
- Penalize popularity
- Promote long-tail items
- Personalized diversity
- Fairness constraints

---

## 🌍 REAL-WORLD APPLICATIONS

### Netflix

**Approach:** Hybrid system
- Collaborative filtering
- Content-based (genres, actors)
- Context-aware (time, device)
- Deep learning models

**Metrics:**
- Streaming hours
- Retention rate
- User satisfaction

### Amazon

**Types:**
- "Customers who bought X also bought Y" (item-based)
- "Frequently bought together"
- "Inspired by your browsing"
- "Recommended for you"

**Business Impact:**
- 35% of sales from recommendations
- Increased basket size
- Customer lifetime value

### YouTube

**Signals:**
- Watch history
- Search history
- Likes/dislikes
- Video metadata
- User demographics

**Challenges:**
- Real-time recommendations
- Billions of videos
- Fresh content prioritization

### Spotify

**Features:**
- Discover Weekly (personalized)
- Daily Mix (by genre)
- Release Radar (new music)
- Collaborative playlists

**Approach:**
- Collaborative filtering
- Audio analysis
- Natural language processing
- User behavior

---

## 💡 KEY TAKEAWAYS

1. **Collaborative Filtering** learns from collective behavior
2. **Matrix Factorization** handles sparsity effectively
3. **User-based** finds similar users, **Item-based** finds similar items
4. **SVD** discovers latent factors automatically
5. **Multiple metrics** needed for complete evaluation
6. **Cold start** is the biggest challenge
7. **Hybrid systems** often work best in practice
8. **Business metrics** matter more than accuracy

---

## 🎓 BEST PRACTICES

### Data Collection:
- Explicit feedback (ratings)
- Implicit feedback (clicks, views)
- Contextual information
- User profiles

### Model Development:
- Start simple (popularity baseline)
- Try collaborative filtering
- Experiment with matrix factorization
- Consider hybrid approaches

### Evaluation:
- Offline metrics (RMSE, Precision@K)
- Online A/B testing
- User surveys
- Business KPIs

### Production:
- Precompute when possible
- Use approximate algorithms
- Monitor performance
- Update regularly

---

**CONGRATULATIONS! 🎉**

You now understand recommendation systems!

Keep learning and building! 🚀
