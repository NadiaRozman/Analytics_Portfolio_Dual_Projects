# 💬 Customer Sentiment Analysis

**Quick Snapshot:**

- Analyze **customer hotel reviews** using **NLP and Machine Learning**
- Demonstrate end-to-end text analytics: **preprocessing, feature engineering (BoW vs TF-IDF), sentiment analysis, and predictive modeling**

---

## 📌 Project Overview

This project applies Natural Language Processing (NLP) and machine learning techniques to extract insights from customer hotel reviews and predict sentiment. Using text preprocessing, feature engineering, and neural network modeling, the analysis identifies patterns in customer feedback and highlights key drivers of satisfaction and dissatisfaction.

The project bridges raw text processing, sentiment classification, and business-oriented decision support.

The analysis showcases:
- Text preprocessing and lemmatization
- Feature engineering comparison (Bag of Words vs TF-IDF)
- VADER sentiment analysis
- Keyword extraction and word cloud visualization
- Multi-Layer Perceptron (MLP) neural network for sentiment classification
- Actionable recommendations for service improvement

---

## 🎯 Background

Sentiment analysis is essential for:
- **Customer experience management** – Understand satisfaction drivers
- **Service optimization** – Identify improvement areas
- **Competitive advantage** – Respond to feedback proactively
- **Brand reputation** – Monitor customer perception

Understanding customer sentiment helps businesses:
- **Prioritize improvements** – Focus on high-impact issues
- **Prevent churn** – Address negative experiences quickly
- **Enhance loyalty** – Strengthen positive aspects
- **Data-driven decisions** – Replace guesswork with insights

---

## 🗂️ Dataset Overview

**Source:** Provided by Data Science Bootcamp (10,139 records)

**Description:** Customer hotel reviews with text feedback and numerical ratings

**Key Features:**
- **reviews.text** – Customer review text (qualitative feedback)
- **reviews.rating** – Numerical rating on 1-5 scale (quantitative metric)

**Problem Type:** Supervised Learning – **Multi-class Text Classification (Negative, Neutral, Positive)**

**Rating Distribution:**

| Rating | Count | Percentage |
|--------|-------|------------|
| 5.0 | 5,399 | 54.0% |
| 4.0 | 2,079 | 20.8% |
| 3.0 | 1,346 | 13.5% |
| 2.0 | 684 | 6.8% |
| 1.0 | 492 | 4.9% |

> **⚠️ Disclaimer**  
> This dataset was provided by Data Science Bootcamp for educational purposes only. The analysis and recommendations are based on this specific dataset and are intended to demonstrate NLP and machine learning techniques rather than represent any real hotel or organization.

---

## 🛠️ Tools I Used

**Programming & Libraries:**
- **Python** – Programming language
- **Pandas, NumPy** – Data manipulation
- **NLTK** – Natural language processing toolkit
- **Scikit-learn** – Machine learning and feature engineering
- **Matplotlib, Seaborn** – Data visualization
- **WordCloud** – Text visualization


**Development Environment:**
- **Jupyter Notebook** – Interactive development and analysis
- **Google Colab** – Optional cloud-based notebook environment

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x or higher
* Jupyter Notebook
* NLTK data packages

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NadiaRozman/analytics-portfolio-dual-projects.git
   cd analytics-portfolio-dual-projects/project-2-sentiment-analysis
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud openpyxl
   ```

3. **Download NLTK data** (first time only):

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

4. **Open and run the notebook**

   * Navigate to `sentiment_nlp_analysis.ipynb`
   * Ensure `hotel_reviews.xlsx` exists in the same directory
   * Run all cells sequentially

---

## 🔬 NLP & ML Implementation

### **1. Text Preprocessing Pipeline**

**Step 1: Basic Cleaning**
- Convert to lowercase
- Remove numerical values
- Remove punctuation and special characters
- Retain only alphabetic characters

**Step 2: Tokenization**
- Split text into individual words (tokens)
- Remove English stopwords ("the", "a", "is", etc.)
- Filter only alphabetic tokens

**Step 3: Lemmatization (Only)**
- Reduce words to base form (e.g., "running" → "run")
- Preserves semantic meaning better than stemming
- Uses WordNet lemmatizer for accuracy

**Example Transformation:**
```
Original:  "The hotel was absolutely amazing! Great staff and beautiful rooms."
Cleaned:   "hotel absolutely amazing great staff beautiful room"
Tokens:    ['hotel', 'absolutely', 'amazing', 'great', 'staff', 'beautiful', 'room']
```

---

### **2. Feature Engineering: BoW vs TF-IDF**

#### **Bag of Words (BoW)**

**Method:**
- Count frequency of each word in document
- Create matrix where each row is a document, each column is a word
- Values represent raw word counts

**Characteristics:**
- ✅ **Simple and interpretable**
- ✅ **Good for keyword frequency analysis**
- ✅ **Fast computation**
- ❌ **Treats all words equally** (common words overweighted)
- ❌ **Ignores document importance**

**Output:** Sparse matrix (10,000 documents × 5,000 features)

---

#### **TF-IDF (Term Frequency-Inverse Document Frequency)**

**Method:**
- Weight words by importance across corpus
- Rare words get higher scores
- Common words get lower scores
- Formula: `TF-IDF = (Term Frequency) × log(Total Documents / Documents with Term)`

**Characteristics:**
- ✅ **Captures word importance**
- ✅ **Better for classification tasks**
- ✅ **Reduces impact of common words**
- ❌ **Less interpretable than BoW**
- ❌ **More computationally intensive**

**Output:** Sparse matrix (10,000 documents × 5,000 features)

---

#### **BoW vs TF-IDF Comparison**

| Aspect | Bag of Words (BoW) | TF-IDF |
|--------|-------------------|---------|
| **Weighting** | Raw frequency counts | Weighted by document importance |
| **Common words** | High values for frequent words | Reduced values for common words |
| **Rare words** | Low values | High values (if discriminative) |
| **Interpretability** | ⭐⭐⭐⭐⭐ High | ⭐⭐⭐ Medium |
| **ML Performance** | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Better |
| **Use Case** | Keyword analysis, frequency | Classification, similarity |
| **Sparsity** | ~99.8% | ~99.8% |

**Decision for This Project:**
- **BoW** → Used for keyword frequency analysis and visualization
- **TF-IDF** → Used for machine learning classification (better performance)

---

### **3. Sentiment Analysis with VADER**

**VADER (Valence Aware Dictionary and sEntiment Reasoner):**
- Lexicon-based sentiment analysis tool
- Optimized for social media and review text
- Outputs compound score ranging from -1 (negative) to +1 (positive)

**Sentiment Classification Rules:**
- **Negative:** compound score < -0.05
- **Neutral:** -0.05 ≤ compound score ≤ 0.05
- **Positive:** compound score > 0.05

**VADER Advantages:**
- Fast and efficient (no training required)
- Handles emojis, slang, and intensifiers
- Works well on short texts
- Provides sentiment polarity scores

---

### **4. Machine Learning Classification**

**Model:** Multi-Layer Perceptron (MLP) Neural Network

**Architecture:**
- **Input Layer:** 5,000 features (TF-IDF vectors)
- **Hidden Layer 1:** 128 neurons with ReLU activation
- **Hidden Layer 2:** 64 neurons with ReLU activation
- **Output Layer:** 3 neurons with Softmax activation (Negative, Neutral, Positive)

**Training Configuration:**
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Categorical Crossentropy
- **Max Iterations:** 300 epochs
- **Class Weights:** Balanced — applied to mitigate dominant positive-class bias
- **Train-Test Split:** 80% training, 20% testing (stratified)

**Target Label Creation:**
- Rating < 2.5 → **Negative (0)**
- 2.5 ≤ Rating ≤ 3.5 → **Neutral (1)**
- Rating > 3.5 → **Positive (2)**

---

## 📊 Analysis Results

### **VADER Sentiment Distribution**

![VADER Sentiment](images/2_sentiment_distribution.png)
*Figure 1: VADER sentiment distribution showing strong positive skew*

| Sentiment    | Count | Percentage |
| ------------ | ----- | ---------- |
| **Positive** | 7,478 | 73.8%      |
| **Neutral**  | 1,345 | 13.3%      |
| **Negative** | 1,176 | 11.6%      |

**Interpretation:** Overall sentiment is strongly positive, indicating high customer satisfaction. However, the presence of a meaningful neutral and negative segment highlights opportunities for targeted service improvements, particularly around core guest experience factors.

---

### **Keyword Analysis**

**Top 10 Positive Keywords:**
1. room
2. hotel
3. stay
4. staff
5. great
6. clean
7. good
8. nice
9. breakfast
10. location

**Top 10 Negative Keywords:**
1. room
2. hotel
3. stay
4. bad
5. good
6. night
7. check
8. one
9. staff
10. bed


**Critical Finding:** 
The **same core features** (room, hotel, staff, stay) appear in both positive and negative contexts. This suggests that **experience quality and execution consistency**, rather than feature availability, are the primary drivers of customer sentiment.

---

### **Word Cloud Visualization**

![Word Cloud](images/3_reviews_wordcloud.png)
*Figure 2: Visual representation of most frequent words in customer reviews*

**Most Prominent Terms:**
- **Primary:** hotel, room, stay (core experience)
- **Secondary:** good, great, staff, front desk, breakfast, area
- **Supporting:** clean, nice, night, bed, check, bathroom, service, parking, location, friendly

**Insight:** Customer experiences center on **core amenities, staff interactions**, and **operational touchpoints** such as check-in and room comfort, reinforcing their importance in shaping overall satisfaction.

---

### **Machine Learning Model Performance**

**Overall Test Accuracy:** **77.1%**

**Dataset Split & Feature Space:**
- **Training Set:** 7,999 reviews
- **Test Set:** 2,000 reviews
- **Feature Dimension:** 5,000 TF-IDF features

**Classification Report:**

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| **Negative**     | 0.55      | 0.48   | 0.51     | 235     |
| **Neutral**      | 0.35      | 0.36   | 0.35     | 269     |
| **Positive**     | 0.88      | 0.89   | 0.88     | 1,496   |
| **Macro Avg**    | 0.59      | 0.58   | 0.58     | 2,000   |
| **Weighted Avg** | 0.77      | 0.77   | 0.77     | 2,000   |

**Performance Visualization:**

![Prediction Comparison](images/4_model_performance.png)
*Figure 3: Predicted vs actual sentiment distribution showing model bias toward positive class*

![Confusion Matrix](images/5_confusion_matrix.png)
*Figure 4: Confusion matrix revealing classification patterns and errors*

>**Note:** Due to strong class imbalance toward positive reviews, macro averages and class-level metrics are emphasized over accuracy alone for a more balanced evaluation.

---

### **Model Analysis**

#### **Strengths:**
- ✅ **Strong performance on positive sentiment** (Recall: 0.89)
- ✅ **Solid overall accuracy** (77.1%)
- ✅ **Reliable for identifying satisfied customers** 

#### **Weaknesses:**
- ❌ **Moderate performance on negative reviews** (Recall: 0.48)
- ❌ **Neutral sentiment remains challenging** (F1-score: 0.35)
- ❌ **Tendency to favor positive predictions due to class imbalance**

#### **Why Neutral is Challenging:**
- Mixed or ambiguous language within reviews (“good, but…”)
- Overlap between neutral and mildly positive phrasing
- Smaller neutral class compared to positive reviews

#### **Business Implication:**
The model is well-suited for **monitoring overall customer satisfaction and positive feedback**, but **manual review or rule-based escalation** is recommended for neutral and negative cases to ensure critical issues are not overlooked.

---

## 💡 Strategic Recommendations

### **High-Priority Actions (🔴)**

#### 1. Improve Room & Sleep Quality 🛏️

**Issue:** "bed" and "night" appear frequently in negative keywords

**Root Causes:**
- Bed comfort issues (mattress quality, pillows)
- Night disturbances (noise, temperature)
- Room maintenance problems

**Actions:**
- Upgrade mattresses to premium quality brands
- Improve soundproofing (windows, walls, doors)
- Address HVAC and plumbing noise issues
- Implement regular maintenance schedules
- Add blackout curtains and white noise machines

**Expected Impact:** 15-20% reduction in negative reviews

**Timeline:** 3-6 months

---

#### 2. Optimize Check-In/Check-Out Process ⏱️

**Issue:** "check" appears frequently in negative context

**Root Causes:**
- Long wait times at front desk
- Complicated procedures
- System inefficiencies
- Insufficient staffing during peak hours

**Actions:**
- Implement mobile/digital check-in options
- Add express check-out (automated)
- Staff training on efficiency and friendliness
- Optimize front desk scheduling for peak times
- Add self-service kiosks

**Expected Impact:** 25% reduction in check-in/out complaints

**Timeline:** 2-3 months

---

#### 3. Standardize Staff Service Quality 🤝

**Issue:** "staff" appears in both positive and negative reviews (inconsistency)

**Root Causes:**
- Inconsistent training across shifts
- Lack of empowerment to resolve issues
- High turnover in front-line roles
- No clear service standards

**Actions:**
- Comprehensive service excellence training program
- Empower staff for immediate issue resolution (up to $X budget)
- Implement service quality monitoring and feedback system
- Create recognition program for excellent service
- Regular refresher training and role-playing scenarios

**Expected Impact:** +15% increase in positive staff mentions

**Timeline:** 3-4 months

---

### **Medium-Priority Actions (🟡)**

#### 1. Preserve & Promote Core Strengths ✅

**Strengths:** "clean", "breakfast", "location" receive consistent positive feedback

**Actions:**
- Maintain high cleanliness standards (quality control checks)
- Keep breakfast quality and variety consistent
- Market these differentiators in promotional materials
- Feature in social media and website prominently
- Train staff to highlight these amenities during check-in

**Expected Impact:** Reinforced brand positioning and competitive advantage

**Timeline:** Ongoing

---

#### 2. Leverage Location & Environment 📍

**Strength:** "location" mentioned positively frequently

**Actions:**
- Partner with local attractions for exclusive offers
- Provide curated area guides and maps
- Highlight accessibility in marketing materials
- Offer local experience packages
- Create walking tour recommendations

**Expected Impact:** +5-10% increase in perceived value

**Timeline:** 1-2 months

---

### **Ongoing Initiatives (🟢)**

#### 1. Implement Real-Time Sentiment Monitoring 📊

**Action:** Build automated sentiment tracking system

**Implementation:**
- Deploy NLP pipeline for new reviews (daily batch processing)
- Create dashboard for tracking sentiment trends over time
- Set up alert system for negative sentiment spikes
- Integrate with customer service workflow
- Generate weekly executive summary reports

**Expected Impact:** Early detection and proactive issue resolution

**Timeline:** 1-2 months

---

#### 2. Target Neutral Experiences 😐

**Issue:** Neutral reviews (3-star ratings) are unstable and easily become negative

**Actions:**
- Proactive follow-up with 3-star reviewers
- Implement "make it right" program for borderline experiences
- Conduct satisfaction surveys for neutral ratings
- Offer small gestures (upgrade, discount) to convert neutral to positive
- Train staff to identify and address lukewarm guests during stay

**Expected Impact:** 20% conversion of neutral to positive experiences

**Timeline:** Ongoing

---

## 📈 Implementation Roadmap

| Priority | Action | Target Metric | Timeline | Owner |
|----------|--------|---------------|----------|-------|
| 🔴 | Bed upgrades & noise reduction | -15% negative reviews | 3-6 months | Operations |
| 🔴 | Digital check-in implementation | -25% check-in complaints | 2-3 months | IT/Front Desk |
| 🔴 | Staff training program | +15% positive staff mentions | 3-4 months | HR/Training |
| 🟡 | Cleanliness & breakfast standards | Maintain 95% satisfaction | Ongoing | Housekeeping/F&B |
| 🟡 | Location marketing enhancement | +10% location mentions | 1-2 months | Marketing |
| 🟢 | Sentiment monitoring system | Real-time tracking | 1-2 months | Analytics/IT |
| 🟢 | Neutral review engagement | +20% neutral→positive | Ongoing | Guest Relations |

---

## 📚 What I Learned

### **Technical Skills**
- Text preprocessing techniques (cleaning, tokenization, lemmatization)
- Feature engineering for NLP (BoW vs TF-IDF comparison)
- Sentiment analysis with VADER
- Neural network architecture for text classification
- Model evaluation and interpretation (precision, recall, F1-score)
- Handling class imbalance in ML

### **NLP Concepts**
- Why lemmatization is preferred over stemming for sentiment analysis
- How TF-IDF improves classification over BoW
- The importance of stopword removal
- Sparse matrix representation for text data
- How sentiment lexicons work (VADER)

### **Business Application**
- Translating NLP findings into actionable insights
- Identifying operational improvements from text data
- Prioritizing actions based on frequency and sentiment
- The value of execution quality over feature quantity
- How to present technical findings to non-technical stakeholders

---

## 🔮 Future Enhancements

- **Advanced models** – Implement BERT/Transformers for better context understanding
- **Aspect-based sentiment** – Separate sentiment for specific aspects (room, staff, location)
- **Time series analysis** – Track sentiment trends over time and seasonality
- **Multi-language support** – Extend to non-English reviews for international properties
- **Real-time dashboard** – Live sentiment tracking with automated alerting
- **Topic modeling** – Use LDA to discover hidden themes in reviews
- **Compare ML models** – Benchmark MLP against Random Forest, XGBoost, LSTM
- **Deploy as API** – Create REST API for real-time sentiment prediction

---

### ✨ Created by Nadia Rozman | January 2026

📂 **Project Structure**
```
Project_2_Sentiment_Analysis/
│
├── hotel_reviews.xlsx                 
├── Sentiment_NLP_Analysis.ipynb       
├── README.md                           
│
└── images/                             
    ├── ratings_overview.png
    ├── sentiment_distribution.png
    ├── keyword_analysis.png
    ├── reviews_wordcloud.png
    ├── model_performance.png
    └── confusion_matrix.png
```

---

**🔗 Connect with me**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

**⭐ If you found this project helpful, please consider giving it a star!**

---

**Part of:** [Analytics Portfolio - Dual Projects](https://github.com/NadiaRozman/Analytics_Portfolio_Dual_Projects)
