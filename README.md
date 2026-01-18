# 🧩 Analytics Portfolio - Dual Projects

**Data Science Bootcamp - Final Capstone**

**Author:** Nadia Rozman

---

## 📋 Project Overview

This portfolio showcases two **independent data science projects** demonstrating comprehensive analytical skills across different domains. Each project features end-to-end workflows from data exploration to actionable business insights.

## 🎯 Project Structure

This repository contains two independent analytics projects across different domains.

### 📊 Project 1: Employee Attrition & Retention Analysis

**Objective:** Analyze workforce data to identify attrition drivers and provide HR recommendations

**Domain:** Human Resources Analytics

**Techniques:**
- Exploratory Data Analysis (EDA)
- Statistical correlation analysis
- Interactive data visualization (Tableau)
- Predictive insights for retention strategies

**Key Findings:**
- 16.12% overall attrition rate
- Work-life balance increases attrition risk by 45.5%
- Employees under 20 have 3x higher turnover
- Long commutes (20km+) correlate with 25.6% attrition

**Tools:** Python, pandas, matplotlib, seaborn, Tableau Public

[→ View Project 1 Details](Project_1_Employee_Attrition)

---

### 💬 Project 2: Customer Sentiment Analysis

**Objective:** Extract insights from customer reviews to improve service quality

**Domain:** Hospitality & Customer Experience

**Techniques:**
- Natural Language Processing (NLP)
- Text preprocessing (tokenization, lemmatization)
- Feature engineering (BoW vs TF-IDF comparison)
- Sentiment analysis (VADER)
- Machine learning classification (Neural Networks)

**Key Findings:**
- 73.8% positive customer sentiment
- Room quality, staff service, and check-in efficiency are critical drivers
- Neural network achieves 77.1% classification accuracy
- Execution quality matters more than feature availability

**Tools:** Python, NLTK, scikit-learn, pandas, matplotlib, WordCloud

[→ View Project 2 Details](Project_2_Sentiment_Analysis)

---

## 🗂️ Repository Structure

```
Analytics_Portfolio_Dual_Projects/
│
├── README.md                          # This file
│
├── Project_1_Employee_Attrition/      # HR Analytics Project
│   ├── employee_data.csv              # Workforce data
│   ├── Attrition_Analysis.ipynb       # Statistical analysis
│   ├── README.md                      # Detailed documentation
│   └── images/                        # Visualizations
│       ├── 1_income_distribution.png
│       ├── 2_tenure_income_relationship.png
│       ├── 3_tableau_main_dashboard.png
│       └── 4_tableau_department_view.png
│
└── Project_2_Sentiment_Analysis/      # NLP & ML Project
    ├── hotel_reviews.xlsx             # Customer reviews
    ├── Sentiment_NLP_Analysis.ipynb   # NLP analysis
    ├── README.md                      # Detailed documentation
    └── images/                        # Visualizations
        ├── 1_ratings_overview.png
        ├── 2_sentiment_distribution.png
        ├── 3_reviews_wordcloud.png
        ├── 4_model_performance.png
        └── 5_confusion_matrix.png
```

## 🛠️ Technical Skills Demonstrated

### Data Analysis & Statistics
- Exploratory Data Analysis (EDA)
- Descriptive statistics
- Correlation analysis
- Distribution analysis
- Statistical inference

### Data Visualization
- Python (matplotlib, seaborn)
- Tableau Public (interactive dashboards)
- Word clouds
- Heatmaps and correlation matrices
- Custom plotting and styling

### Natural Language Processing
- Text preprocessing (cleaning, tokenization)
- Lemmatization
- Stop word removal
- Feature extraction (BoW, TF-IDF)
- Sentiment analysis (VADER)

### Machine Learning
- Multi-Layer Perceptron (Neural Networks)
- Train-test split and cross-validation
- Model evaluation (accuracy, precision, recall, F1-score)
- Confusion matrix analysis
- Class imbalance handling
- Feature engineering comparison

### Programming & Tools
- Python 3.x
- Jupyter Notebooks
- Pandas, Numpy (data manipulation)
- Scikit-learn (ML framework)
- NLTK (NLP toolkit)
- Tableau (business intelligence)
- Git/GitHub (version control)

## 📊 Key Visualizations

### Project 1: Employee Attrition Analysis

**Interactive Tableau Dashboard:**

![Tableau Dashboard](project-1-employee-attrition/images/3_tableau_main_dashboard.png)
*Interactive dashboard showing 16.12% attrition rate and key risk factors*

**Statistical Analysis:**

![Income vs Tenure](project-1-employee-attrition/images/2_tenure_income_relationship.png)
*Relationship between tenure, job level, and compensation revealing moderate correlation (r=0.51)*

### Project 2: Sentiment Analysis

**Word Cloud Visualization:**

![Word Cloud](project-2-sentiment-analysis/images/3_reviews_wordcloud.png)
*Visual representation of most frequent terms in 10,000 customer reviews*

**Sentiment Distribution:**

![Sentiment Distribution](project-2-sentiment-analysis/images/2_sentiment_distribution.png)
*VADER analysis showing 73.8% positive, 11.6% negative, 13.3% neutral sentiment*

**Model Performance:**

![Model Performance](project-2-sentiment-analysis/images/4_model_performance.png)
*Neural network achieving 77.1% accuracy with strong performance on positive class*

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.x required
python --version

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud openpyxl scipy jupyter
```

### Running the Projects

**Project 1 (Employee Analysis):**
```bash
cd project-1-employee-attrition
jupyter notebook attrition_analysis.ipynb
```

**Project 2 (Sentiment Analysis):**
```bash
cd project-2-sentiment-analysis
jupyter notebook sentiment_nlp_analysis.ipynb
```

**Tableau Dashboard (Project 1):**
- View online: [Tableau Public Dashboard](https://public.tableau.com/shared/FF83FP5W4?:display_count=n&:origin=viz_share_link)

## 📈 Project Highlights

### Business Impact

**Project 1 - Employee Attrition:**
- Identified potential savings of $2.1M+ through retention improvements
- Provided 6 actionable recommendations for HR strategy
- Created interactive dashboard for ongoing monitoring
- Revealed that job level drives income more than tenure

**Project 2 - Customer Sentiment:**
- Revealed operational improvements with 15-20% impact potential
- Enabled data-driven service quality decisions
- Built predictive model for sentiment classification
- Identified that execution quality matters more than features

### Technical Achievement

- **End-to-end workflows** from raw data to actionable insights
- **Multiple analytical approaches** (statistical, visual, NLP, ML)
- **Production-ready code** with comprehensive documentation
- **Reproducible results** with clear methodology
- **Business-focused recommendations** from technical findings

## 🎓 Learning Outcomes

This portfolio demonstrates proficiency in:

1. **Data Science Fundamentals**
   - Problem formulation and scoping
   - Data cleaning and preprocessing
   - Feature engineering and selection

2. **Statistical Analysis**
   - Descriptive and inferential statistics
   - Correlation and relationship analysis
   - Distribution analysis and interpretation

3. **Machine Learning**
   - Supervised learning (classification)
   - Model evaluation and validation
   - Performance optimization and tuning

4. **Natural Language Processing**
   - Text preprocessing pipelines
   - Sentiment analysis techniques
   - Feature extraction methods (BoW vs TF-IDF)

5. **Data Visualization**
   - Static visualizations (matplotlib, seaborn)
   - Interactive dashboards (Tableau)
   - Effective visual storytelling

6. **Business Communication**
   - Translating technical findings to business insights
   - Actionable recommendations with timelines
   - Executive summaries and documentation

---

### ✨ Created by Nadia Rozman | January 2026

**🔗 Connect with me**
- GitHub: [@NadiaRozman](https://github.com/NadiaRozman)
- LinkedIn: [Nadia Rozman](https://www.linkedin.com/in/nadia-rozman-4b4887179/)

**⭐ If you found this project helpful, please consider giving it a star!**
