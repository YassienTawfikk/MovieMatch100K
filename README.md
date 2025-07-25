# MovieMatch100K

> MovieMatch100K is a modular recommendation system built using the MovieLens 100K dataset. It explores multiple collaborative filtering techniques with evaluation metrics to measure their practical performance.

> This project was developed as part of the Elevvo AI Internship, focusing on hands-on implementation, comparison, and evaluation of recommender system models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c3fc5505-8734-40fe-b995-baa4d1cdac39" alt="Futuristic Movie Interface" width="800" />
</p>

---

## Overview

The system supports three collaborative filtering methods:

- User-Based Collaborative Filtering  
- Item-Based Collaborative Filtering  
- Matrix Factorization using Singular Value Decomposition (SVD)

Each method predicts top-k movie recommendations for a user and can optionally be evaluated using ranking-based metrics.

---

## Dataset

- **Source**: [MovieLens 100K Dataset](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset)
- 943 users, 1682 movies
- Ratings range: 1 to 5
- Additional metadata (titles, genres) is used to display recommendations more clearly

---

## Project Structure
```
MovieMatch100K/
│
├── data/
│   ├── raw/                          # Original unmodified data files (uploaded or downloaded)
│   │   ├── u.data
│   │   ├── u.item
│   │   ├── u.user
│   │   ├── u.genre
│   │   ├── u.occupation
│   │   ├── u.info
│   │   ├── u1.base
│   │   └── ... (optional other .test/.base files)
│   │
│   ├── processed/                   # Cleaned & combined dataset ready for modeling
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── genres.csv
│   │   ├── users.csv
│   │   └── occupations.csv
│   │
│   └── curated/                    # Split Data
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── __init.py                   # init for creating all directories
│   ├── data_setup.py               # Setup the data and save it locally
│   ├── data_loader.py              # Load and return data
│   ├── user_cf.py                  # User-based collaborative filtering logic
│   ├── item_cf.py                  # (Optional) Item-based recommender
│   ├── svd.py                      # (Optional) SVD matrix factorization
│   └── evaluation.py               # Precision@K, RMSE, etc.
│
├── notebooks/
│   ├── 01_data_setup.ipynb                   # Load, clean, and save the dataset
│   ├── 02_user_cf_demo.ipynb                 # User-based filtering walkthrough
│   ├── 03_item_cf_demo.ipynb                 # Item-based filtering demo
│   └── 04_svd_demo.ipynb                     # Matrix factorization demo
│
├── main.py         # Main entry point to run recommendations
├── README.md
├── requirements.txt
└── .gitignore
```
---

## How to Run

1. Clone the repository:

```
git clone https://github.com/YassienTawfikk/MovieMatch100K.git
```
```
cd MovieMatch100K
```

2.	Install dependencies:
```
pip install -r requirements.txt
```
3.	Launch the system via:
```
python main.py
```
You can set the method and evaluation mode in the main() function:

main(method="svd", evaluate=True, k_movies=5)


---

## Evaluation Results

All models are evaluated using:

- **Precision@k**: Ratio of recommended items that are relevant  
- **Recall@k**: Ratio of relevant items that are recommended

| Method                      | Precision@5 | Recall@5 | Remarks                                                                 |
|----------------------------|-------------|----------|-------------------------------------------------------------------------|
| User-Based CF              | ~6.3%       | ~1.9%    | Struggles with sparse user overlap and poor generalization.            |
| Item-Based CF              | ~5–6%       | ~2.0%    | More stable but limited by low item co-ratings and lack of context.    |
| Matrix Factorization (SVD) | **41.78%**  | **14.75%** | Learns latent features; performs well even with sparse data.            |

**Notes**
- Movie names are shown in outputs (loaded from `data/processed/movies.csv`)
- Evaluation can be toggled via a flag in `main()`
- All models skip cold-start users (users not present in training)

---

## Submission

This project was developed as part of the **Elevvo AI Internship**, showcasing hands-on application of recommendation systems — from data curation to evaluation — with clear insights into model limitations and comparative strengths.

---

## Author

<div>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/YassienTawfikk" target="_blank">
        <img src="https://avatars.githubusercontent.com/u/126521373?v=4" width="150px;" alt="Yassien Tawfik"/>
        <br>
        <sub><b>Yassien Tawfik</b></sub>
      </a>
    </td>
  </tr>
</table>
</div>
