
# Movie Recommendation System

This project is done as part of the Harvard Data Science Professional Capstone - PH125.9x.
[Data Science: Capstone](https://www.edx.org/course/data-science-capstone) is the final course for the [Professional Certificate Program in Data Science](https://www.edx.org/professional-certificate/harvardx-data-science) taught by the famous Prof. of Biostatistics Rafael Irizarry from Harvard University through [edX platform](https://www.edx.org).

<br/>

## Certificate of Completion
You can see the [Certificate of Completion](https://drive.google.com/drive/folders/1BxyOJFxdxlNcJ4o5Tf8cIS0t9aHWpK2n?usp=share_link) and other certificates in my [Certificates Repo](https://drive.google.com/drive/folders/18qDSyJrg_XFfTNX3burojlAMCdH3XgOx?usp=sharing) that contains all my certificates obtained through my journey as a self-made Data Science and better developer.

<br/>
# A recommender system based on MovieLens database that predict movie ratings.
MovieLens Recommender System Project
The primary objective of this project is to create a recommender system using the MovieLens dataset.

## Dataset Description
The version of the MovieLens dataset used for this assignment contains approximately 10 million movie ratings, divided into 9 million for training and 1 million for validation. This dataset is a small subset of a much larger and well-known dataset with several millions of ratings. The training dataset consists of around 70,000 users and 11,000 distinct movies, categorized into 20 genres such as Action, Adventure, Horror, Drama, Thriller, and more.

Evaluation Metric
After an initial data exploration, the recommender systems built on this dataset are evaluated and selected based on the Root Mean Squared Error (RMSE), which should be at least lower than 0.87750.

RMSE = sqrt(Σ(e^2)/n)

Where:
e: difference between predicted and actual rating
n: total number of ratings

## Model
For accomplishing this goal, we developed a Regularized Movie+User+Genre Model, which was capable of reaching an RMSE of 0.8628, a highly satisfactory result.

### Data Exploration
The dataset was loaded and analyzed to understand the distribution of movie ratings, user preferences, and movie genres.
Missing or erroneous data points were identified and treated accordingly.

### Model Selection
Several recommender system models were explored, including content-based filtering, collaborative filtering, and matrix factorization.
The performance of each model was evaluated using the RMSE metric.
The Regularized Movie+User+Genre Model was chosen due to its ability to provide the lowest RMSE among the tested models.

### Model Implementation
The selected model was implemented using appropriate machine learning libraries and techniques.
Model parameters were tuned to optimize the performance, and the final model achieved an RMSE of 0.8628.

### Model Deployment
The final recommender system model was deployed for use in predicting movie ratings for users based on their preferences and movie attributes.

### Future Enhancements
The model can be further refined by incorporating additional features such as user demographic data, movie release dates, and movie directors or actors.
Alternative machine learning techniques, such as deep learning, can also be explored to improve the model's performance.
