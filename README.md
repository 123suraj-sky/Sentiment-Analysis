# Twitter US Airline Sentiment Analysis

This project predicts the sentiment (**positive** or **negative**) of US airline tweets using a deep learning LSTM model and also compares the result with the TextBlob sentiment analysis library. The app provides a user-friendly web interface built with Streamlit, where you can:

- Enter any tweet and instantly see sentiment predictions from both the LSTM model and TextBlob side by side.
- Visualize model performance and dataset insights using pre-generated images (accuracy, loss, confusion matrix, word clouds, and sentiment ratio pie chart).

The project was developed as part of the *AI Training Programme - Data Scientist*.

[Link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment") to the dataset used here.

---

## Getting Started

### 1. Clone the repository
```sh
git clone https://github.com/123suraj-sky/Sentiment-Analysis
cd Sentiment-Analysis
```

### 2. Create a virtual environment (using venv)
```sh
python -m venv venv
```

### 3. Activate the virtual environment
- **On Windows:**
    ```sh
    venv\Scripts\activate
    ```
- **On macOS/Linux:**
    ```sh
    source venv/bin/activate
    ```

### 4. Install dependencies
```sh
pip install -r requirements.txt
```

### 5. Run the Streamlit app
```sh
streamlit run code/app.py
```

### 6. Stop the app
Press `Ctrl+C` in the terminal where Streamlit is running.
