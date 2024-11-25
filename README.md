# Digital Transformation in Banking Industry

This project aims to enhance decision-making in the banking sector by predicting the potential occurrence of the next tech bubble. By leveraging various AI and machine learning techniques, this project combines a classification model with time series analysis to forecast critical economic indicators, providing insights that can aid in risk assessment and investment strategies.

## Project Overview

The **Digital Transformation in Banking Industry** project utilizes supervised and ensemble machine learning models to predict the next tech bubble. Key economic parameters such as GDP, CPI, labor productivity, and R&D spending serve as input features to this model. The project also includes a robust time series model trained on several algorithms to forecast these parameters over the next five years, ultimately feeding into the classification model to improve tech bubble prediction accuracy.

### Key Components

1. **Classification Model**: Built using supervised and ensemble techniques, this model classifies whether a tech bubble is likely to occur based on predicted values from the time series model.  
2. **Time Series Model**: Developed using algorithms like ARIMA, SARIMAX, and Prophet, this model predicts economic indicators over a five-year period. Among these, the Prophet algorithm was chosen for its high accuracy and minimal loss.

### Economic Parameters Used
- **Gross Domestic Product (GDP)**
- **Consumer Price Index (CPI)**
- **Labor Productivity**
- **R&D Spending**
- **Stock Value**

### Methodology

1. **Time Series Forecasting**:
   - The time series model forecasts the next five years of GDP, CPI, labor productivity, R&D spending, and stock values.
   - The model was trained using multiple time series algorithms (ARIMA, SARIMAX, and Prophet). Prophet was selected due to its superior performance in minimizing loss and achieving higher accuracy.
   
2. **Classification for Tech Bubble Prediction**:
   - Predictions from the time series model serve as inputs to the classification model.
   - This classification model employs supervised and ensemble techniques to analyze the forecasted data, providing a probabilistic prediction of the occurrence of a tech bubble.

### Model Architecture

The project consists of two interconnected components:
- **Time Series Model**: Outputs five-year forecasts for key economic indicators.
- **Classification Model**: Consumes time series predictions to classify the likelihood of a tech bubble.

### Technologies and Algorithms

- **Time Series Algorithms**: ARIMA, SARIMAX, Prophet
- **Classification Techniques**: Supervised learning, ensemble methods
- **Programming Languages**: Python
- **Libraries**: Scikit-learn, Prophet, Statsmodels, Pandas, Numpy, Matplotlib

## Getting Started

### Prerequisites
- Python 3.8 or above
- Libraries: Install dependencies using `pip install -r requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/whoisusmanali/Digital-Transformation-in-Banking-Industry.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Digital-Transformation-in-Banking-Industry
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Train the Time Series Model**:
   ```bash
   python train_time_series.py
   ```
2. **Train the Classification Model**:
   ```bash
   python train_classification.py
   ```
3. **Run Predictions**:
   ```bash
   python predict_tech_bubble.py
   ```

## Results

The selected Prophet algorithm for time series predictions achieved high accuracy with minimal loss. The five-year forecasts from this model were then used by the classification model to predict the likelihood of a tech bubble occurrence.

## Project Structure

- `data/`: Contains datasets for training and testing.
- `models/`: Stores trained model files.
- `scripts/`: Contains the main scripts for training and prediction.
- `README.md`: Project documentation.

## Future Enhancements

- Integrate additional economic indicators to improve model accuracy.
- Develop a visualization dashboard for real-time tracking of predictions.
- Explore advanced deep learning techniques for time series forecasting.


## TimeSeries Forecasting
![alt text](<Screenshot 2024-11-01 at 6.18.27 PM.png>)

![alt text](<Screenshot 2024-11-01 at 6.17.53 PM-1.png>)


## Insights:
<img width="1235" alt="Screenshot 2024-11-15 at 7 34 36 PM" src="https://github.com/user-attachments/assets/37876f37-227e-443e-9e00-f0b45cfcb67a">

---
