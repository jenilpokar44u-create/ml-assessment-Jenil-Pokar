# Part B: Business Case Analysis

## B1. Problem Formulation

### (a) Machine Learning Problem Setup
This is a **Regression** problem. Our goal is to predict a continuous numerical value—specifically, how many items will be sold. 
* **Target Variable:** `items_sold`
* **Candidate Input Features:** `store_id`, `store_size`, `location_type`, `competition_density`, `promotion_type`, `month`, `is_weekend`, and `is_festival`.
* **Why this type?** By framing this as regression, we can simulate running all 5 promotions for a specific store in a specific month, predict the `items_sold` for each scenario, and simply recommend the promotion that yields the highest predicted number.

### (b) Target Variable Selection: Volume vs. Revenue
Using "items sold" (volume) is much more reliable here than total revenue because promotions fundamentally alter the price of items. For example, a "Flat Discount" or "BOGO" will naturally lower the revenue per item. If we used total revenue as our target, the model might penalize these promotions, even if they successfully drove massive foot traffic and cleared out tons of inventory. 
* **Broader Principle:** This illustrates that your ML target variable must align strictly with the specific business objective (moving inventory/maximizing volume), rather than just defaulting to the highest-level financial metric (revenue) which can be distorted by the experiment itself.

### (c) Alternative Modelling Strategy
Running one global model across all 50 stores might "underfit" the local nuances, but building 50 separate models is an operational nightmare to maintain. 
* **Alternative:** A clustering-based approach. We should first group the 50 stores into 3 or 4 clusters based on their static attributes (e.g., location type, store size, competition density). Then, we train one regression model per cluster. This strikes a perfect balance: it accounts for the fact that a rural small store behaves completely differently than a large urban flagship, without creating massive technical debt from managing 50 separate models.

---

## B2. Data and EDA Strategy

### (a) Data Joining and Granularity
I would use the transaction table as the base and perform a series of left joins. I'd join the calendar table on the date, the store attributes table on `store_id`, and the promotion details on both `store_id` and the date.
* **Final Grain:** Since the marketing team makes decisions on a *monthly* basis for each store, the modeling dataset needs to be aggregated to the **Store-Month** level. One row = One Store in One Month. 
* **Aggregations:** We would sum the `items_sold` for that month, take the mode (most frequent) of `promotion_type`, and sum the `is_weekend` and `is_festival` flags to get a count of busy days in that month.

### (b) Exploratory Data Analysis (EDA)
Before modeling, I would look at:
1. **Average Items Sold by Promotion Type (Bar Chart):** To see the baseline effectiveness of each promo across the whole company.
2. **Items Sold by Location Type & Promotion (Grouped Bar Chart):** To find interactions. We might see that rural stores respond best to BOGO, while urban stores prefer Loyalty Points. This confirms we need interaction terms or a tree-based model.
3. **Time-Series of Sales over 3 Years (Line Graph):** To identify monthly seasonality (e.g., big spikes in December). This tells us we definitely need to extract `month` as a feature.
4. **Correlation Heatmap of Numerical Features:** To see if `store_size` and `competition_density` are highly correlated with sales, which helps in feature selection.

### (c) Handling Unbalanced Promotion Data
If 80% of transactions have no promotion, a standard model will become biased toward predicting the "baseline" non-promotional sales and might struggle to understand the true impact of the 5 specific promotions. To fix this, I would use **sample weights** during model training. By giving higher weight to the rows where a promotion *was* active, we force the model to pay closer attention to the relationships and patterns during promotional periods, rather than just optimizing for the 80% of normal days.

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Metrics
* **The Split:** I would use an **Out-of-Time Validation** split. I'd train the model on the first 2.5 years of data and test it on the most recent 6 months. A random split is inappropriate because it leaks future data into the training set (the model shouldn't be allowed to learn from December 2023 to predict November 2023).
* **Metrics:** I would use **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**. MAE is highly interpretable for the business—it tells us, on average, how many items our prediction is off by. RMSE penalizes large errors heavier, which is important because we really want to avoid failing massively during a peak holiday month.

### (b) Explaining Different Recommendations
To explain why the model suggested Loyalty Points in December and Flat Discount in March for the same store, I would use **SHAP values** (which explain individual predictions). I would show the marketing team that in December, temporal features like `is_festival` or `month=12` had a massive positive interaction with Loyalty Points (e.g., people buying holiday gifts want to rack up points for themselves). In March, a slower month without festivals, the SHAP plot would likely show that the baseline footfall is lower, and the model learned that a Flat Discount is required as a stronger hook to drive traffic.

### (c) End-to-End Deployment
1. **Saving the model:** After training, I would serialize the final Pipeline (including the scaler, encoder, and regression model) using `joblib` into a `.pkl` file.
2. **Monthly Generation:** I'd set up a monthly batch script (via a cron job or Airflow). At the end of the month, the script pulls the upcoming month's calendar data and current store attributes. For each of the 50 stores, it generates 5 rows of data (one for each possible promotion). It feeds these 250 rows into the loaded `.pkl` model, gets the predicted sales, and filters for the maximum value for each store to generate the final recommendation table.
3. **Monitoring:** I would build a simple tracking dashboard that logs "Predicted vs. Actual" sales at the end of every month. I would set a drift alert: if the overall MAE exceeds a certain threshold (e.g., we are consistently off by more than 15%), it triggers an alert to investigate changing consumer behavior and retrain the model on the freshest data.
