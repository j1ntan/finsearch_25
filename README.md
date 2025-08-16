# Black-Scholes Option Pricing & Accuracy Analysis (Nifty50 Options)

📌 **Overview**
This Python script implements the **Black-Scholes-Merton (BSM)** model to estimate the theoretical price of **Nifty50 options** and compare them with **actual market prices** from NSE option chain data.

It calculates **accuracy metrics** like MAE, MSE, RMSE, and visualizes the difference between **market prices** and **model-predicted prices** for both **Call** and **Put** options.

---

⚙️ **Features**
- Black-Scholes-Merton Pricing for Call & Put options.
- Historical Volatility Calculation using Yahoo Finance (`yfinance`).
- Options Data Parsing from NSE CSV format.
- Error Analysis: MAE, MSE, RMSE.
- Scatter Plot Visualization for comparing market vs model prices.

---

📦 **Dependencies**
```bash
pip install numpy pandas scipy yfinance matplotlib
```
**Imported Libraries:**
- `numpy` → Numerical calculations
- `pandas` → Data handling & preprocessing
- `scipy.stats.norm` → CDF for normal distribution in BSM formula
- `yfinance` → Fetching Nifty historical data
- `matplotlib` → Plotting comparison graphs
- `datetime` → Date handling for expiry calculations

---

🔹 **Script Structure**

**1. Black-Scholes-Merton Model**
```python
def black_scholes(S, K, T, r, sigma, option_type):
    ...
```
**Inputs:**
- `S` → Current underlying price (Nifty50 index)
- `K` → Strike price of the option
- `T` → Time to expiry in years
- `r` → Risk-free interest rate
- `sigma` → Annualized volatility
- `option_type` → `'call'` or `'put'`

**Output:** Theoretical option price using BSM formula.

**2. Historical Volatility Calculation**
```python
def get_historical_volatility(ticker, period="1y"):
    ...
```
- Downloads historical Nifty50 data.
- Calculates **log returns**.
- Computes **annualized volatility** using:
  $$ \sigma_{annual} = \text{std}(\text{log returns}) \times \sqrt{252} $$

**3. Option Data Preparation**
```python
def prepare_data(file_path):
    ...
```
- Reads NSE option chain CSV.
- Cleans column names and formats strike prices.
- Splits **Call** and **Put** data.
- Extracts **expiry date** from the filename.
- Calculates **time to expiry (T)** in years.
- Converts numeric fields for calculations.
- Removes invalid price rows.

**4. Main Execution Flow**
```python
if __name__ == "__main__":
    ...
```
- Fetches **historical volatility** for Nifty50.
- Loads option chain CSV from NSE.
- Calculates **Black-Scholes price** for each option.
- Computes error metrics:
  - **MAE** → Mean Absolute Error
  - **MSE** → Mean Squared Error
  - **RMSE** → Root Mean Squared Error
- Generates **scatter plot** comparing:
  - X-axis → Market Price (LTP)
  - Y-axis → Black-Scholes Price
  - Blue points → Call options
  - Red points → Put options
  - Dashed line → Perfect fit

---

📊 **Output Example**

**Console Output:**
```
Fetching historical volatility for ^NSEI...
Annualized Historical Volatility: 15.23%
Using Underlying Price (S): 22750.45

Loading and preparing options data from dataset(calls).csv...
Total options loaded: 120

--- Model Accuracy Assessment ---
Mean Absolute Error (MAE): 12.4567
Mean Squared Error (MSE): 345.6789
Root Mean Squared Error (RMSE): 18.5901
Note: Errors are in the same units as option prices (INR).

--- Sample Results (First 10 Rows) ---
   Strike Price Option Type     LTP  BS_Price     Error
0       22500.0        call  275.45    260.12    -15.33
1       22600.0        call  198.60    205.10      6.50
...
```

**Graph:**
A scatter plot showing how close BSM prices match actual market prices.

---

📂 **File Naming Convention**
The script tries to **extract expiry date from the filename**:
`options-data-14-Aug-2025.csv`

If not found, it defaults to `14-Aug-2025`.

---

🚀 **Usage**
1.  **Download NSE option chain CSV** from NSE India.
2.  **Place CSV in the script directory** and update:
    ```python
    OPTIONS_DATA_FILE = 'dataset(calls).csv'
    ```
3.  **Run the script:**
    ```bash
    python black_scholes_analysis.py
    ```
4.  View console results & scatter plot.

---

🧠 **Notes**
- The model assumes **constant volatility & risk-free rate**.
- **Dividends are ignored** in this implementation.
- Accuracy depends on:
  - Volatility estimation
  - Time-to-expiry precision
  - Market liquidity
