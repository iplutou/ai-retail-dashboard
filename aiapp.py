import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
# from reportlab.lib.styles import getSampleStyleSheet
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
# import io

st.set_page_config(page_title="AI Smart Retail Dashboard", layout="wide")
st.title("AI Smart Retail Dashboard")

df = pd.read_csv("downloaded_superstore_data.csv")

from textblob import TextBlob

# CREATE SENTIMENT COLUMN

def get_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity
    
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment'] = df['Review'].apply(get_sentiment)

# ADD SYNTHETIC DATE
if 'Order Date' not in df.columns:
    df['Order Date'] = pd.to_datetime(
        np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), size=len(df))
    )

df['Year'] = pd.to_datetime(df['Order Date']).dt.year

df['Order Date'] = pd.to_datetime(df['Order Date'])

# SIDEBAR FILTERS
st.sidebar.subheader("State Filter")
state = st.sidebar.selectbox("Select State", ["All"] + sorted(df['State'].unique()))

st.sidebar.subheader("Date Filter")

start_date = st.sidebar.date_input(
    "Start Date",
    value=df['Order Date'].min().date()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df['Order Date'].max().date()
)


# FILTER LOGIC
filtered_df = df.copy()

# State filter
if state != "All":
    filtered_df = filtered_df[filtered_df['State'] == state]

# Date filter (safe conversion)
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = filtered_df[
    (filtered_df['Order Date'] >= start_date) &
    (filtered_df['Order Date'] <= end_date)
]


# SHOW INFO
st.write(f"Showing data from {start_date.date()} to {end_date.date()}")

# NLP INPUT
st.subheader("Ask Your Data")

query = st.text_input("Ask something:", placeholder="e.g. Show sales by category")

# NLP MODE
if query:

    query = query.lower()

    if "sales by category" in query:
        data = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        st.plotly_chart(px.bar(data, x='Category', y='Sales', color='Category'))

    elif "sales by state" in query:
        data = filtered_df.groupby('State')['Sales'].sum().reset_index()
        st.plotly_chart(px.bar(data, x='State', y='Sales', color='State'))

    elif "profit by year" in query:
        data = filtered_df.groupby('Year')['Profit'].sum().reset_index()
        st.plotly_chart(px.bar(data, x='Year', y='Profit'))

    else:
        st.warning("Sorry, I didn't understand your question.")

# DASHBOARD MODE
else:

    st.subheader("Dashboard Overview")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", int(filtered_df['Sales'].sum()))
    col2.metric("Total Profit", int(filtered_df['Profit'].sum()))
    col3.metric("Total Orders", len(filtered_df))

    # Sales by Category
    st.subheader("Sales by Category")
    st.plotly_chart(px.bar(filtered_df, x='Category', y='Sales', color='Category'), use_container_width=True)

    st.subheader("Sales Trend Over Time (Monthly)")
    trend = filtered_df.copy()
    trend['Month'] = trend['Order Date'].dt.to_period('M').astype(str)

    monthly_sales = trend.groupby('Month')['Sales'].sum().reset_index()

    monthly_sales = monthly_sales.sort_values('Month')
    fig = px.line(
        monthly_sales,
        x='Month',
        y='Sales',
        markers=True  
    )
    st.plotly_chart(fig)

    # Top 10 Cities by Profit
    st.subheader("🌍 Top 10 Cities by Profit")

    top_cities = (
        filtered_df.groupby('City')['Profit']
        .sum()
        .nlargest(10)
        .reset_index()
        .sort_values(by='Profit', ascending=True)  # 🔥 important for horizontal chart
    )

    st.plotly_chart(
        px.bar(
            top_cities,
            x='Profit',
            y='City',
            orientation='h'
        ),
        use_container_width=True
    )

    df['Month'] = df['Order Date'].dt.month_name()

    # Profit by Year
    st.subheader("Profit by Year")
    profit_year = filtered_df.groupby('Year')['Profit'].sum().reset_index()
    st.plotly_chart(px.bar(profit_year, x='Year', y='Profit'), use_container_width=True)

    # -----------------------------
    # Top 5 Products
    # -----------------------------
    st.subheader("Top vs Low Products Comparison")
    col1, col2 = st.columns(2)

    with col1:
        # st.markdown("## Top 5 Products")

        top = (
            filtered_df.groupby('Sub-Category')['Sales']
            .sum()
            .nlargest(5)
            .reset_index()
            .sort_values(by='Sales')   # sort for horizontal
        )

        fig_top = px.bar(
            top,
            x='Sales',
            y='Sub-Category',
            orientation='h',   # ⭐ horizontal
            # color='Sales'
        )

        st.plotly_chart(fig_top, use_container_width=True)


    # -----------------------------
    # Low 5 Products
    # -----------------------------
    with col2:
        # st.markdown("Low 5 Products")

        low = (
            filtered_df.groupby('Sub-Category')['Sales']
            .sum()
            .nsmallest(5)
            .reset_index()
            .sort_values(by='Sales')
        )

        fig_low = px.bar(
            low,
            x='Sales',
            y='Sub-Category',
            orientation='h',   # ⭐ horizontal
            # color='Sales'
        )

        st.plotly_chart(fig_low, use_container_width=True)

    
    st.subheader("Rating Distribution")

    rating_counts = filtered_df['Rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']

    fig = px.bar(
        rating_counts,
        x='Rating',
        y='Count',
        title="Rating Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    import plotly.express as px

    fig2 = px.bar(
        sentiment_counts,
        x='Sentiment',
        y='Count',
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

    
    st.subheader("Profit & Pricing Insights")

    # Create Price
    filtered_df['Price'] = filtered_df['Sales'] / filtered_df['Quantity']

    # Group data
    price_profit_df = (
        filtered_df.groupby('Sub-Category')
        .agg({
            'Profit': 'sum',
            'Price': 'mean'
        })
        .reset_index()
        .sort_values(by='Profit')
        .head(5)
    )

    # Remove index
    price_profit_df.reset_index(drop=True, inplace=True)

    # Round values
    price_profit_df['Profit'] = price_profit_df['Profit'].round(2)
    price_profit_df['Price'] = price_profit_df['Price'].round(2)

    # Status logic
    def check_status(row):
        if row['Profit'] < 0 and row['Price'] < price_profit_df['Price'].mean():
            return "Pricing Too Low"
        elif row['Profit'] < 0:
            return "Low Profit"
        else:
            return "Normal"

    price_profit_df['Status'] = price_profit_df.apply(check_status, axis=1)

    # Highlight
    def highlight(row):
        if row['Status'] == "Low Profit":
            return ['background-color: #0000'] * len(row)
        elif row['Status'] == "Pricing Too Low":
            return ['background-color: #0000'] * len(row)
        else:
            return [''] * len(row)

    # ✅ FORMAT ONLY HERE (correct place)
    st.dataframe(
        price_profit_df.style
        .format({
            "Profit": "{:,.2f}",
            "Price": "{:,.2f}"
        })
        .apply(highlight, axis=1),
        use_container_width=True
    )

    # MACHINE LEARNING
    st.subheader("🔮 Profit Prediction")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# FEATURES (ONLY 3)
X = df[['Sales', 'Discount', 'Quantity']]
y = df['Profit']

# MODEL
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# CROSS-VALIDATION
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
r2 = scores.mean()

st.metric("Model R² Score", round(r2, 2))

# TRAIN MODEL
model.fit(X, y)

# USER INPUT
sales = st.number_input("Sales", value=100.0)
discount = st.number_input("Discount", value=0.1)
quantity = st.number_input("Quantity", value=1)

# PREPARE INPUT
input_data = pd.DataFrame({
    'Sales': [sales],
    'Discount': [discount],
    'Quantity': [quantity]
})

# PREDICT
if st.button("Predict Profit"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Profit: {prediction:.2f}")

    if prediction > 0:
        st.info("This looks profitable")
    else:
        st.warning("This may result in loss")

    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    st.subheader("Dataset Summary (AI)")

# GENERATE BUTTON
if st.button("Generate Summary"):

    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()

    top_category = filtered_df.groupby('Category')['Sales'].sum().idxmax()
    top_state = filtered_df.groupby('State')['Sales'].sum().idxmax()

    avg_discount = filtered_df['Discount'].mean()

    summary = f"""The total sales is {total_sales:.2f} and total profit is {total_profit:.2f}.

The top performing category is {top_category}.

The state with highest sales is {top_state}.

The average discount is {avg_discount:.2f}.

Higher discounts may reduce profit.
"""

    # 🔥 Save summary
    st.session_state.summary = summary

# SHOW + DOWNLOAD
if "summary" in st.session_state:

    # Show summary
    st.success(st.session_state.summary)

    # CREATE PDF
    def create_pdf(summary_text):
        print(summary_text)