
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Video Game Sales Analysis",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Sales Analysis")
st.write("Explore and analyze video game sales across different regions.")

# --------------------------------
# 1. Upload Dataset
# --------------------------------

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload vgsales.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

else:
    try:
        df = pd.read_csv("vgsales.csv")
    except FileNotFoundError:
        st.warning("Please upload your vgsales.csv dataset.")
        st.stop()

# Keep original data
df = df.copy()

required_columns = [
    'Platform', 'Genre', 'Year', 'Publisher',
    'NA_Sales', 'EU_Sales', 'JP_Sales',
    'Other_Sales', 'Global_Sales'
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing columns: {missing_columns}")
    st.stop()

# --------------------------------
# 2. Missing Values
# --------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Dataset")
st.dataframe(df.head(10), use_container_width=True)

if st.checkbox("Show Missing Values"):
    st.dataframe(
        df.isnull().sum().rename("Missing Count"),
        use_container_width=True
    )

# Fill missing values
df['Year'] = df['Year'].fillna(0)
df['Publisher'] = df['Publisher'].fillna('unknown')

df['Year'] = df['Year'].astype('int64')

st.subheader("After Handling Missing Values")
st.dataframe(
    df.isnull().sum().rename("Missing Count"),
    use_container_width=True
)

# --------------------------------
# 3. Data Types and Statistics
# --------------------------------

st.header("2. Statistical Analysis")

if st.checkbox("Show Data Types"):
    st.write(df.dtypes)

if st.checkbox("Show Statistical Summary"):
    st.dataframe(df.describe(), use_container_width=True)

# --------------------------------
# 4. Histograms
# --------------------------------

st.header("3. Sales Distribution")

sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales']

col1, col2 = st.columns(2)

with col1:
    st.subheader("Platform-wise Regional Sales")

    platform_sales = df.groupby('Platform')[
        sales_columns
    ].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    platform_sales.plot(kind='bar', ax=ax)
    ax.set_ylabel("Sales (Millions)")
    ax.set_xlabel("Platform")
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Genre-wise Regional Sales")

    genre_sales = df.groupby('Genre')[
        sales_columns
    ].sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    genre_sales.plot(kind='bar', ax=ax)
    ax.set_ylabel("Sales (Millions)")
    ax.set_xlabel("Genre")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------
# 5. Boxplot Before Outlier Removal
# --------------------------------

st.header("4. Outlier Detection")

st.subheader("Before Outlier Removal")

fig, ax = plt.subplots(figsize=(9, 5))
sb.boxplot(data=df[sales_columns], ax=ax)
ax.set_ylabel("Sales (Millions)")
st.pyplot(fig)
plt.close(fig)

# --------------------------------
# 6. IQR Outlier Removal
# --------------------------------

st.subheader("Remove Outliers Using IQR")

remove_outliers = st.sidebar.checkbox(
    "Apply IQR Outlier Removal",
    value=False
)

if remove_outliers:

    original_rows = len(df)

    for i in sales_columns:

        q1 = np.percentile(df[i], 25)
        q3 = np.percentile(df[i], 75)

        iqr = q3 - q1

        c1 = q1 - 1.5 * iqr
        c2 = q3 + 1.5 * iqr

        df = df[
            (df[i] >= c1) & (df[i] <= c2)
        ]

    removed_rows = original_rows - len(df)

    st.success("Outliers removed successfully!")

    col1, col2 = st.columns(2)

    col1.metric("Rows Before", original_rows)
    col2.metric("Rows After", len(df))

    st.write("Total rows removed:", removed_rows)

else:
    st.info("Enable 'Apply IQR Outlier Removal' from the sidebar.")

# --------------------------------
# 7. Boxplot After Outlier Removal
# --------------------------------

st.subheader("After Outlier Removal")

fig, ax = plt.subplots(figsize=(9, 5))
sb.boxplot(data=df[sales_columns], ax=ax)
ax.set_ylabel("Sales (Millions)")
st.pyplot(fig)
plt.close(fig)

# --------------------------------
# 8. Global Sales Analysis
# --------------------------------

st.header("5. Global Sales Analysis")

sorted_sales = df.sort_values(
    'Global_Sales',
    ascending=False
)

st.subheader("Games Sorted by Global Sales")

st.dataframe(
    sorted_sales.head(20),
    use_container_width=True
)

total_sales = df['Global_Sales'].sum()

st.metric(
    "Total Global Sales",
    f"{total_sales:.2f} Million"
)

# --------------------------------
# 9. Platform-wise Global Sales
# --------------------------------

st.header("6. Platform-wise Global Sales")

platform_global = df.groupby(
    'Platform'
)['Global_Sales'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 5))
sb.barplot(
    x=platform_global.index,
    y=platform_global.values,
    ax=ax
)

ax.set_xlabel("Platform")
ax.set_ylabel("Global Sales (Millions)")
ax.tick_params(axis='x', rotation=90)

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    platform_global.rename("Global Sales"),
    use_container_width=True
)

# --------------------------------
# 10. Genre-wise Global Sales
# --------------------------------

st.header("7. Genre-wise Global Sales")

genre_global = df.groupby(
    'Genre'
)['Global_Sales'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
sb.barplot(
    x=genre_global.index,
    y=genre_global.values,
    ax=ax
)

ax.set_xlabel("Genre")
ax.set_ylabel("Global Sales (Millions)")
ax.tick_params(axis='x', rotation=45)

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    genre_global.rename("Global Sales"),
    use_container_width=True
)

# --------------------------------
# 11. Genre-wise Year Sales
# --------------------------------

st.header("8. Genre-wise Year Analysis")

genre_year = df.groupby(
    'Genre'
)['Year'].sum().sort_values(ascending=False)

st.dataframe(
    genre_year.rename("Total Year"),
    use_container_width=True
)

# --------------------------------
# 12. Top 10 Publishers
# --------------------------------

st.header("9. Top 10 Publishers")

top_publishers = df.groupby(
    'Publisher'
)['Global_Sales'].sum().nlargest(10)

fig, ax = plt.subplots(figsize=(12, 5))
sb.barplot(
    x=top_publishers.values,
    y=top_publishers.index,
    ax=ax
)

ax.set_xlabel("Global Sales (Millions)")
ax.set_ylabel("Publisher")

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    top_publishers.rename("Global Sales"),
    use_container_width=True
)

# --------------------------------
# 13. Regional Sales
# --------------------------------

st.header("10. Regional Sales Comparison")

regional_columns = [
    'NA_Sales',
    'EU_Sales',
    'JP_Sales',
    'Other_Sales'
]

regional_sales = df[regional_columns].sum()

fig, ax = plt.subplots(figsize=(9, 5))
sb.barplot(
    x=regional_sales.index,
    y=regional_sales.values,
    ax=ax
)

ax.set_xlabel("Regions")
ax.set_ylabel("Total Sales (Millions)")

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    regional_sales.rename("Total Sales"),
    use_container_width=True
)

# --------------------------------
# 14. Sort Regional Sales
# --------------------------------

st.header("11. Regional Sales Sorting")

sort_column = st.selectbox(
    "Select Sales Column",
    regional_columns
)

sorted_region = df.sort_values(
    sort_column,
    ascending=False
)

st.dataframe(
    sorted_region[
        ['Name', 'Platform', 'Genre'] + regional_columns
    ].head(20),
    use_container_width=True
)

# --------------------------------
# 15. Download Processed Dataset
# --------------------------------

st.header("12. Download Dataset")

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Processed CSV",
    data=csv,
    file_name="processed_vgsales.csv",
    mime="text/csv"
)

st.success("Video Game Sales Analysis Completed!")
