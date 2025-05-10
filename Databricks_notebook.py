# Databricks notebook source
# MAGIC %md
# MAGIC ## Data Sourcing and Structure
# MAGIC ### Reading and tagging Miami, Tampa, and Philly datasets

# COMMAND ----------

%pip install azure-storage-blob pandas matplotlib seaborn ipywidgets

from azure.storage.blob import BlobServiceClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import ipywidgets as widgets
from io import StringIO

# Azure storage account details
connection_string = "DefaultEndpointsProtocol=https;AccountName=healthcaredatastore;AccountKey=roKT3df915NsRDAGMFQhXIZgVIstGL+k3yUKM4IQojZkaLjywwAhqaUWKgT32Vb1ioWq5CgTsIu5+AStnKGG6Q==;EndpointSuffix=core.windows.net"
container_name = "hospital"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Function to read blob data into a pandas DataFrame
def read_blob_to_df(blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_data = blob_client.download_blob().content_as_text()
    return pd.read_csv(StringIO(blob_data), dtype={'standard_charge|negotiated_dollar': 'str'}, low_memory=False)

# Load the files
miami_df = read_blob_to_df('Miami_data.csv')
philly_df = read_blob_to_df('Philly_data.csv')
tampa_df = read_blob_to_df('Tampa_data.csv')

# Add region column
miami_df['region'] = 'Miami'
philly_df['region'] = 'Philadelphia'
tampa_df['region'] = 'Tampa'

# Combine the datasets
combined_df = pd.concat([miami_df, philly_df, tampa_df], ignore_index=True)

# Remove unnamed columns
combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]

# Preview the combined data
display(combined_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis (EDA)
# MAGIC ### Summary of missing values, unique entries, and data types

# COMMAND ----------

print(combined_df.info())

# COMMAND ----------

# Understanding the data and unique field count
print(combined_df.nunique())

# COMMAND ----------

# Checking if there are any missing values in the data
print(combined_df.isnull().sum())

# COMMAND ----------

# Sampling the data for test code and visual analysis
sample_df = combined_df.sample(10000, random_state=1)
print(sample_df.nunique())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analysis

# COMMAND ----------

# Hospital Pricing Summary – Grouped by Region and Sorted by Average Price
combined_df['standard_charge|negotiated_dollar'] = pd.to_numeric(
    combined_df['standard_charge|negotiated_dollar'], errors='coerce'
)

hospital_summary = (
    combined_df.groupby('hospital_name')
    .agg(
        region=('region', 'first'),
        avg_price=('standard_charge|negotiated_dollar', 'mean'),
        cpt_count=('cpt_code', pd.Series.nunique),
        payer_count=('payer_name', pd.Series.nunique)
    )
    .round(2)
    .sort_values(by=['region', 'avg_price'], ascending=[True, True])
)
pd.set_option('display.width', 200)  # or any number you prefer

print(hospital_summary)

# COMMAND ----------

# Group by payer and region, then aggregate
hospital_summary = (
    combined_df
    .groupby(['payer_name', 'region'])
    .agg(
        avg_price=('standard_charge|negotiated_dollar', 'mean'),
        cpt_count=('cpt_code', pd.Series.nunique)
    )
    .round(2)
)

# Pivot to make regions into column headers with 2 metrics
pivoted_summary = hospital_summary.unstack(level='region')

# Optional: clean column names for readability
pivoted_summary.columns = [f"{stat} ({region})" for stat, region in pivoted_summary.columns]

# Display result
print(pivoted_summary.head(10))

# COMMAND ----------

# Filter for charges up to $1000
filtered_df = combined_df[combined_df['standard_charge|negotiated_dollar'] <= 1000]

# Create box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='region', y='standard_charge|negotiated_dollar')
plt.title("Price Distribution by Region (Charges ≤ $1000)")
plt.xlabel("Region")
plt.ylabel("Standard Charge ($)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Create pivot: average charge
avg_charge_pivot = combined_df.pivot_table(
    index='payer_name',
    columns='hospital_name',
    values='standard_charge|negotiated_dollar',
    aggfunc='mean'
).round(2)

# Create pivot: CPT code count
cpt_count_pivot = combined_df.pivot_table(
    index='payer_name',
    columns='hospital_name',
    values='cpt_code',
    aggfunc=pd.Series.nunique
).fillna(0).astype(int)

# Combine the two into formatted string
combined_pivot = avg_charge_pivot.fillna(0).astype(str) + " (" + cpt_count_pivot.astype(str)+ ")"

# Display
combined_pivot

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Group data and keep region info
summary = (
    combined_df.groupby('hospital_name')
    .agg(
        avg_price=('standard_charge|negotiated_dollar', 'mean'),
        cpt_count=('cpt_code', pd.Series.nunique),
        region=('region', 'first')
    )
    .reset_index()
)

# Step 2: Create combined label for legend
summary['label'] = summary['region'] + "-" + summary['hospital_name']

# Step 3: Marker styles by region
region_markers = {
    'Miami': 'o',      # Circle
    'Philly': '^',     # Triangle
    'Tampa': 's'       # Square
}

# Step 4: Plot with custom markers and legend labels
plt.figure(figsize=(12, 6))

for region, marker in region_markers.items():
    subset = summary[summary['region'] == region]
    sns.scatterplot(
        data=subset,
        x='cpt_count',
        y='avg_price',
        hue='label',
        marker=marker,
        s=100,
        legend='full'
    )

plt.title("Hospital CPT Coverage vs. Avg Price (Region + Hospital)")
plt.xlabel("Unique CPT Codes")
plt.ylabel("Average Charge ($)")
plt.legend(title="Hospital (Region)", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()


# COMMAND ----------

# Step 1: Prepare payer summary
payer_summary = (
    combined_df.groupby('payer_name')
    .agg(
        avg_price=('standard_charge|negotiated_dollar', 'mean'),
        cpt_count=('cpt_code', pd.Series.nunique)
    )
    .round(2)
)

# Step 2: Top 10 by CPT count
top_payers = payer_summary.sort_values(by='cpt_count', ascending=False).head(10).reset_index()

# Step 3: Shorter figure height for PPT
sns.set_context("notebook", font_scale=0.85)  # Reduce font size

fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # shorter height

# CPT Code Coverage
sns.barplot(data=top_payers, y='payer_name', x='cpt_count', palette='Blues_d', ax=axes[0])
axes[0].set_title("Top 10 Payers by CPT Code Coverage")
axes[0].set_xlabel("CPT Count")
axes[0].set_ylabel("Payer")

# Avg Price
sns.barplot(data=top_payers, y='payer_name', x='avg_price', palette='Greens_d', ax=axes[1])
axes[1].set_title("Avg Negotiated Price per Payer")
axes[1].set_xlabel("Avg Price ($)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Statement 2:
# MAGIC ###### Cost Transparency for Patients(Find the cheapest hospital per treatment based on the standard_charge|negotiated_dollar.)

# COMMAND ----------

import pandas as pd

# Convert the 'standard_charge|negotiated_dollar' column to numeric
sample_df['standard_charge|negotiated_dollar'] = pd.to_numeric(
    sample_df['standard_charge|negotiated_dollar'], 
    errors='coerce'
)

# Group by treatment and hospital to get average charge
avg_price_per_treatment = (
    sample_df.groupby(['description', 'hospital_name'])['standard_charge|negotiated_dollar']
    .mean()
    .reset_index()
)

# Sort by procedure and price to get lowest-cost hospital per procedure
lowest_cost_hospitals = (
    avg_price_per_treatment
    .sort_values(['description', 'standard_charge|negotiated_dollar'])
    .groupby('description')
    .head(1)
    .reset_index(drop=True)
)

# Preview top 10
display(lowest_cost_hospitals.head(10))

# COMMAND ----------

# Step 1: Group by description, hospital, and CPT to get average price
avg_price_per_treatment = (
    sample_df
    .groupby(['description', 'hospital_name', 'cpt_code'])['standard_charge|negotiated_dollar']
    .mean()
    .reset_index()
)

# Step 2: Get top 10 descriptions to include
top10_descriptions = avg_price_per_treatment['description'].unique()[:10]

# Step 3: Filter for only those 10 descriptions
filtered_df = avg_price_per_treatment[avg_price_per_treatment['description'].isin(top10_descriptions)]

# Step 4: Plot using CPT code for x-axis
plt.figure(figsize=(14, 6))
sns.barplot(
    data=filtered_df,
    x='cpt_code',
    y='standard_charge|negotiated_dollar',
    hue='hospital_name'
)

# Step 5: Customize
plt.xlabel('CPT Code')
plt.ylabel('Average Negotiated Price ($)')
plt.title('Average Procedure Cost by Hospital (Top 10 Descriptions, Shown by CPT Code)')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'${x:,.2f}'))
plt.legend(title='Hospital')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Widget:
# MAGIC ###### Which are the top 5 lowest charges across the data based on given CPT Code

# COMMAND ----------

# --- FIX: Ensure charges are numeric for proper sorting ---
combined_df['standard_charge|negotiated_dollar'] = pd.to_numeric(
    combined_df['standard_charge|negotiated_dollar'], errors='coerce'
)

combined_df['payer_name'] = combined_df['payer_name'].astype(str).str.strip()
combined_df['cpt_code'] = combined_df['cpt_code'].astype(str)

# --- Constants ---
page_size = 20
current_page = 1
total_pages = 1

# --- Widgets for Search Input ---
search_input = widgets.Text(placeholder='Enter CPT code or keyword...', description='Search:')
region_dropdown = widgets.Dropdown(
    options=['All'] + sorted(combined_df['region'].dropna().unique().tolist()),
    value='All', description='Region:')
payer_dropdown = widgets.Dropdown(
    options=['All'] + sorted(combined_df['payer_name'].dropna().unique().tolist()),
    value='All', description='Insurance Company:')
toggle_show_all = widgets.ToggleButton(value=False, description='Show All', icon='list', tooltip='Show all or top 5')
search_button = widgets.Button(description='Search', icon='search', button_style='success')
reset_button = widgets.Button(description='Reset', icon='refresh', button_style='warning')

# Pagination widgets
prev_button = widgets.Button(description='👈', layout=widgets.Layout(width='50px'))
next_button = widgets.Button(description='👉', layout=widgets.Layout(width='50px'))
page_label = widgets.Label()
pagination_controls = widgets.HBox([prev_button, page_label, next_button])

# Output areas
summary_output = widgets.Output()
table_output = widgets.Output()
latest_results = pd.DataFrame()

# --- Download Button ---
export_button = widgets.Button(description='Export CSV', icon='download', button_style='info')

def on_export_click(b):
    if not latest_results.empty:
        export_path = '/dbfs/tmp/filtered_results.csv'  # DBFS path for storing the file
        latest_results.to_csv(export_path, index=False)
        with summary_output:
            print(f"✅ Filtered results exported as 'filtered_results.csv' to {export_path}")
    else:
        with summary_output:
            print("⚠️ No results to export. Please perform a search first.")

export_button.on_click(on_export_click)

# --- Helper Functions ---
def update_pagination_label():
    page_label.value = f"Page {current_page} of {total_pages}"

def cpt_search(cpt_or_keyword, region_input, payer_input, show_all):
    df = combined_df.copy()
    cpt_or_keyword = str(cpt_or_keyword).strip()

    if region_input != 'All':
        df = df[df['region'] == region_input]
    if payer_input != 'All':
        df = df[df['payer_name'] == payer_input]

    if cpt_or_keyword.isdigit():
        df = df[df['cpt_code'] == cpt_or_keyword]
    else:
        df = df[df['description'].str.contains(cpt_or_keyword, case=False, na=False)]

    if df.empty:
        return None, f"No entries found for: '{cpt_or_keyword}' in region: {region_input}, payer: {payer_input}"

    # ✅ Sort results by lowest price
    df_sorted = df.sort_values(by='standard_charge|negotiated_dollar')
    if not show_all:
        df_sorted = df_sorted.head(5)

    return df_sorted, None

def show_paginated_table():
    if latest_results.empty:
        return
    start = (current_page - 1) * page_size
    end = start + page_size
    # Display directly in the notebook
    display(latest_results.iloc[start:end][['cpt_code', 'description', 'standard_charge|negotiated_dollar',
                                            'region', 'payer_name', 'plan_name', 'hospital_name']])

# --- Event Handlers ---
def on_search_click(b=None):
    global current_page, total_pages, latest_results

    summary_output.clear_output()  # Clear previous messages
    table_output.clear_output()    # Clear previous table

    df, error_msg = cpt_search(
        search_input.value,
        region_dropdown.value,
        payer_dropdown.value,
        toggle_show_all.value
    )

    if df is None:
        latest_results = pd.DataFrame()
        total_pages = 1
        current_page = 1
        update_pagination_label()
    else:
        latest_results = df
        total_pages = max(1, (len(df) - 1) // page_size + 1)
        current_page = 1
        update_pagination_label()
        show_paginated_table()

    # Display result summary in summary_output
    with summary_output:
        if error_msg:
            print(error_msg)
        else:
            print(f"🔍 {len(latest_results)} result(s) found for '{search_input.value}'"
                  f" | Region: {region_dropdown.value} | Payer: {payer_dropdown.value}")

def on_reset_click(b):
    global current_page, total_pages, latest_results
    search_input.value = ''
    region_dropdown.value = 'All'
    payer_dropdown.value = 'All'
    toggle_show_all.value = False
    summary_output.clear_output()
    table_output.clear_output()
    current_page = 1
    total_pages = 1
    latest_results = pd.DataFrame()
    update_pagination_label()

def go_to_prev(b):
    global current_page
    if current_page > 1:
        current_page -= 1
        update_pagination_label()
        show_paginated_table()

def go_to_next(b):
    global current_page
    if current_page < total_pages:
        current_page += 1
        update_pagination_label()
        show_paginated_table()

# --- Bind Events ---
search_button.on_click(on_search_click)
search_input.on_submit(on_search_click)
reset_button.on_click(on_reset_click)
prev_button.on_click(go_to_prev)
next_button.on_click(go_to_next)

# --- Title ---
title = widgets.HTML("""
    <h3>🔍 Hospital Price Lookup Tool</h3>
    <p>Search by <b>CPT code</b> or keyword. Filter by <b>region</b> and <b>insurance company</b>. Click <b>Search</b> or press <b>Enter</b>. Reset to start over. Use arrows to navigate pages.</p>
""")

# --- Button Layouts for uniform size ---
button_style = widgets.Layout(width='120px')
search_button.layout = button_style
reset_button.layout = button_style
export_button.layout = button_style
toggle_show_all.layout = widgets.Layout(width='150px')

# --- Inputs Section ---
input_box = widgets.VBox([
    widgets.HTML("<h4>🔎 Search Filters</h4>"),
    widgets.HBox([search_input]),
    widgets.HBox([region_dropdown, payer_dropdown]),
    widgets.HBox([toggle_show_all, reset_button, search_button, export_button])
], layout=widgets.Layout(
    border='solid 2px lightgray',
    padding='15px',
    margin='10px 0px 10px 0px',
    background_color='#f9f9f9',
    border_radius='10px'
))

pagination_controls = widgets.HBox([
    prev_button,
    page_label,
    next_button
], layout=widgets.Layout(
    justify_content='center',
    align_items='center',
    padding='10px'
))

# --- Results Section ---
results_box = widgets.VBox([
    summary_output,
    pagination_controls,
    table_output
], layout=widgets.Layout(
    border='solid 1px lightgray',
    padding='10px',
    background_color='#ffffff',
    border_radius='10px'
))

# --- Display Everything Together ---
display(widgets.VBox([
    title,
    input_box,
    widgets.HTML("<hr>"),
    results_box
]))


# COMMAND ----------

# Step 1: Import necessary libraries
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO

# Step 2: Define your Azure Blob Storage connection details
connection_string = "DefaultEndpointsProtocol=https;AccountName=healthcaredatastore;AccountKey=roKT3df915NsRDAGMFQhXIZgVIstGL+k3yUKM4IQojZkaLjywwAhqaUWKgT32Vb1ioWq5CgTsIu5+AStnKGG6Q==;EndpointSuffix=core.windows.net"  # actual connection string
container_name = "hospital"  # Your container name
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Step 3: Define a function to read CSV files from Azure Blob Storage
def read_csv_from_blob(blob_name):
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)
    blob_data = blob_client.download_blob()
    content = blob_data.readall().decode('utf-8')
    return pd.read_csv(StringIO(content))

# Step 4: Load the data files from the container
philly_df = read_csv_from_blob('Philly_data.csv')
tampa_df = read_csv_from_blob('Tampa_data.csv')
miami_df = read_csv_from_blob('Miami_data.csv')

# Step 5: Clean column names to remove invalid characters
def clean_column_names(df):
    df.columns = df.columns.str.replace('[ ,;{}()\n\t=]', '_', regex=True)
    return df

philly_df = clean_column_names(philly_df)
tampa_df = clean_column_names(tampa_df)
miami_df = clean_column_names(miami_df)

# Step 6: Convert Pandas DataFrames to Spark DataFrames
spark_philly_df = spark.createDataFrame(philly_df)
spark_tampa_df = spark.createDataFrame(tampa_df)
spark_miami_df = spark.createDataFrame(miami_df)

# Step 7: Define the table names
philly_table_name = "philly_data_table"
tampa_table_name = "tampa_data_table"
miami_table_name = "miami_data_table"

# Step 8: Save the Spark DataFrames to Databricks SQL Warehouse (Delta Tables)
spark_philly_df.write.format("delta").mode("overwrite").saveAsTable(philly_table_name)
spark_tampa_df.write.format("delta").mode("overwrite").saveAsTable(tampa_table_name)
spark_miami_df.write.format("delta").mode("overwrite").saveAsTable(miami_table_name)

# Step 9: Optionally, verify the data is saved by querying the tables
display(spark.sql(f"SELECT * FROM {philly_table_name} LIMIT 10"))
display(spark.sql(f"SELECT * FROM {tampa_table_name} LIMIT 10"))
display(spark.sql(f"SELECT * FROM {miami_table_name} LIMIT 10"))