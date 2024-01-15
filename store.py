import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import plotly.express as px
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 10)
import streamlit as st
import matplotlib.pyplot as plt


st.title("Super Store EDA")



st.write("Dataset:")

@st.cache_data
def getdata():
    df=pd.read_csv("cleaned_store.csv",index_col=0)
    return df

df=getdata()
df[0:4]





st.title("TOP 5 selling products in each Category")
def top_5_products(category):
    #Product ID is summed and sorted with respect to quantity
    top_products = df[['Category', 'Quantity', 'Product ID', 'Profit']].loc[df['Category'] == category].groupby(['Product ID']).sum().sort_values(by='Quantity', ascending=False).head(5)
    # join with original dataframe to get product names
    top_products_names = top_products.merge(on='Product ID',right=df[['Product ID','Product Name']],).drop_duplicates(subset='Product ID')
    return top_products_names

# for each category, run above function
categories = df['Category'].unique()
for category in categories:
    print(f'Top 5 {category} products:')
    st.write(category)
    st.write(top_5_products(category)['Product Name'])




st.title("Analysis of Sales and Profit increase or decrease over time")
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Extract month and year
df['Order_month'] = df['Order Date'].dt.month
df['Order_year'] = df['Order Date'].dt.year

# Plot sales over order_month
grouped_df_month = df[['Order_month', 'Sales']].groupby('Order_month').sum().reset_index()
fig_month = px.line(grouped_df_month, x='Order_month', y='Sales', title='Sales Over Order Month')
fig_month.update_layout(width=400, height=400)

# Plot sales over order_year
grouped_df_year = df[['Order_year', 'Sales']].groupby('Order_year').sum().reset_index()
fig_year = px.line(grouped_df_year, x='Order_year', y='Sales', title='Sales Over Order Year')
fig_year.update_layout(width=400, height=400)

# Use Streamlit to display both line charts
st.plotly_chart(fig_month)
st.write('Conclusion: 1 to 2 month has low sales.  3 to 8 and 10 month have medium sales. 9,11,12 month have more sales')
st.plotly_chart(fig_year)
st.write("Conclusion:  Each year sales is increasing.")





st.title("No. of order for each category")
categories = df['Category'].value_counts().reset_index(name='Orders')
# Create a Pie Chart using Plotly Express
fig = px.pie(data_frame=categories, values='Orders', names='index', title='Number of Orders for each Category')
# Use Streamlit to display the Pie Chart
fig.update_layout(width=400, height=400)
st.plotly_chart(fig)
st.write("Conclusion: Office Supplies have more order and then Furniture and then Technology")




st.title("Which category generates the highest Sales and Profit?")
# group by category then sums up the sales and profit.
group_category = df[['Profit','Sales', 'Category']].groupby('Category').sum().reset_index()
fig = px.bar(group_category, x='Category', y=['Profit', 'Sales'], barmode='group')
fig.update_layout(
    xaxis_title='Category',
    yaxis_title='Profit and sales',
    title='Profit and Sales by Category',
    width=400, 
    height=400,
)
st.plotly_chart(fig)
st.write("Conclusion: Office Supplies have more sales and then furniture then Tech. Office Supplies have more profit and then Tech then Furniture.")




st.title("Which Region generates the most Sales and Profit")
# group by the region
group_region = df[['Region', 'Sales','Profit']].groupby('Region').sum().reset_index()
fig = px.bar(group_region, x='Region', y=['Profit', 'Sales'], barmode='group')
fig.update_layout(
    xaxis_title='Region',
    yaxis_title='Profit and sales',
    title='Profit and Sales in Region',
    width=400, 
    height=400,
)
st.plotly_chart(fig)
st.write("Conclusion: West are the most region that contains sales and profit.")



st.title("Which state have more number of order?")
states_orders =df['State'].value_counts()
fig=px.bar(states_orders,title='Number of Orders for each State')
fig.update_xaxes(title_text='State')
fig.update_yaxes(title_text='No. of order')
st.plotly_chart(fig)
st.write("Conclusion: California seems to have more number of order.")




st.title("Which state generates more Profit and Sales?")
group_state = df[['State', 'Profit','Sales']].groupby('State').sum().reset_index()
fig = px.bar(group_state, x='State', y=['Profit', 'Sales'], barmode='group')
fig.update_layout(
    xaxis_title='State',
    yaxis_title='Profit and Sales',
    title='Profit and Sales in state',
)
st.plotly_chart(fig)
st.write("Conclusion: California, New York, Washington have more profit. California, New York , Texas have more sales")




st.title("Impact of the discount on the sales and profit")
# Add in column ie Has_discount if discount more than 20%. if discount more than 20% then it's true, else false
df['Has_discount'] = df['Discount'] >= 0.2
group_discount = df[['Profit','Sales', 'Has_discount']].groupby('Has_discount').sum().reset_index()
fig = px.bar(group_discount, x='Has_discount', y=['Profit', 'Sales'], barmode='group')
fig.update_layout(
    xaxis_title='Has Discount',
    yaxis_title='Profit and Sales',
    title='Profit and Sales when discount is given',
    width=400,
    height=400,
)
st.plotly_chart(fig)
st.write("Conclusion: More sales when discounts >= 20%.  More profit when there is no discounts.")







st.title("No. of the orders considering ship type and discounts.")
# Assuming df is your DataFrame
ship_mode_counts = df.groupby('Ship Mode')['Has_discount'].value_counts().unstack()

# Plot the bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ship_mode_counts.plot(kind='bar', ax=ax)
plt.xlabel('Has_discount')
plt.ylabel('Orders')
plt.title('Count of Has Discount')
plt.legend(title='Has_discount', loc='upper left')

# Use Streamlit to display the Matplotlib figure
st.pyplot(fig)
st.write("Conclusion: Standard class have high order even with or without discount")





st.title("Which sub_category has the highest demand?")
grouped_sub_category = df[['Sub-Category', 'Quantity']].groupby('Sub-Category').sum().reset_index()
fig=px.bar(grouped_sub_category, x='Sub-Category', y='Quantity').update_xaxes(categoryorder='total descending')
fig.update_layout(width=400,height=400)
st.plotly_chart(fig)
st.write("Conclusion: Binders, Paper, Art are the top 3 high demand sub categories")




st.title("Which sub_category has more sales?")
grouped_sub_category = df[['Sub-Category', 'Sales']].groupby('Sub-Category').sum().reset_index()
fig=px.bar(grouped_sub_category, x='Sub-Category', y='Sales').update_xaxes(categoryorder='total descending')
fig.update_layout(width=400,height=400)
st.plotly_chart(fig)
st.write("Conclusion: Phones,Storage,Chairs are the top 3 sale of sub-category")



st.title("Which sub categories are more profitable?")
grouped_sub_category = df[['Sub-Category', 'Profit']].groupby('Sub-Category').mean().reset_index()
fig=px.histogram(grouped_sub_category, x='Sub-Category', y='Profit').update_xaxes(categoryorder='total descending')
fig.update_layout(width=400,height=400)
st.plotly_chart(fig)
st.write("Conclusion: Machines, Appliances, Accessories are more profitable sub-category​")