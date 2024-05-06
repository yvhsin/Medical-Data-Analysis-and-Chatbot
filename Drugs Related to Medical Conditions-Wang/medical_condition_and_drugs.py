import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('.\\Drugs Related to Medical Conditions\\drugs_for_common_treatments.csv')


columns_to_lower = ['drug_name', 'medical_condition', 'medical_condition_description']
data[columns_to_lower] = data[columns_to_lower].apply(lambda x: x.str.lower())


data['no_of_reviews'] = data['no_of_reviews'].fillna(0)

data.to_csv('.\\Drugs Related to Medical Conditions\\processed_drugs_data.csv', index=False)


df = pd.read_csv('.\\Drugs Related to Medical Conditions\\processed_drugs_data.csv')


selected_columns = ['drug_name', 'medical_condition', 'rating', 'no_of_reviews']
df_selected = df[selected_columns]


df_selected['rating'] = df_selected['rating'].fillna(-1)


df_selected.to_csv('.\\Drugs Related to Medical Conditions\\cleaned_drugs_data.csv', index=False)


df = pd.read_csv('.\\Drugs Related to Medical Conditions\\cleaned_drugs_data.csv')


df.drop(['drug_name', 'rating'], axis=1, inplace=True)


grouped_df = df.groupby('medical_condition').sum().reset_index()


grouped_df.to_csv('.\\Drugs Related to Medical Conditions\\summarized_drugs_data.csv', index=False)



df1 = pd.read_csv('.\\Drugs Related to Medical Conditions\\cleaned_drugs_data.csv')  # 第一个表
df2 = pd.read_csv('.\\Drugs Related to Medical Conditions\\summarized_drugs_data.csv')  # 第二个表


top_conditions = df2.nlargest(10, 'no_of_reviews')


plt.figure(figsize=(10, 6))
plt.pie(top_conditions['no_of_reviews'], labels=top_conditions['medical_condition'], autopct='%1.1f%%')
plt.title('Top 10 Popular Medical Conditions')
plt.savefig('.\\Drugs Related to Medical Conditions\\medical_conditions_pie_chart.png', dpi=300)
plt.show()


for condition in top_conditions['medical_condition']:
    subset = df1[df1['medical_condition'] == condition]
    top_drugs = subset.nlargest(5, 'no_of_reviews')
    print(top_drugs['drug_name'])


    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.pie(top_drugs['no_of_reviews'], labels=top_drugs['drug_name'], autopct='%1.1f%%')
    plt.title(f'No of Sales Volume for Drugs Treating {condition}')


    top_drugs['rating'] = top_drugs['rating'].replace(-1, 0)
    plt.subplot(1, 2, 2)
    plt.bar(top_drugs['drug_name'], top_drugs['rating'], color='green')
    plt.title(f'Drug Ratings for {condition}')
    plt.xlabel('Drug Name')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)

    plt.tight_layout()
    filename = f'.\\Drugs Related to Medical Conditions\\{condition}_drugs_charts.png'
    plt.savefig(filename, dpi=300)
    plt.show()