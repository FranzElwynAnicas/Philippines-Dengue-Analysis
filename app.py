#When I wrote this, only God and I understood what I was doing. 
#Now, God only knows 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


st.set_page_config(
    page_title="Philippines Dengue Analysis",
    page_icon="🦟",
    layout="wide"
)

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/FranzElwynAnicas/Data-Analysis-1/main/doh-epi-dengue-data-2016-2021.csv'
    df = pd.read_csv(url)
    df_cleaned = df.iloc[1:].copy()
    
    df_cleaned["date"] = pd.to_datetime(df_cleaned["date"], errors="coerce")
    df_cleaned["year"] = df_cleaned["date"].dt.year
    df_cleaned["deaths"] = pd.to_numeric(df_cleaned["deaths"], errors="coerce")
    df_cleaned["cases"] = pd.to_numeric(df_cleaned["cases"], errors="coerce")
    df_cleaned["month_name"] = df_cleaned["date"].dt.strftime("%B")
    df_cleaned["time_period"] = pd.cut(
        df_cleaned["year"],
        bins=[2015, 2017, 2019, 2021],
        labels=["2016-2017", "2018-2019", "2020-2021"]
    )
    return df_cleaned

df_cleaned = load_data()

st.sidebar.header("Filters")
selected_years = st.sidebar.multiselect(
    "Select years",
    options=sorted(df_cleaned["year"].unique()),
    default=sorted(df_cleaned["year"].unique())
)
selected_regions = st.sidebar.multiselect(
    "Select regions",
    options=sorted(df_cleaned["Region"].unique()),
    default=sorted(df_cleaned["Region"].unique())
)

filtered_df = df_cleaned[
    (df_cleaned["year"].isin(selected_years)) & 
    (df_cleaned["Region"].isin(selected_regions))
]

st.title("🦟 Philippines Dengue Cases Analysis (2016-2021)")
st.write("Analyzing dengue cases and deaths across different regions and time periods")

col1, col2, col3 = st.columns(3)
total_cases = filtered_df["cases"].sum()
total_deaths = filtered_df["deaths"].sum()
avg_cases = filtered_df["cases"].mean()

col1.metric("Total Cases", f"{total_cases:,}")
col2.metric("Total Deaths", f"{total_deaths:,}")
col3.metric("Average Cases", f"{avg_cases:,.1f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Time Trends", 
    "Regional Analysis", 
    "City-Level Insights",
    "Statistical Tests"
])

with tab1:
    st.header("Time Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Dengue Cases by Year")
        monthly_cases = filtered_df.groupby(['year', 'month_name'])['cases'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=monthly_cases, 
            x='month_name', 
            y='cases', 
            hue='year',
            marker='o',
            ax=ax,
            linewidth=2.5
        )
        ax.set_title('Monthly Dengue Cases by Year', fontweight='bold')
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Number of Cases', fontweight='bold')
        plt.xticks(rotation=45)
        

        max_2019 = monthly_cases[monthly_cases['year'] == 2019]['cases'].max()
        ax.annotate(f'2019 Peak: {max_2019:,.0f} cases', 
                   xy=('August', max_2019), 
                   xytext=(10, 10), 
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->'))
        
        st.pyplot(fig)
        
        with st.expander("📊 Insights: Monthly Trends"):
            st.markdown("""
            - **Consistent seasonal patterns**: Cases peak mid-year (June-August) and often show secondary peaks late in the year
            - **Record outbreak in 2019**: August 2019 saw over 80,000 cases in a single month (rainy season peak)
            - **Post-2020 decline**: 2020-21 cases dropped significantly due to:
              - COVID-19 lockdowns reducing mosquito exposure
              - Enhanced public health measures
              - Possible underreporting during pandemic
            - **Long-term trend**: 2021 cases remained low, suggesting lasting impact of prevention efforts
            """)
    
    with col2:
        st.subheader("COVID-19 Impact: 2019 vs 2020")
        df_covid = filtered_df[filtered_df["year"].isin([2019, 2020])]
        covid_cases = df_covid.groupby("year")["cases"].sum().reset_index()
        
        pct_change = ((covid_cases.iloc[1]['cases'] - covid_cases.iloc[0]['cases']) / covid_cases.iloc[0]['cases']) * 100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bar = sns.barplot(x="year", y="cases", data=covid_cases, palette="coolwarm")
        ax.set_title("Dengue Cases Before vs During Pandemic", fontweight='bold')
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Total Cases', fontweight='bold')
        
        for p in bar.patches:
            bar.annotate(f"{p.get_height():,.0f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        fontweight='bold')
        
        st.pyplot(fig)
        
        with st.expander("🦠 Insights: Pandemic Impact"):
            st.markdown(f"""
            - **Dramatic {abs(pct_change):.0f}% decrease** in cases from 2019 to 2020
            - **Key factors**:
              - Lockdowns reduced outdoor mosquito exposure
              - Increased hygiene and sanitation measures
              - Possible healthcare system focus on COVID affecting reporting
            - **Public health implications**:
              - Behavior changes can significantly impact dengue transmission
              - Need to maintain prevention efforts post-pandemic
            """)


    st.markdown("""
    ### 🔍 Key Takeaways:
    - **Seasonal Preparedness**: The consistent mid-year peaks highlight the need for pre-rainy season interventions
    - **Pandemic Lessons**: The 2020-21 decline shows human behavior's significant impact on disease transmission
    - **Monitoring Needed**: Watch for post-pandemic rebound cases as normal activities resume
    """)

with tab2:
    st.header("Regional Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Cases by Region")
        region_cases = filtered_df.groupby('Region')['cases'].sum().reset_index()
        region_cases = region_cases.sort_values('cases', ascending=False)
        
        top_regions = region_cases.head(2)
        highlight_colors = ['firebrick' if x in top_regions['Region'].values else 'steelblue' for x in region_cases['Region']]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bar = sns.barplot(
            x='cases',
            y='Region',
            data=region_cases,
            palette=highlight_colors
        )
        ax.set_title('Dengue Cases by Region (2016-2021)', fontweight='bold')
        ax.set_xlabel('Total Cases', fontweight='bold')
        ax.set_ylabel('')
        
        for p in bar.patches:
            width = p.get_width()
            plt.text(width + 5000, p.get_y()+0.55*p.get_height(),
                    f'{int(width):,}',
                    ha='center', va='center')
        
        st.pyplot(fig)
        
        with st.expander("🗺 Insights: Region case"):
            st.markdown("""
            **Top Affected Regions:**
            - **CALABARZON (IV-A)**: 140,000+ cases - Dense urban populations + favorable breeding conditions
            - **CENTRAL LUZON (III)**: 120,000+ cases - Industrial areas with water storage containers
            
            **Notable Observations:**
            - **NCR**: Ranks 4th despite highest population density - Possible underreporting during COVID
            - **BARMM/CAR**: Lowest cases - Cooler climates and lower population density
            
            **Environmental Factors:**
            - Tropical regions (VISAYAS/MINDANAO) show moderate-high cases
            - Urban areas with poor drainage consistently vulnerable
            """)
    
    with col2:
        st.subheader("Total Deaths by Region")
        region_deaths = filtered_df.groupby('Region')['deaths'].sum().reset_index()
        region_deaths = region_deaths.sort_values('deaths', ascending=False)
        
        merged = pd.merge(region_cases, region_deaths, on='Region')
        merged['fatality_rate'] = (merged['deaths']/merged['cases'])*100
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bar = sns.barplot(
            x='deaths',
            y='Region',
            data=region_deaths,
            palette='Reds_r'
        )
        ax.set_title('Dengue Deaths by Region (2016-2021)', fontweight='bold')
        ax.set_xlabel('Total Deaths', fontweight='bold')
        ax.set_ylabel('')
        
        for p in bar.patches:
            width = p.get_width()
            region = region_deaths.iloc[int(p.get_y())]['Region']
            fatality = merged[merged['Region']==region]['fatality_rate'].values[0]
            plt.text(width + 20, p.get_y()+0.55*p.get_height(),
                    f'{int(width):,} ({fatality:.1f}%)',
                    ha='left', va='center', fontsize=9)
        
        st.pyplot(fig)
        
        with st.expander("☠ Insights: Mortality rates"):
            st.markdown("""
            **High Fatality Regions:**
            - **NCR**: Highest deaths (4th in cases) → 2.1% fatality rate
            - **SOCCSKSARGEN (XII)**: 3rd in deaths → 2.3% fatality rate
            
            **Key Concerns:**
            - Disproportionate deaths in REGION V (BICOL) and MIMAROPA
            - Western Visayas (VI) shows moderate deaths despite high cases
            
            **Possible Explanations:**
            - Healthcare system strain in urban centers
            - Delayed treatment seeking behavior
            - Virulent dengue strains in specific regions
            """)

    st.markdown("""
    ### 🔍 Key Takeaways:
    
    **Transmission Hotspots:**
    - **Luzon Dominance**: 5 of top 6 affected regions in Luzon (Urbanization + tropical climate)
    - **Urban Vulnerability**: NCR, CALABARZON show high transmission in dense populations
    
    **Mortality Patterns:**
    - **Case-Fatality Mismatch**: NCR leads deaths despite 4th in cases (2.1% fatality rate)
    -**Success Stories**: CAR maintains low cases AND deaths (0.8% fatality)
    
    **Actionable Insights:**
    1. Targeted interventions in CALABARZON/Central Luzon urban centers
    2. Review healthcare response in high-fatality regions (NCR, SOCCSKSARGEN)
    3. Study CAR's successful prevention strategies for replication
    """)

with tab3:
    st.header("City-Level Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Cities by Cases")
        city_cases = filtered_df.groupby('loc')['cases'].sum().reset_index()
        top_cities = city_cases.sort_values('cases', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x='cases',
            y='loc',
            data=top_cities,
            palette='coolwarm'
        )
        ax.set_title('Cities with Highest Cases')
        st.pyplot(fig)

        with st.expander("🏙 Insights: High Cases on Cities"):
            st.markdown("""
            **Key Observations:**
            - **Cavite, Laguna, and Cebu** lead in dengue cases, reflecting high population density and tropical conditions ideal for mosquito breeding.
            - **Quezon City** appears in both high cases and deaths, indicating severe outbreaks or healthcare challenges.
            """)

    with col2:
        st.subheader("Top 10 Cities by Deaths")
        city_deaths = filtered_df.groupby('loc')['deaths'].sum().reset_index()
        top_deaths = city_deaths.sort_values('deaths', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x='deaths',
            y='loc',
            data=top_deaths,
            palette='coolwarm'
        )
        ax.set_title('Cities with Highest Deaths')
        st.pyplot(fig)

        with st.expander("🏨 Insights: High Deaths on Cities"):
            st.markdown("""
            **Key Observations:**
            - **Quezon City**, **South Cotabato**, and **General Santos City** top fatalities, suggesting gaps in healthcare response or higher virulence of dengue strains.
            - **Negros Occidental** and **Cebu** appear in both lists, linking high transmission to elevated mortality.
            """)

    st.markdown("""
    ### 🔍 Key Takeaways:
    - **Cavite, Laguna, and Cebu** are epicenters of high cases; **Quezon City** faces dual burdens of high cases and deaths.
    - **South Cotabato** and **General Santos City** have alarming death rates despite not topping case counts.
    - **Negros Occidental** and **Cebu** show consistent high cases and deaths, signaling systemic dengue risks.
    """)

    


with tab4:
    st.header("Statistical Tests")
    
    st.subheader("ANOVA Test: Deaths Across Time Periods")
    st.write("""
    **Research Question:** Is there a significant difference in dengue deaths across:
    - 2016-2017
    - 2018-2019
    - 2020-2021
    """)
    
    if st.button("Run ANOVA Test"):
        group_1 = filtered_df[filtered_df["time_period"] == "2016-2017"]["deaths"].dropna()
        group_2 = filtered_df[filtered_df["time_period"] == "2018-2019"]["deaths"].dropna()
        group_3 = filtered_df[filtered_df["time_period"] == "2020-2021"]["deaths"].dropna()
        
        f_stat, p_value = stats.f_oneway(group_1, group_2, group_3)
        
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"P-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.success("Reject H₀: Significant difference exists")
            st.write("Conclusion: Death rates differ significantly across time periods")
        else:
            st.success("Fail to reject H₀: No significant difference")
            st.write("Conclusion: Death rates are consistent across periods")

if st.checkbox("Show raw data"):
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)
    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='dengue_data_filtered.csv',
        mime='text/csv'
    )

st.markdown("---")
st.markdown("""
**Data Source:** [DOH EPI Dengue Data](https://github.com/FranzElwynAnicas/Data-Analysis-1)
""")