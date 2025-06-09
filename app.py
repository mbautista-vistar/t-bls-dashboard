import streamlit as st
import pandas as pd
import psycopg2
import os
import re
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from streamlit_oauth import OAuth2Component


client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
redirect_uri = "http://localhost:8501"  # Change to Heroku URI after deploy

oauth2 = OAuth2Component(
    client_id=client_id,
    client_secret=client_secret,
    authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
    token_endpoint="https://oauth2.googleapis.com/token",
    revoke_endpoint="https://oauth2.googleapis.com/revoke",
)

result = oauth2.authorize_button(
    name="Continue with Google",
    redirect_uri=redirect_uri,
    scope="https://www.googleapis.com/auth/userinfo.email",
    key="google",
)

if result and "token" in result:
    st.success("You are logged in")
    st.write(result)
else:
    st.warning("You must log in to view this app")
    st.stop()


@st.cache_resource(ttl=3600)
def connect_to_redshift():
    conn = psycopg2.connect(
        host=st.secrets["host"],
        port=st.secrets["port"],
        database=st.secrets["database"],
        user=st.secrets["user"],
        password=st.secrets["password"]
    )
    return conn

def query_campaign_data(conn):
    query = """
    SELECT
        campaign,
        cleaned_sf_id,
        brand,
        answer_label,
        exposed_rate,
        relative_lift
    FROM master_table_brand_study_summary
    WHERE segmentation = 'Campaign'
        AND (answer_label in ('Awareness', 'Consideration', 'Intent'))
        AND brand like 'T_Mobile';
    """
    return pd.read_sql_query(query, conn)

def query_venue_performance(conn):
    query = """
        SELECT cleaned_sf_id, answer_label, segmentation_type, segmentation, control_rate, exposed_rate, relative_lift
        FROM master_table_brand_study_summary
        WHERE answer_label in ('Awareness', 'Consideration', 'Intent')
            AND segmentation_type = 'venue_type'
            AND segmentation <> 'Control'
    """
    return pd.read_sql_query(query, conn)

# query for metrics to aggregate under the telecom vertical
def query_vertical_performance(conn):
    query = """
            select brand, campaign, mfour_campaign, cleaned_sf_id, answer_label, segmentation_type, segmentation, relative_lift, exposed_rate
            from master_table_brand_study_summary
            where vertical ilike 'Telecom'
                and brand <> 'T_Mobile'
                and segmentation_type like 'venue_type'
                and answer_label in ('Awareness', 'Consideration', 'Intent')
                and segmentation like 'Campaign';
    """
    return pd.read_sql_query(query, conn)

def extract_quarter_year(text):
    match = re.search(r'q(\d)(\d{2})', text.lower())
    if match:
        quarter = match.group(1)
        year = "20" + match.group(2)
        return f"Q{quarter} {year}"
    return None

# Convert to YYYY-MM for sorting
def quarter_to_date(qy):
    if qy is None:
        return None
    q, y = qy.split(" ")
    q_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    return pd.to_datetime(f"{y}-{q_map[q]}-01")

def get_opps_with_audience(db_path="sample.db"):
    db_path = os.path.join(os.path.dirname(__file__), "sample.db")
    conn = sqlite3.connect(db_path)
    query = """
            SELECT opp_name, case_safe_id, total_aud_rev
            FROM tmo_opps_audiences
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("T-Mobile BLS Benchmark Dashboard")

    multi = '''Note: For some opportunities (refer to `cleaned_sf_id` values), there are multiple campaigns (i.e., multiple `campaign` entries in the `brand_study_summary` table per `cleaned_sf_id`). \n\nThus, data reflected here might not reflect what was shared in PPTs with the client unless these tables are updated or removed so that each `cleaned_sf_id` value has a unique `campaign` value.\n\nThe `campaign` column from the dataframe below will be dropped to improve readability once the above is resolved.
    '''
    st.markdown(multi)

    # connect to database and query data
    conn = connect_to_redshift()
    df = query_campaign_data(conn)

    # pull SQLite opp names and IDs with audience rev > 0
    audience_df = get_opps_with_audience()

    df = df.merge(
        audience_df[['case_safe_id', 'opp_name', 'total_aud_rev']],
        left_on='cleaned_sf_id',
        right_on='case_safe_id',
        how='left'
    )

    df['audience'] = None
    df.loc[df['total_aud_rev'] > 0, 'audience'] = True
    df.loc[df['total_aud_rev'] == 0, 'audience'] = False

    df = df.drop(columns=['case_safe_id', 'total_aud_rev'])
    final_df = df.drop_duplicates()

    # quarter year combos
    final_df['year'] = final_df['campaign'].apply(lambda x: "20" + re.search(r'q\d(\d{2})', x).group(1) if re.search(r'q\d(\d{2})', x) else None)
    final_df['quarter'] = final_df['campaign'].apply(lambda x: "Q" + re.search(r'q(\d)\d{2}', x).group(1) if re.search(r'q(\d)\d{2}', x) else None)
    final_df['quarter_year'] = final_df['campaign'].apply(extract_quarter_year)
    final_df['quarter_start'] = final_df['quarter_year'].apply(quarter_to_date)

    # bc these initially had an obj dtype
    final_df['relative_lift'] = pd.to_numeric(final_df['relative_lift'])
    final_df['exposed_rate'] = pd.to_numeric(final_df['exposed_rate'])

    # Reorder columns
    desired_order = ['opp_name', 'campaign', 'cleaned_sf_id', 'audience', 'answer_label', 'exposed_rate', 'relative_lift', 'quarter', 'year'] + [col for col in final_df.columns if col not in ['opp_name', 'campaign', 'cleaned_sf_id', 'audience', 'answer_label', 'exposed_rate', 'relative_lift', 'quarter', 'year']]
    final_df = final_df[desired_order]

    # Sort
    final_df = final_df.sort_values(by=["brand", "year", "quarter"], ascending=[True, True, True])

    # ---- Streamlit ----

    st.sidebar.header("Filters")

    # Sidebar filters
    opportunity = st.sidebar.multiselect('Select Campaign(s)', options=sorted(final_df['opp_name'].dropna().unique()))
    lob_search = st.sidebar.text_input("Search LOB", "")
    years = st.sidebar.multiselect('Select Study Years', options=sorted(final_df['year'].dropna().unique()))
    audience_used = st.sidebar.multiselect('Filter Audience Targeting Used/Not Used', options=final_df['audience'].dropna().unique())
    
    
    # Apply filters
    filtered_df = final_df.copy()
    if opportunity:
        filtered_df = filtered_df[filtered_df['opp_name'].isin(opportunity)]
    if lob_search:
        filtered_df = filtered_df[
            filtered_df['opp_name'].str.contains(lob_search, case=False, na=False, regex=True)
        ]
    if years:
        filtered_df = filtered_df[filtered_df['year'].isin(years)]
    if audience_used:
        filtered_df = filtered_df[filtered_df['audience'].isin(audience_used)]
    

    st.subheader(f"Displaying {filtered_df['cleaned_sf_id'].nunique()} Unique Campaigns")
    st.dataframe(filtered_df.drop(columns=['brand', 'quarter_year', 'quarter_start']), use_container_width=True, hide_index=True)
    # st.dataframe(filtered_df, use_container_width=True)


    if not filtered_df.empty:
        st.subheader("Average Lift per Metric")
        col1, col2, col3 = st.columns(3)

        awareness_lift = filtered_df[filtered_df['answer_label'] == 'Awareness']['relative_lift'].mean()
        consideration_lift = filtered_df[filtered_df['answer_label'] == 'Consideration']['relative_lift'].mean()
        intent_lift = filtered_df[filtered_df['answer_label'] == 'Intent']['relative_lift'].mean()

        col1.metric("Awareness Lift (%)", f"{awareness_lift*100:.0f}%")
        col2.metric("Consideration Lift (%)", f"{consideration_lift*100:.0f}%")
        col3.metric("Intent Lift (%)", f"{intent_lift*100:.0f}%")

        # download button
        st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), file_name="selected_benchmark_data.csv")


        # lift filters
        metrics = st.multiselect(label="Select metrics to display", options=sorted(filtered_df['answer_label'].dropna().unique()), default=["Awareness", "Consideration", "Intent"])
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overall", "Yearly", "Quarterly (General)", "QoQ Lifts", "Exposed Rates per Quarter", "Lift per Venue Type", "Benchmarks for the Telecom Vertical"])

        filtered_lift_df = filtered_df[filtered_df['answer_label'].isin(metrics)]

        # Group and average
        tmo_overall = filtered_lift_df.groupby(['brand', 'answer_label'])['relative_lift'].mean().reset_index()
        year_lifts = filtered_lift_df.groupby(['year', 'answer_label'])['relative_lift'].mean().reset_index()
        # let's see avg metrics per quarter
        quarter_lifts = filtered_lift_df.groupby(['quarter', 'answer_label'])['relative_lift'].mean().reset_index()

        # now to show QoQ growth
        # Group by quarter and answer_label
        quarterly_lift = (
            filtered_lift_df[filtered_lift_df['answer_label'].isin(metrics)]  # limit to selected metrics if needed
            .groupby(['quarter_year', 'quarter_start', 'answer_label'])[['exposed_rate', 'relative_lift']]
            .mean()
            .reset_index()
            .sort_values('quarter_start')
        )

        quarterly_lift['qoq_change'] = (
            quarterly_lift
            .groupby('answer_label')['relative_lift']
            .pct_change()
        )

        # VENUE PERFORMANCE
        venue_performance_df = query_venue_performance(conn)
        # Then filter in pandas using campaign identifiers
        allowed_ids = filtered_df['cleaned_sf_id'].unique()
        filtered_venue_df = venue_performance_df[venue_performance_df['cleaned_sf_id'].isin(allowed_ids)]
        filtered_venue_df['relative_lift'] = pd.to_numeric(filtered_venue_df['relative_lift'])

        # VERTICAL BENCHMARKS
        telecom_df = query_vertical_performance(conn)
        telecom_df['relative_lift'] = pd.to_numeric(telecom_df['relative_lift'])
        telecom_df['exposed_rate'] = pd.to_numeric(telecom_df['exposed_rate'])

        with tab1:
            # Plot
            fig = px.bar(
                tmo_overall,
                x="brand",
                y="relative_lift",
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"relative_lift": "Average Lift (%)", "answer_label": "Metric", "brand": "Brand"},
                title="Average Lifts Overall",
                text=[f"{v:.0%}" for v in tmo_overall['relative_lift']]
            )
            fig.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = px.bar(
                year_lifts,
                x="year",
                y="relative_lift",
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"relative_lift": "Average Lift (%)", "answer_label": "Metric", "year":"Year"},
                title="Average Lifts by Year",
                text=[f"{v:.0%}" for v in year_lifts['relative_lift']]
            )
            fig2.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            fig3 = px.bar(
                quarter_lifts,
                x="quarter",
                y="relative_lift",
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"relative_lift": "Average Lift (%)", "answer_label": "Metric", "quarter": "Quarter"},
                title="Average Lifts by Quarter (General)",
                text=[f"{v:.0%}" for v in quarter_lifts['relative_lift']]
            )
            fig3.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            fig3.update_traces(textposition='outside')
            fig3.update_xaxes(categoryorder='category ascending')
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            color_map = {
                "Awareness": "#1d3db0",       
                "Consideration": "#689dfa",   
                "Intent": "#0d1b42"           
            }

            fig_lift = go.Figure()
            for label in ['Awareness', 'Consideration', 'Intent']:
                label_df = quarterly_lift[quarterly_lift['answer_label'] == label].copy()

                # Format the qoq_change as % string for text
                label_df['qoq_label'] = label_df['qoq_change'].apply(
                    lambda x: f"+{x:.0%}" if x > 0 else (f"{x:.0%}" if pd.notna(x) else "")
                )

                fig_lift.add_trace(go.Bar(
                    x=label_df['quarter_year'],
                    y=label_df['relative_lift'],
                    name=f"{label}",
                    marker_color=color_map[label],
                    text=label_df['qoq_label'],
                    textposition='outside',
                    offsetgroup=label
                ))

            # Final layout settings
            fig_lift.update_layout(
                title="Average Lift by Specific Quarter (with QoQ Change as Labels)",
                xaxis_title="Quarter",
                yaxis_title="Average Lift (%)",
                yaxis_tickformat=".0%",
                yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                barmode='group',
                legend=dict(title=dict(text="Metric"),font=dict(size=14)),
                clickmode='event+select',
                margin=dict(t=50, b=40),
                font=dict(color="black")
            )

            st.plotly_chart(fig_lift, use_container_width=True)


        with tab5:
            fig4 = px.bar(
                quarterly_lift,
                x="quarter_year",
                y="exposed_rate",
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"exposed_rate": "Average Exposed Rate (%)", "answer_label": "Metric", "quarter_year": "Quarter"},
                title="Average Exposed Rates by Specific Quarter",
                text=[f"{v:.0%}" for v in quarterly_lift['exposed_rate']]
            )
            fig4.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            fig4.update_traces(textposition='outside')
            st.plotly_chart(fig4, use_container_width=True)
        
        with tab6:
            st.subheader("Average Lifts per Venue Type")

            # Group to get average lift by metric and venue type
            grouped = (
                filtered_venue_df
                .groupby(['answer_label', 'segmentation'])['relative_lift']
                .mean()
                .reset_index()
            )

            # Loop over metrics to plot 3 subplots
            for metric in ['Awareness', 'Consideration', 'Intent']:
                subset = grouped[grouped['answer_label'] == metric]
                subset = subset.sort_values(by='relative_lift', ascending=False)

                bar_df = subset[subset['segmentation'] != 'Campaign']
                benchmark_value = subset[subset['segmentation'] == 'Campaign']['relative_lift'].mean()

                fig5 = go.Figure()

                # Bar trace for all venue segmentations except campaign
                fig5.add_trace(go.Bar(
                    x=bar_df['segmentation'],
                    y=bar_df['relative_lift'],
                    name='Relative Lift (%)',
                    marker_color='rgb(118,170,248)',
                    text=[f"{v:.0%}" for v in bar_df['relative_lift']],
                    textposition='outside'
                ))
                # Add benchmark line
                fig5.add_trace(go.Scatter(
                    x=bar_df['segmentation'],
                    y=[benchmark_value] * len(bar_df),  # horizontal line across all x
                    mode='lines',
                    name='Campaign Lift',
                    line=dict(color='rgb(242,138,76)', width=5, dash='dash')
                ))
                fig5.update_layout(
                    title=f"{metric} Lift by Venue Type",
                    yaxis_title='Average Lift (%)',
                    xaxis_title='Venue Type',
                    barmode='group',
                    yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                    legend=dict(title=dict(text="Metric"),font=dict(size=14)),
                    font=dict(color="black")
                )
                st.plotly_chart(fig5, use_container_width=True)
        
        with tab7:
            st.subheader("Average Rates for Telecom")
            # add a note as to how many different studies, maybe how many different brands
            # Group to get average lift by metric and venue type
            telecom_grouped = (
                telecom_df
                .groupby(['segmentation', 'answer_label'])['relative_lift']
                .mean()
                .reset_index()
            )

            telecom_exposed = (
                telecom_df
                .groupby(['segmentation', 'answer_label'])['exposed_rate']
                .mean()
                .reset_index()
            )

            telecom_fig = px.bar(
                telecom_grouped,
                x=telecom_grouped["segmentation"],
                y=telecom_grouped["relative_lift"],
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"relative_lift": "Average Lift (%)", "answer_label": "Metric", "segmentation": "Overall Telecom Campaigns"},
                title="Average Lifts for our Telecom BLS Studies",
                text=[f"{v:.0%}" for v in telecom_grouped['relative_lift']]
            )
            telecom_fig.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            telecom_fig.update_traces(textposition='outside')
            st.plotly_chart(telecom_fig, use_container_width=True)

            telecom_exposed_fig = px.bar(
                telecom_exposed,
                x=telecom_exposed["segmentation"],
                y=telecom_exposed["exposed_rate"],
                color="answer_label",
                color_discrete_map={
                    "Awareness": "#1d3db0",       
                    "Consideration": "#689dfa",   
                    "Intent": "#0d1b42"           
                },
                barmode="group",
                labels={"exposed": "Average Exposed Rates", "answer_label": "Metric", "segmentation": "Overall Telecom Campaigns"},
                title="Average Exposed Rates for our Telecom BLS Studies",
                text=[f"{v:.0%}" for v in telecom_exposed['exposed_rate']]
            )
            telecom_exposed_fig.update_layout(yaxis=dict(range=[0, 1], showgrid=False, tickformat=".0%"),
                              legend=dict(font=dict(size=14)),
                              clickmode='event+select',
                              font=dict(color="black")
            )
            telecom_exposed_fig.update_traces(textposition='outside')
            st.plotly_chart(telecom_exposed_fig, use_container_width=True)

if __name__ == "__main__":
    main()
