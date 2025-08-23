# FILE NAME: app.py
# VERSION 14.0: "Grandmaster" UI Edition - Final, All Features Included, Complete Code

import streamlit as st
import pandas as pd
import os
import sys
import json
import plotly.express as px
import plotly.graph_objects as go

# --- Path setup and module imports ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from student_tools.study_planner import StudyPlanner
from data_ingestion.syllabus_data import ALL_SYLLABUS_DATA
# This import is now for the cache_data function
try:
    from statistical_analysis.analyze_history import analyze_historical_data
except ImportError:
    # A fallback function if the file doesn't exist, though it should
    def analyze_historical_data():
        print("Warning: analyze_history.py not found.")
        return {}

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="GATE-Astra Oracle", page_icon="🎧",
    layout="wide", initial_sidebar_state="expanded"
)

# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================
@st.cache_resource
def get_planner_instance(_iit_name):
    """Creates and caches a StudyPlanner instance."""
    try:
        planner = StudyPlanner(organizing_iit=_iit_name)
        if planner.prediction_df is None: planner._generate_prediction()
        return planner
    except Exception as e:
        st.error(f"Failed to initialize prediction engine: {e}")
        return None

@st.cache_data
def load_mock_questions(_iit_name):
    """Loads and caches mock questions."""
    filename = f"generation_results/mock_questions_{_iit_name.replace(' ', '_')}.json"
    if os.path.exists(filename):
        with open(filename, 'r') as f: return json.load(f)
    return []

@st.cache_data
def get_feature_data(_iit_name):
    """Loads the raw feature data for analysis."""
    try:
        from src.statistical_analysis.create_features import FeatureEngineerV2
        engineer = FeatureEngineerV2()
        features_df = engineer.create_master_feature_set(_iit_name)
        engineer.close()
        return features_df
    except Exception as e:
        print(f"Could not load feature data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_historical_patterns():
    """Loads the pre-calculated historical patterns file for comparisons."""
    patterns_path = "analysis_results/historical_patterns.json"
    if os.path.exists(patterns_path):
        with open(patterns_path, 'r') as f: return json.load(f)
    return {}

# ==============================================================================
# UI HELPER FUNCTIONS
# ==============================================================================
def render_gauge_chart(value, title):
    """Creates a Plotly gauge chart for an AI signal score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00f2ea"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }))
    fig.update_layout(height=200, margin=dict(l=30, r=30, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)", font_color=st.get_option("theme.textColor"))
    return fig

# ==============================================================================
# MAIN APP UI
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("GATE-Astra")
    st.markdown("##### The Exam Multiverse Oracle")
    st.markdown("---")
    
    iit_options = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IIT Roorkee", "IIT Guwahati", "IISc Bangalore"]
    selected_iit = st.selectbox("Select Organizing IIT", options=iit_options, index=3)

    if st.button("Generate Forecast", type="primary", use_container_width=True):
        st.session_state.prediction_ran = True
        st.session_state.selected_iit = selected_iit
    
    st.markdown("---")
    
    with st.expander("Edit Your Subject Strengths", expanded=True):
        all_subjects = sorted(list(set(item['subject'] for item in ALL_SYLLABUS_DATA)))
        user_proficiency = {}
        for subject in all_subjects:
            if "DA -" in subject: continue
            user_proficiency[subject] = st.select_slider(
                f"{subject.replace('CS - ', '').replace('GA - ', '')}",
                options=["Weak", "Medium", "Strong"], value="Medium", key=f"prof_{subject}"
            )

# --- Main Display Area ---
st.header("GATE-Astra Forecast Dashboard")

if not st.session_state.get('prediction_ran', False):
    st.info("Select an organizing IIT from the sidebar and click **'Generate Forecast'** to awaken the Oracle.")
else:
    iit_to_predict = st.session_state.selected_iit
    
    with st.spinner("Synthesizing predictions from the Multiverse..."):
        planner = get_planner_instance(iit_to_predict)
        features_df = get_feature_data(iit_to_predict)
        patterns = get_historical_patterns()

    if planner is None:
        st.error("The prediction engine could not be initialized.")
    else:
        prediction_df = planner.prediction_df
        st.subheader(f"Oracle Forecast for: **{iit_to_predict}**")

        tab1, tab2, tab3 = st.tabs(["Main Forecast", "Personalized Plan", "AI Question Bank"])

        with tab1:
            st.markdown("#### Overall Subject Weightage Forecast")
            if prediction_df is not None and not prediction_df.empty:
                display_df = prediction_df.copy()
                display_df['Weightage'] = (display_df['predicted_weight'].astype(float) * 100)
                
                iit_past_biases = patterns.get('iit_biases', {}).get(iit_to_predict, {})
                overall_avg_probs = patterns.get('overall_subject_probabilities', {})
                iit_historical_avg = {s: (overall_avg_probs.get(s, 0) + iit_past_biases.get(s, 0)) * 100 for s in overall_avg_probs}
                
                display_df[f'{iit_to_predict} Past Avg.'] = display_df['subject'].map(iit_historical_avg).fillna(0)
                display_df['Overall Hist. Avg.'] = display_df['subject'].map(overall_avg_probs).fillna(0) * 100
                display_df.rename(columns={'Weightage': f'Oracle Forecast'}, inplace=True)

                with st.container(border=True):
                    st.markdown(f"**Plot 1: Forecast vs. Historical Patterns**")
                    st.caption(f"""
                    **What this shows:** This chart provides a deep comparative analysis.
                    - **Oracle Forecast:** The AI's prediction for the upcoming exam by **{iit_to_predict}**.
                    - **{iit_to_predict} Past Avg.:** How this specific IIT has weighted subjects in their past papers.
                    - **Overall Hist. Avg.:** The average weightage across all exams by all IITs.
                    
                    **How to use this:** Look for subjects where the **Oracle Forecast** bar is significantly different from the others. This highlights where the AI predicts a strategic shift, giving you a competitive edge.
                    """, unsafe_allow_html=True)
                    plot_df_compare = display_df[display_df[f'Oracle Forecast'] > 1.0].melt(
                        id_vars='subject', value_vars=[f'Oracle Forecast', f'{iit_to_predict} Past Avg.', 'Overall Hist. Avg.'],
                        var_name='Analysis Type', value_name='Percentage'
                    )
                    fig_compare = px.bar(plot_df_compare, x='subject', y='Percentage', color='Analysis Type', barmode='group')
                    fig_compare.update_xaxes(tickangle=45, title_text="")
                    fig_compare.update_layout(legend_title="Analysis Type", yaxis_title="Predicted Weightage (%)")
                    st.plotly_chart(fig_compare, use_container_width=True)

                with st.container(border=True):
                    st.markdown("**Plot 2: Detailed Weightage Distribution**")
                    st.caption("This treemap provides another view of the Oracle's forecast. The size of each rectangle is proportional to its predicted weightage, making it easy to see the most dominant subjects at a glance.")
                    fig_treemap = px.treemap(
                        display_df[display_df['Oracle Forecast'] > 0.5], path=[px.Constant("All Subjects"), 'subject'],
                        values='Oracle Forecast', color='Oracle Forecast', color_continuous_scale='Mint'
                    )
                    fig_treemap.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                    st.plotly_chart(fig_treemap, use_container_width=True)

                st.markdown("#### Chain of Reasoning")
                st.caption("This section deconstructs the forecast, showing the underlying AI signals that influenced the prediction for the top subjects.")
                merged_df = pd.merge(display_df, features_df, on='subject', how='left')
                numeric_cols_to_agg = ['Oracle Forecast', 'monte_carlo_prob', 'nptel_heat_score', 'prof_bias_score']
                subject_level_features = merged_df.groupby('subject')[numeric_cols_to_agg].mean().reset_index()
                top_subjects_df = subject_level_features.nlargest(5, 'Oracle Forecast')
                
                for index, row in top_subjects_df.iterrows():
                    with st.container(border=True):
                        st.markdown(f"##### {row['subject']}")
                        col1, col2, col3 = st.columns(3)
                        subject_key = row['subject'].replace(" ", "_")
                        with col1: st.plotly_chart(render_gauge_chart(row['monte_carlo_prob'], "Statistical Score"), use_container_width=True, key=f"gauge_mc_{subject_key}")
                        with col2: st.plotly_chart(render_gauge_chart(row['nptel_heat_score'], "NPTEL Heat Score"), use_container_width=True, key=f"gauge_nptel_{subject_key}")
                        with col3: st.plotly_chart(render_gauge_chart(row['prof_bias_score'], "Professor Bias Score"), use_container_width=True, key=f"gauge_prof_{subject_key}")

        with tab2:
            st.header("Your Personalized Study Plan")
            personalized_plan_df = planner.create_plan(user_proficiency_map=user_proficiency)
            
            with st.container(border=True):
                st.markdown("**Plot 3: Strategic Study Matrix**")
                st.caption("""
                This plot helps you decide where to focus based on the forecast and your proficiency.
                - **Top-Right (High-Yield):** High importance, low proficiency. Your top priority.
                - **Bottom-Right (Strongholds):** High importance, high proficiency. Master these to secure guaranteed marks.
                - **Top-Left (Opportunity):** Low importance, low proficiency. Cover if you have extra time.
                - **Bottom-Left (Backburners):** Low importance, high proficiency. A brief revision is sufficient.
                """)
                matrix_df = personalized_plan_df.copy()
                matrix_df['Proficiency Score'] = matrix_df['proficiency_text'].map({"Weak": 1, "Medium": 2, "Strong": 3})
                matrix_df['Subject Importance'] = matrix_df['predicted_weight']
                fig_matrix = px.scatter(
                    matrix_df, x="Proficiency Score", y="Subject Importance", size="study_time_percent",
                    color="subject", hover_name="subject", size_max=60
                )
                fig_matrix.update_layout(xaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Weak', 'Medium', 'Strong'], title="Your Proficiency"), yaxis=dict(title="Predicted Subject Importance"))
                fig_matrix.add_vline(x=1.5, line_width=1, line_dash="dash", line_color="grey"); fig_matrix.add_vline(x=2.5, line_width=1, line_dash="dash", line_color="grey")
                fig_matrix.add_hline(y=matrix_df['Subject Importance'].median(), line_width=1, line_dash="dash", line_color="grey")
                st.plotly_chart(fig_matrix, use_container_width=True)

            with st.container(border=True):
                st.markdown("**Plot 4: Recommended Time Allocation**")
                st.caption("A clear, proportional breakdown of how you should allocate your study time among your highest-priority subjects, as determined by the Strategic Matrix.")
                plan_to_plot = personalized_plan_df[personalized_plan_df['study_time_percent'] > 2.0]
                fig_donut = px.pie(
                    plan_to_plot, values='study_time_percent', names='subject',
                    title='Study Time Breakdown for High-Priority Subjects', hole=.4
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        with tab3:
            st.header("AI-Generated Question Bank")
            st.write(f"Challenging questions generated in the predicted style of {iit_to_predict}.")
            mock_questions = load_mock_questions(iit_to_predict)
            if mock_questions:
                topics_in_bank = sorted(list(set([q.get('topic_name', 'Unknown') for q in mock_questions])))
                selected_topic = st.selectbox("Filter by topic:", ["All"] + topics_in_bank)
                
                for i, q in enumerate(mock_questions):
                    if selected_topic == "All" or q.get('topic_name') == selected_topic:
                        with st.container(border=True):
                            st.markdown(f"**Question {i+1}:** {q.get('question_text', '')}")
                            st.caption(f"Topic: {q.get('topic_name', 'N/A')}")
                            
                            parsed_options = {opt.get('option_label', ''): opt.get('option_text', '') for opt in q.get('options', [])}
                            if parsed_options:
                                user_answer = st.radio("Select your answer:", list(parsed_options.keys()), key=f"q_{i}", format_func=lambda x: f"{x}) {parsed_options.get(x, '')}", horizontal=True)
                                with st.expander("Show Answer & Explanation"):
                                    correct_key = q.get('answer_key')
                                    if user_answer == correct_key: st.success(f"Correct!")
                                    else: st.error(f"Incorrect. The correct answer is **{correct_key}**.")
                                    st.info(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")
            else:
                st.warning("No mock question bank found for this IIT. Please run the generation script first.")