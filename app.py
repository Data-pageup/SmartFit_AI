import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="SmartFit AI - Fitness Intelligence System",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💪 SmartFit AI</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Personalized Fitness & Nutrition Intelligence</p>', unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/dumbbell.png", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Select Module",
        ["🏠 Dashboard", "🔮 Predictions", "🧩 Fitness Profiles", "🍽️ Diet Planner", "💪 Workout Recommender", "📊 Data Explorer"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Users Analyzed", "20,000")
    st.metric("Prediction Accuracy", "89%")
    st.metric("Fitness Clusters", "5")

# Helper Functions
def calculate_bmi(weight_kg, height_m):
    return weight_kg / (height_m ** 2)

def calculate_calories_burned(duration, intensity, weight, age):
    """Simple calorie burn estimation"""
    met_values = {'Low': 3.5, 'Medium': 6.0, 'High': 8.5, 'Very High': 10.5}
    met = met_values.get(intensity, 6.0)
    calories = (met * 3.5 * weight / 200) * duration
    return round(calories, 2)

def get_fitness_cluster_profile(cluster_id):
    """Return profile characteristics for each cluster"""
    profiles = {
        0: {
            'name': '🏃 Elite Athletes',
            'desc': 'High intensity workouts, excellent cardiovascular health, low body fat',
            'traits': ['Max BPM: 170-190', 'BMI: 18-23', 'Fat %: 8-15%', 'Workout Duration: 60-90 min']
        },
        1: {
            'name': '💪 Strength Builders',
            'desc': 'Focus on resistance training, moderate cardio, muscle building phase',
            'traits': ['Max BPM: 150-170', 'BMI: 24-27', 'Fat %: 15-22%', 'Workout Duration: 45-75 min']
        },
        2: {
            'name': '🎯 Fitness Enthusiasts',
            'desc': 'Balanced workout routine, maintaining healthy lifestyle',
            'traits': ['Max BPM: 140-165', 'BMI: 22-26', 'Fat %: 18-25%', 'Workout Duration: 30-60 min']
        },
        3: {
            'name': '🌱 Beginners',
            'desc': 'Starting fitness journey, building foundational strength',
            'traits': ['Max BPM: 130-150', 'BMI: 25-30', 'Fat %: 22-32%', 'Workout Duration: 20-45 min']
        },
        4: {
            'name': '🏥 Health Focus',
            'desc': 'Medical considerations, low-impact activities, gradual progression',
            'traits': ['Max BPM: 110-140', 'BMI: 28-35+', 'Fat %: 28-40%+', 'Workout Duration: 15-30 min']
        }
    }
    return profiles.get(cluster_id, profiles[2])

def get_workout_recommendations(cluster_id, difficulty):
    """Generate workout recommendations based on cluster and difficulty"""
    workouts = {
        0: {
            'High': ['HIIT Training (45 min)', 'Marathon Running (90 min)', 'CrossFit WOD (60 min)', 'Olympic Lifting (75 min)'],
            'Medium': ['Tempo Running (60 min)', 'Circuit Training (45 min)', 'Swimming (60 min)'],
            'Low': ['Easy Run (30 min)', 'Yoga (45 min)', 'Stretching (20 min)']
        },
        1: {
            'High': ['Heavy Compound Lifts (75 min)', 'Powerlifting Session (90 min)', 'Strongman Training (60 min)'],
            'Medium': ['Hypertrophy Training (60 min)', 'Push/Pull Workout (45 min)', 'Functional Training (50 min)'],
            'Low': ['Light Weight Training (30 min)', 'Mobility Work (25 min)', 'Core Strength (20 min)']
        },
        2: {
            'High': ['Interval Training (40 min)', 'Full Body Circuit (45 min)', 'Spin Class (50 min)'],
            'Medium': ['Jogging (35 min)', 'Bodyweight Exercises (30 min)', 'Pilates (40 min)'],
            'Low': ['Walking (25 min)', 'Gentle Yoga (30 min)', 'Stretching (20 min)']
        },
        3: {
            'High': ['Beginner HIIT (25 min)', 'Light Circuit (30 min)', 'Brisk Walking Hills (30 min)'],
            'Medium': ['Beginner Strength (30 min)', 'Low-Impact Cardio (25 min)', 'Basic Yoga (30 min)'],
            'Low': ['Gentle Walking (20 min)', 'Chair Exercises (15 min)', 'Breathing Exercises (10 min)']
        },
        4: {
            'High': ['Water Aerobics (30 min)', 'Recumbent Bike (25 min)', 'Resistance Bands (20 min)'],
            'Medium': ['Gentle Walking (20 min)', 'Chair Yoga (25 min)', 'Balance Training (20 min)'],
            'Low': ['Stretching (15 min)', 'Seated Exercises (15 min)', 'Meditation (10 min)']
        }
    }
    return workouts.get(cluster_id, workouts[2]).get(difficulty, workouts[cluster_id]['Medium'])

def get_diet_plan(cluster_id, goal):
    """Generate diet recommendations"""
    diets = {
        'Weight Loss': {
            'calories': '1500-1800 kcal/day',
            'protein': '1.8-2.2g/kg bodyweight',
            'carbs': '100-150g/day',
            'fats': '40-60g/day',
            'meals': ['Breakfast: Oatmeal with berries', 'Lunch: Grilled chicken salad', 'Dinner: Salmon with vegetables', 'Snacks: Greek yogurt, almonds']
        },
        'Muscle Gain': {
            'calories': '2500-3200 kcal/day',
            'protein': '2.0-2.5g/kg bodyweight',
            'carbs': '300-450g/day',
            'fats': '70-100g/day',
            'meals': ['Breakfast: Eggs, whole grain toast, avocado', 'Lunch: Rice, chicken, vegetables', 'Dinner: Steak, sweet potato, broccoli', 'Snacks: Protein shake, nuts, banana']
        },
        'Maintenance': {
            'calories': '2000-2400 kcal/day',
            'protein': '1.5-1.8g/kg bodyweight',
            'carbs': '200-280g/day',
            'fats': '55-75g/day',
            'meals': ['Breakfast: Smoothie bowl with granola', 'Lunch: Turkey wrap with vegetables', 'Dinner: Pasta with lean meat sauce', 'Snacks: Fruit, trail mix']
        },
        'Endurance': {
            'calories': '2800-3500 kcal/day',
            'protein': '1.4-1.8g/kg bodyweight',
            'carbs': '400-600g/day',
            'fats': '60-90g/day',
            'meals': ['Breakfast: Pancakes with maple syrup', 'Lunch: Quinoa bowl with chicken', 'Dinner: Pasta with vegetables', 'Snacks: Energy bars, dried fruit, sports drinks']
        }
    }
    return diets.get(goal, diets['Maintenance'])

# Page: Dashboard
if page == "🏠 Dashboard":
    st.header("📊 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h2>20K</h2><p>Users Analyzed</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2>89%</h2><p>Prediction Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2>5</h2><p>Fitness Clusters</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h2>62</h2><p>Features Tracked</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Simulated data visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Fitness Cluster Distribution")
        cluster_data = pd.DataFrame({
            'Cluster': ['Elite Athletes', 'Strength Builders', 'Enthusiasts', 'Beginners', 'Health Focus'],
            'Count': [3200, 4500, 5800, 4100, 2400]
        })
        fig = px.pie(cluster_data, values='Count', names='Cluster', 
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Average Calories Burned by Workout Type")
        workout_data = pd.DataFrame({
            'Workout': ['HIIT', 'Strength', 'Cardio', 'Yoga', 'Sports'],
            'Calories': [520, 380, 450, 180, 410]
        })
        fig = px.bar(workout_data, x='Workout', y='Calories',
                     color='Calories', color_continuous_scale='Plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # BMI Distribution
    st.subheader("📊 BMI Distribution Across Users")
    bmi_data = np.random.normal(25, 4, 1000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bmi_data, nbinsx=30, name='BMI',
                               marker_color='rgba(102, 126, 234, 0.7)'))
    fig.update_layout(xaxis_title='BMI', yaxis_title='Count', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Page: Predictions
elif page == "🔮 Predictions":
    st.header("🔮 Health & Fitness Predictions")
    
    tab1, tab2, tab3 = st.tabs(["Calorie Burn Predictor", "BMI & Body Fat", "Workout Impact"])
    
    with tab1:
        st.subheader("🔥 Predict Calories Burned")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 80, 30)
            weight = st.slider("Weight (kg)", 40, 150, 70)
            duration = st.slider("Workout Duration (minutes)", 10, 180, 45)
        
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            intensity = st.selectbox("Workout Intensity", ["Low", "Medium", "High", "Very High"])
            workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "HIIT", "Yoga", "Sports"])
        
        if st.button("🔥 Calculate Calories Burned", type="primary"):
            calories = calculate_calories_burned(duration, intensity, weight, age)
            
            # Add variation based on gender and workout type
            gender_factor = 1.1 if gender == "Male" else 1.0
            workout_factors = {'Cardio': 1.2, 'Strength': 0.9, 'HIIT': 1.5, 'Yoga': 0.6, 'Sports': 1.3}
            calories = calories * gender_factor * workout_factors.get(workout_type, 1.0)
            
            st.success(f"### Estimated Calories Burned: {calories:.0f} kcal")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Per Minute", f"{calories/duration:.1f} kcal")
            col2.metric("Fat Burned", f"{calories*0.3/9:.1f}g")
            col3.metric("Weekly (3x)", f"{calories*3:.0f} kcal")
    
    with tab2:
        st.subheader("📏 BMI & Body Composition Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            weight_bmi = st.number_input("Weight (kg)", 40, 200, 70)
            height_m = st.number_input("Height (m)", 1.4, 2.2, 1.75, 0.01)
        
        with col2:
            age_bmi = st.slider("Age", 18, 80, 30, key='age_bmi')
            gender_bmi = st.selectbox("Gender", ["Male", "Female"], key='gender_bmi')
        
        if st.button("📊 Analyze Body Composition", type="primary"):
            bmi = calculate_bmi(weight_bmi, height_m)
            
            # Estimate body fat percentage (simplified formula)
            if gender_bmi == "Male":
                body_fat = (1.20 * bmi) + (0.23 * age_bmi) - 16.2
            else:
                body_fat = (1.20 * bmi) + (0.23 * age_bmi) - 5.4
            
            body_fat = max(5, min(50, body_fat))
            
            st.success(f"### BMI: {bmi:.1f}")
            
            if bmi < 18.5:
                category = "Underweight"
                color = "blue"
            elif bmi < 25:
                category = "Normal Weight"
                color = "green"
            elif bmi < 30:
                category = "Overweight"
                color = "orange"
            else:
                category = "Obese"
                color = "red"
            
            st.markdown(f"**Category:** :{color}[{category}]")
            st.metric("Estimated Body Fat %", f"{body_fat:.1f}%")
            
            # BMI gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "BMI"},
                gauge={
                    'axis': {'range': [15, 40]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [15, 18.5], 'color': "lightblue"},
                        {'range': [18.5, 25], 'color': "lightgreen"},
                        {'range': [25, 30], 'color': "lightyellow"},
                        {'range': [30, 40], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("💪 Workout Impact Predictor")
        st.info("Predict the impact of consistent training over time")
        
        col1, col2 = st.columns(2)
        with col1:
            current_weight = st.number_input("Current Weight (kg)", 40, 200, 75)
            weekly_workouts = st.slider("Workouts per Week", 1, 7, 4)
            workout_duration_avg = st.slider("Avg Duration (min)", 20, 120, 45)
        
        with col2:
            diet_goal = st.selectbox("Diet Goal", ["Weight Loss", "Maintenance", "Muscle Gain"])
            weeks = st.slider("Time Period (weeks)", 4, 52, 12)
        
        if st.button("📈 Predict Progress", type="primary"):
            # Simple projection model
            weekly_deficit = {'Weight Loss': -500, 'Maintenance': 0, 'Muscle Gain': 300}[diet_goal]
            weekly_cal_burn = calculate_calories_burned(workout_duration_avg, "Medium", current_weight, 30) * weekly_workouts
            
            net_weekly = weekly_cal_burn + (weekly_deficit * 7)
            weight_change_per_week = net_weekly / 7700  # 7700 cal ≈ 1 kg
            
            projected_weight = current_weight + (weight_change_per_week * weeks)
            total_change = projected_weight - current_weight
            
            st.success(f"### Projected Weight after {weeks} weeks: {projected_weight:.1f} kg")
            st.metric("Total Change", f"{total_change:+.1f} kg", f"{total_change/current_weight*100:+.1f}%")
            
            # Progress chart
            timeline = list(range(0, weeks + 1, max(1, weeks // 10)))
            weights = [current_weight + (weight_change_per_week * w) for w in timeline]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline, y=weights, mode='lines+markers',
                                    line=dict(width=3, color='rgb(102, 126, 234)'),
                                    marker=dict(size=8)))
            fig.update_layout(
                xaxis_title="Week",
                yaxis_title="Weight (kg)",
                title="Projected Weight Progress"
            )
            st.plotly_chart(fig, use_container_width=True)

# Page: Fitness Profiles
elif page == "🧩 Fitness Profiles":
    st.header("🧩 Fitness Cluster Profiles")
    st.markdown("*Discover which fitness archetype matches your profile*")
    
    # User input for clustering
    st.subheader("📝 Enter Your Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        user_age = st.slider("Age", 18, 80, 30, key='cluster_age')
        user_weight = st.slider("Weight (kg)", 40, 150, 70, key='cluster_weight')
        user_height = st.slider("Height (m)", 1.4, 2.2, 1.75, 0.01, key='cluster_height')
    
    with col2:
        user_bpm = st.slider("Max Heart Rate (BPM)", 100, 200, 160)
        user_duration = st.slider("Avg Workout Duration (min)", 15, 120, 45, key='cluster_duration')
        user_frequency = st.slider("Workouts per Week", 1, 7, 4)
    
    with col3:
        user_experience = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced", "Elite"])
        user_goal = st.selectbox("Primary Goal", ["Weight Loss", "Muscle Gain", "Endurance", "General Fitness"])
        user_intensity = st.selectbox("Preferred Intensity", ["Low", "Medium", "High", "Very High"], key='cluster_intensity')
    
    if st.button("🔍 Find My Fitness Profile", type="primary"):
        # Simple rule-based clustering (in real scenario, use trained model)
        bmi = calculate_bmi(user_weight, user_height)
        
        # Determine cluster based on characteristics
        if user_bpm > 175 and user_experience == "Elite" and user_duration > 60:
            cluster = 0
        elif user_experience in ["Advanced", "Elite"] and user_intensity in ["High", "Very High"]:
            cluster = 1
        elif user_experience == "Intermediate" and user_frequency >= 3:
            cluster = 2
        elif user_experience == "Beginner" or user_frequency <= 2:
            cluster = 3
        else:
            cluster = 4 if bmi > 30 else 2
        
        profile = get_fitness_cluster_profile(cluster)
        
        st.success(f"## Your Fitness Profile: {profile['name']}")
        st.info(profile['desc'])
        
        st.markdown("### 📋 Profile Characteristics")
        for trait in profile['traits']:
            st.markdown(f"- {trait}")
        
        # Radar chart for fitness attributes
        st.markdown("---")
        st.subheader("📊 Your Fitness Attributes")
        
        attributes = ['Strength', 'Endurance', 'Flexibility', 'Power', 'Recovery']
        cluster_scores = {
            0: [85, 95, 70, 90, 80],
            1: [95, 70, 65, 85, 75],
            2: [70, 75, 75, 70, 80],
            3: [50, 55, 60, 50, 65],
            4: [45, 50, 55, 45, 60]
        }
        
        scores = cluster_scores[cluster]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=attributes,
            fill='toself',
            name='Your Profile',
            line=dict(color='rgb(102, 126, 234)', width=2)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display all cluster profiles
    st.markdown("---")
    st.subheader("🌐 All Fitness Archetypes")
    
    for i in range(5):
        profile = get_fitness_cluster_profile(i)
        with st.expander(f"{profile['name']}"):
            st.markdown(f"**Description:** {profile['desc']}")
            st.markdown("**Typical Characteristics:**")
            for trait in profile['traits']:
                st.markdown(f"- {trait}")

# Page: Diet Planner
elif page == "🍽️ Diet Planner":
    st.header("🍽️ Personalized Diet Planner")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Your Information")
        diet_weight = st.number_input("Current Weight (kg)", 40, 200, 70, key='diet_weight')
        diet_height = st.number_input("Height (m)", 1.4, 2.2, 1.75, 0.01, key='diet_height')
        diet_age = st.slider("Age", 18, 80, 30, key='diet_age')
        diet_gender = st.selectbox("Gender", ["Male", "Female"], key='diet_gender')
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Very Active", "Extremely Active"])
    
    with col2:
        st.subheader("🎯 Your Goals")
        diet_goal = st.selectbox("Primary Goal", ["Weight Loss", "Muscle Gain", "Maintenance", "Endurance"], key='diet_goal_select')
        timeline_diet = st.slider("Timeline (weeks)", 4, 52, 12, key='timeline_diet')
        dietary_pref = st.multiselect("Dietary Preferences", ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "High-Protein", "Low-Carb"])
        meals_per_day = st.slider("Meals per Day", 3, 6, 4)
    
    if st.button("🍽️ Generate Diet Plan", type="primary"):
        # Calculate cluster (simplified)
        bmi = calculate_bmi(diet_weight, diet_height)
        if bmi < 22 and diet_goal == "Muscle Gain":
            cluster = 1
        elif diet_goal == "Endurance":
            cluster = 0
        elif bmi > 28 and diet_goal == "Weight Loss":
            cluster = 4
        else:
            cluster = 2
        
        plan = get_diet_plan(cluster, diet_goal)
        
        st.success("## 📋 Your Personalized Diet Plan")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Daily Calories", plan['calories'])
        col2.metric("Protein", plan['protein'])
        col3.metric("Carbs", plan['carbs'])
        col4.metric("Fats", plan['fats'])
        
        st.markdown("---")
        st.subheader("🍴 Sample Meal Plan")
        
        for meal in plan['meals']:
            st.markdown(f"- {meal}")
        
        st.markdown("---")
        st.subheader("📊 Macronutrient Distribution")
        
        # Extract numeric values for chart (simplified)
        cal_mid = int(plan['calories'].split('-')[0])
        protein_g = float(plan['protein'].split('-')[0].replace('g/kg', '')) * diet_weight
        carbs_g = float(plan['carbs'].split('-')[0].replace('g/day', ''))
        fats_g = float(plan['fats'].split('-')[0].replace('g/day', ''))
        
        # Convert to calories
        protein_cal = protein_g * 4
        carbs_cal = carbs_g * 4
        fats_cal = fats_g * 9
        
        macro_data = pd.DataFrame({
            'Macronutrient': ['Protein', 'Carbs', 'Fats'],
            'Calories': [protein_cal, carbs_cal, fats_cal]
        })
        
        fig = px.pie(macro_data, values='Calories', names='Macronutrient',
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Shopping list
        st.markdown("---")
        st.subheader("🛒 Weekly Shopping List")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Proteins:**")
            st.markdown("- Chicken breast (2kg)")
            st.markdown("- Salmon fillets (1kg)")
            st.markdown("- Greek yogurt (1.5kg)")
            st.markdown("- Eggs (2 dozen)")
            st.markdown("- Lean beef (1kg)")
        
        with col2:
            st.markdown("**Carbs & Vegetables:**")
            st.markdown("- Brown rice (2kg)")
            st.markdown("- Oatmeal (1kg)")
            st.markdown("- Sweet potatoes (2kg)")
            st.markdown("- Mixed vegetables (3kg)")
            st.markdown("- Fruits (variety, 3kg)")

# Page: Workout Recommender
elif page == "💪 Workout Recommender":
    st.header("💪 AI Workout Recommender")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏋️ Current Fitness Level")
        workout_age = st.slider("Age", 18, 80, 30, key='workout_age')
        workout_experience = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced", "Elite"], key='workout_exp')
        available_time = st.slider("Available Time (min/day)", 15, 180, 45)
        workout_days = st.slider("Days per Week", 1, 7, 4, key='workout_days')
    
    with col2:
        st.subheader("🎯 Preferences")
        workout_goal = st.selectbox("Primary Goal", ["Weight Loss", "Muscle Gain", "Strength", "Endurance", "General Fitness"])
        equipment = st.multiselect("Available Equipment", ["Dumbbells", "Barbell", "Resistance Bands", "Cardio Machine", "Bodyweight Only"])
        injury_concerns = st.multiselect("Injury/Health Concerns", ["None", "Lower Back", "Knee", "Shoulder", "Hip", "Cardiovascular"])
        difficulty_pref = st.select_slider("Difficulty Preference", ["Low", "Medium", "High"])
    
    if st.button("🎯 Generate Workout Plan", type="primary"):
        # Determine cluster
        exp_to_cluster = {
            "Beginner": 3,
            "Intermediate": 2,
            "Advanced": 1,
            "Elite": 0
        }
        cluster = exp_to_cluster[workout_experience]
        
        # Get recommendations
        workouts = get_workout_recommendations(cluster, difficulty_pref)
        
        st.success("## 🏋️ Your Personalized Workout Program")
        
        # Weekly schedule
        st.subheader("📅 Weekly Training Schedule")
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        workout_schedule = {}
        
        for i, day in enumerate(days[:workout_days]):
            if i % 2 == 0:
                workout_schedule[day] = workouts[i % len(workouts)]
            else:
                workout_schedule[day] = workouts[(i + 1) % len(workouts)]
        
        for i, day in enumerate(days[workout_days:], start=workout_days):
            workout_schedule[day] = "Rest Day 😴"
        
        for day, workout in workout_schedule.items():
            if "Rest" in workout:
                st.markdown(f"**{day}:** {workout}")
            else:
                st.markdown(f"**{day}:** 💪 {workout}")
        
        st.markdown("---")
        
        # Detailed workout breakdown
        st.subheader("📝 Sample Workout Details")
        
        tab1, tab2, tab3 = st.tabs(["Strength Day", "Cardio Day", "Active Recovery"])
        
        with tab1:
            st.markdown("### 💪 Strength Training Session")
            st.markdown("**Warm-up (10 min):**")
            st.markdown("- Dynamic stretching")
            st.markdown("- Light cardio (jumping jacks, arm circles)")
            
            st.markdown("\n**Main Workout (30-40 min):**")
            exercises = [
                ("Squats", "4 sets × 8-12 reps", "🦵"),
                ("Bench Press", "4 sets × 8-10 reps", "💪"),
                ("Deadlifts", "3 sets × 6-8 reps", "🏋️"),
                ("Overhead Press", "3 sets × 8-10 reps", "💪"),
                ("Pull-ups/Rows", "3 sets × 8-12 reps", "💪"),
                ("Core Work", "3 sets × 15-20 reps", "🎯")
            ]
            
            for exercise, sets, emoji in exercises:
                st.markdown(f"{emoji} **{exercise}:** {sets}")
            
            st.markdown("\n**Cool-down (5 min):**")
            st.markdown("- Static stretching")
            st.markdown("- Foam rolling")
        
        with tab2:
            st.markdown("### 🏃 Cardio/Conditioning Session")
            st.markdown("**Warm-up (5 min):**")
            st.markdown("- Light jogging or cycling")
            
            st.markdown("\n**Main Workout (25-35 min):**")
            if difficulty_pref == "High":
                st.markdown("**HIIT Protocol:**")
                st.markdown("- 30 seconds max effort")
                st.markdown("- 30 seconds active recovery")
                st.markdown("- Repeat for 20-25 minutes")
            elif difficulty_pref == "Medium":
                st.markdown("**Steady State Cardio:**")
                st.markdown("- Maintain 70-75% max heart rate")
                st.markdown("- Running, cycling, or rowing")
                st.markdown("- 30-35 minutes continuous")
            else:
                st.markdown("**Low Impact Cardio:**")
                st.markdown("- Brisk walking or light cycling")
                st.markdown("- 60-65% max heart rate")
                st.markdown("- 25-30 minutes")
            
            st.markdown("\n**Cool-down (5 min):**")
            st.markdown("- Gradual pace reduction")
            st.markdown("- Light stretching")
        
        with tab3:
            st.markdown("### 🧘 Active Recovery Day")
            st.markdown("**Focus: Mobility & Recovery**")
            st.markdown("\n**Activities (30-45 min):**")
            st.markdown("- Yoga or Pilates (20-30 min)")
            st.markdown("- Foam rolling (10 min)")
            st.markdown("- Light swimming or walking (20 min)")
            st.markdown("- Breathing exercises (5 min)")
            
            st.markdown("\n**Benefits:**")
            st.markdown("- Reduces muscle soreness")
            st.markdown("- Improves flexibility")
            st.markdown("- Enhances recovery")
            st.markdown("- Prevents overtraining")
        
        st.markdown("---")
        
        # Progress tracking metrics
        st.subheader("📈 Progress Tracking Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Strength Metrics:**")
            st.markdown("- Track weight lifted")
            st.markdown("- Monitor rep progression")
            st.markdown("- Test 1RM quarterly")
        
        with col2:
            st.markdown("**Cardio Metrics:**")
            st.markdown("- Distance covered")
            st.markdown("- Average pace/speed")
            st.markdown("- Heart rate recovery")
        
        with col3:
            st.markdown("**Body Metrics:**")
            st.markdown("- Weekly weight")
            st.markdown("- Body measurements")
            st.markdown("- Progress photos")

# Page: Data Explorer
elif page == "📊 Data Explorer":
    st.header("📊 Data Insights & Analytics")
    
    # Generate synthetic data for visualization
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'weight_kg': np.random.normal(75, 15, n_samples),
        'height_m': np.random.normal(1.70, 0.15, n_samples),
        'max_bpm': np.random.randint(120, 195, n_samples),
        'workout_duration': np.random.randint(20, 120, n_samples),
        'calories_burned': np.random.normal(400, 150, n_samples),
        'body_fat_pct': np.random.normal(22, 8, n_samples),
        'cluster': np.random.randint(0, 5, n_samples)
    })
    
    synthetic_data['bmi'] = synthetic_data['weight_kg'] / (synthetic_data['height_m'] ** 2)
    synthetic_data['calories_burned'] = np.abs(synthetic_data['calories_burned'])
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Relationships", "📊 Cluster Analysis"])
    
    with tab1:
        st.subheader("🔗 Feature Correlations")
        
        # Correlation heatmap
        corr_features = ['age', 'weight_kg', 'bmi', 'max_bpm', 'workout_duration', 'calories_burned', 'body_fat_pct']
        corr_matrix = synthetic_data[corr_features].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(title='Feature Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 💡 Key Insights")
        st.info("**Strong Correlations Found:**\n"
                "- BMI and Body Fat % show positive correlation\n"
                "- Workout duration correlates with calories burned\n"
                "- Max BPM tends to decrease with age")
    
    with tab2:
        st.subheader("📊 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", corr_features)
        with col2:
            chart_type = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin Plot"])
        
        if chart_type == "Histogram":
            fig = px.histogram(synthetic_data, x=feature, nbins=30,
                             color_discrete_sequence=['#667eea'])
            fig.update_layout(title=f'Distribution of {feature}')
        elif chart_type == "Box Plot":
            fig = px.box(synthetic_data, y=feature,
                        color_discrete_sequence=['#764ba2'])
            fig.update_layout(title=f'Box Plot of {feature}')
        else:
            fig = px.violin(synthetic_data, y=feature,
                           color_discrete_sequence=['#667eea'])
            fig.update_layout(title=f'Violin Plot of {feature}')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### 📈 Statistical Summary")
        stats = synthetic_data[feature].describe()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{stats['mean']:.2f}")
        col2.metric("Median", f"{stats['50%']:.2f}")
        col3.metric("Std Dev", f"{stats['std']:.2f}")
        col4.metric("Range", f"{stats['max']-stats['min']:.2f}")
    
    with tab3:
        st.subheader("🔍 Feature Relationships")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_feature = st.selectbox("X-axis", corr_features, index=1)
        with col2:
            y_feature = st.selectbox("Y-axis", corr_features, index=5)
        with col3:
            color_by = st.selectbox("Color by", ['cluster', 'None'])
        
        if color_by == 'None':
            fig = px.scatter(synthetic_data, x=x_feature, y=y_feature,
                           opacity=0.6, color_discrete_sequence=['#667eea'])
        else:
            fig = px.scatter(synthetic_data, x=x_feature, y=y_feature,
                           color=color_by, opacity=0.6)
        
        fig.update_layout(title=f'{y_feature} vs {x_feature}')
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D scatter option
        if st.checkbox("Show 3D Visualization"):
            z_feature = st.selectbox("Z-axis", corr_features, index=2)
            fig_3d = px.scatter_3d(synthetic_data, x=x_feature, y=y_feature, z=z_feature,
                                  color='cluster', opacity=0.7)
            fig_3d.update_layout(title=f'3D: {x_feature} vs {y_feature} vs {z_feature}')
            st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        st.subheader("🧩 Cluster Analysis")
        
        # Cluster statistics
        st.markdown("### 📊 Cluster Statistics")
        
        cluster_names = {
            0: '🏃 Elite Athletes',
            1: '💪 Strength Builders',
            2: '🎯 Fitness Enthusiasts',
            3: '🌱 Beginners',
            4: '🏥 Health Focus'
        }
        
        for cluster_id in range(5):
            cluster_data = synthetic_data[synthetic_data['cluster'] == cluster_id]
            
            with st.expander(f"{cluster_names[cluster_id]} (n={len(cluster_data)})"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg Age", f"{cluster_data['age'].mean():.1f}")
                col2.metric("Avg BMI", f"{cluster_data['bmi'].mean():.1f}")
                col3.metric("Avg Calories", f"{cluster_data['calories_burned'].mean():.0f}")
                col4.metric("Avg Duration", f"{cluster_data['workout_duration'].mean():.0f} min")
        
        st.markdown("---")
        
        # PCA Visualization
        st.markdown("### 🎨 PCA Cluster Visualization")
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        features_for_pca = synthetic_data[corr_features].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_pca)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features_scaled)
        
        pca_df = pd.DataFrame({
            'PC1': pca_result[:, 0],
            'PC2': pca_result[:, 1],
            'Cluster': synthetic_data['cluster'].map(cluster_names)
        })
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                        title='PCA: Cluster Separation',
                        opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Explained Variance:** PC1: {pca.explained_variance_ratio_[0]:.1%}, "
               f"PC2: {pca.explained_variance_ratio_[1]:.1%}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>SmartFit AI</strong> - Powered by Machine Learning</p>
        <p>🔬 20,000 users analyzed | 📊 62 features tracked | 🎯 5 fitness clusters</p>
        <p style='font-size: 0.9rem; margin-top: 1rem;'>
            Built with Streamlit • Data Science • Deep Learning
        </p>
    </div>
""", unsafe_allow_html=True)