import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth

# -----------------------------
# 1. User Credentials (hashed)
# -----------------------------
hashed_passwords = stauth.Hasher(['Olympia@123','Olympia@123','Olympia@123']).generate()

config = {
    'credentials': {
        'usernames': {
            'alice': {
                'name': 'Alice',
                'password': hashed_passwords[0]
            },
            'bob': {
                'name': 'Bob',
                'password': hashed_passwords[1]
            },
            'charlie': {
                'name': 'Charlie',
                'password': hashed_passwords[2]
            }
        }
    },
    'cookie': {
        'name': 'dashboard_cookie',
        'key': 'random_signature_key',
        'expiry_days': 1
    },
    'preauthorized': {}
}

# -----------------------------
# 2. Authenticator
# -----------------------------
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# -----------------------------
# 3. Background & Logo
# -----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpapers.com/images/hd/cream-color-background-r77rf8azq8e16x65.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("OLYMPIA EDU 2.png", width=500)

st.markdown("<h2 style='text-align: center; color: Black;'> Welcome to </h2>", unsafe_allow_html=True)

# -----------------------------
# 4. Login
# -----------------------------
# --- Custom Styling ---
# --- Custom Styling ---
st.markdown("""
    <style>
    div[data-testid="stForm"] {
        background-color: #f9f9f9;
        padding: 30px;
        border-radius: 15px;
        border: 2px solid #4CAF50;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
    }

    div[data-testid="stForm"] label {
        color: #333333 !important;
        font-weight: bold !important;
    }

    div[data-testid="stForm"] input {
        border-radius: 8px;
        border: 1px solid #4CAF50;
        background-color: white !important;
        color: black !important;
        caret-color: black !important;  /* 👈 Ensures the blinking cursor is visible */
    }

    button[kind="primary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }

    /* Optional: Style the Login title */
    .custom-login-title {
        color: teal;
        font-weight: bold;
        font-size: 26px;
        text-align: center;
        margin-bottom: -5px;
        margin-top: -10px;
    }
    </style>
""", unsafe_allow_html=True)



# --- Show Login title ONLY before successful login ---
if "authentication_status" not in st.session_state or st.session_state["authentication_status"] != True:
    st.markdown("<div class='custom-login-title'>Login</div>", unsafe_allow_html=True)

# --- Authenticator Login Form ---
name, authentication_status, username = authenticator.login("", "main")

# Save authentication status in session state
st.session_state["authentication_status"] = authentication_status
# Ensure variable is always defined
if authentication_status:
    authenticator.logout("Logout", "sidebar",key="unique_logout_button")
    st.sidebar.success(f"Welcome, {name} 👋")

    st.markdown(
        "<h1 class='custom-main-title' style='text-align: center;'>Data Centre Olympia Education</h1>",
        unsafe_allow_html=True
    )
    # -----------------------------
    # 🎓 Dataset picker (dropdown only)
    # -----------------------------
    st.header("📊 Data Centre Olympia Education")
    file_map = {
        "Tvet": "Tvet.csv",
        "Diploma": "diploma.csv",
        "Bachelor": "bachelor.csv",
        "Master": "https://docs.google.com/spreadsheets/d/1L94h6N7paMdbOlchCw2WpYVKYUJZmaER9p1ym9VUT0M/export?format=csv&gid=0",
        "PhD": "https://docs.google.com/spreadsheets/d/1MiXnjviVRG5XRhbe8Mth5-C29FXV3Kn---qrQD6a-hE/export?format=csv&gid=0"
    }
    # -----------------------------
    # 📥 Load Data Function
    # -----------------------------
    @st.cache_data(ttl=300)  # refresh every 5 minutes
    def load_data(url):
        return pd.read_csv(url,skiprows=2).dropna(how="all").reset_index(drop=True)
    degree = st.selectbox("Select Degree Level", list(file_map.keys()))
    try:
        df = load_data(file_map[degree])
        st.subheader("Data Table")
        st.dataframe(df)
        # -----------------------------
        # 🔎 Global Search Filter
        # -----------------------------
        search_term = st.text_input("Search anything (name, state, phone, etc.):")
    
        filtered_df = df.copy()
        if search_term:
            # Case-insensitive search across all columns
            mask = filtered_df.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_df = filtered_df[mask]
    
        # -----------------------------
        # ⚙️ Advanced Filters
        # -----------------------------
        with st.expander("More Filter Options", expanded=False):
            filter_columns = st.multiselect(
                "Select columns to filter:",
                df.columns
            )
            
            for col in filter_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Numeric filter
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    user_min, user_max = st.slider(
                        f"Filter {col}",
                        min_val, max_val,
                        (min_val, max_val)
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col] >= user_min) & (filtered_df[col] <= user_max)
                    ]
                else:
                    # Text/categorical filter
                    unique_vals = df[col].dropna().unique()
                    selected_vals = st.multiselect(
                        f"Filter {col}",
                        unique_vals
                    )
                    if selected_vals:
                        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    
        # -----------------------------
        # 📋 Show & Download
        # -----------------------------
        st.success(f"✅ Showing {len(filtered_df)} rows after filtering.")
        st.dataframe(filtered_df)
    
        # Download filtered data
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"{degree.lower()}_filtered_data.csv",
            mime="text/csv"
        )
    
        # -----------------------------
        # 📥 Download Filtered Data
        # -----------------------------
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ No data to download. Apply filters to get results.")
        
        # -----------------------------
        # 📊 Summary of Filtered Data
        # -----------------------------
        if len(filtered_df) > 0:
            st.subheader("📊 Summary of Filtered Data")
            st.write(f"Total Rows: **{len(filtered_df)}**")
        
            # ✅ Numeric summary
            numeric_cols = [
                col for col in filtered_df.select_dtypes(include=['number']).columns
                if "phone" not in col.lower()
            ]
        
            if numeric_cols:
                st.markdown("### 🔢 Numeric Columns Summary")
                numeric_summary = filtered_df[numeric_cols].describe().T
                st.dataframe(numeric_summary)
        
                for col in numeric_cols:
                    if filtered_df[col].dropna().shape[0] > 0:
                        fig, ax = plt.subplots()
                        ax.hist(filtered_df[col].dropna(), bins=20, color="skyblue", edgecolor="black")
                        ax.set_title(f"Distribution of {col}")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
        
            # ✅ Categorical summary (excluding identifiers)
            excluded_cats = ["name", "email", "phone", "state"]  # fields to skip
            categorical_cols = [
                col for col in filtered_df.select_dtypes(include=['object']).columns
                if col.lower() not in excluded_cats
            ]
        
            if categorical_cols:
                st.markdown("### 🔤 Categorical Columns Summary")
                for col in categorical_cols:
                    if filtered_df[col].dropna().shape[0] > 0:
                        st.write(f"🔹 {col}")
                        value_counts = filtered_df[col].value_counts().head(10)  # keep top 10 if useful
                        st.dataframe(value_counts)
        
                        # Bar chart only for non-unique categoricals
                        if filtered_df[col].nunique() <= 20:  # avoid messy charts
                            fig, ax = plt.subplots()
                            value_counts.plot(kind="bar", ax=ax, color="lightgreen", edgecolor="black")
                            ax.set_title(f"Distribution of {col}")
                            ax.set_ylabel("Count")
                            st.pyplot(fig)
        else:
            st.subheader("📊 Summary of Filtered Data")
            st.warning("⚠️ No data available after filtering. Please adjust your filters.")
        # -----------------------------
        # 📊 Dashboards Section
        # -----------------------------
        st.subheader("📊 Dashboards")
    
        # Button to show Status Dashboard
        if st.button("Prospek Status Dashboard"):
            try:
                # Select only columns 8–19
                status_df = df.iloc[:, 7:19]
                status_df.columns = status_df.columns.str.strip()
    
                status_counts = status_df.apply(pd.to_numeric, errors="coerce").sum()
                status_summary = pd.DataFrame({
                    "Status": status_counts.index,   
                    "Count": status_counts.values
                })
    
                st.subheader("Status Summary (Bar Chart)")
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(status_summary["Status"], status_summary["Count"], color="skyblue")
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval),
                            ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Count")
                ax.set_xlabel("Status")
                ax.set_title("Prospek Status Summary")
                ax.set_xticklabels(status_summary["Status"], rotation=45, ha="right")
                st.pyplot(fig)
    
                st.subheader("📋 Status Table")
                st.dataframe(status_summary)
    
            except Exception as e:
                st.error(f"Error creating status dashboard: {e}")
    
        # Source Dashboard
        if st.button("Source Dashboard"):
            try:
                if "Source" in df.columns:
                    source_counts = df["Source"].value_counts().reset_index()
                    source_counts.columns = ["Source", "Count"]
    
                    st.subheader("Source Distribution (Pie Chart)")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(source_counts["Count"], labels=source_counts["Source"],
                           autopct="%1.1f%%", startangle=140)
                    ax.set_title("Source Summary")
                    st.pyplot(fig)
    
                    st.subheader("📋 Source Table")
                    st.dataframe(source_counts)
                else:
                    st.warning("⚠️ 'Source' column not found in the dataset.")
            except Exception as e:
                st.error(f"Error creating source dashboard: {e}")
    
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}") 

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
        st.subheader("Data Table")
        st.dataframe(df)
        

        st.subheader("📂 Upload Custom Data")
        uploaded_file = st.file_uploader(f"Upload {degree} Data (CSV/Excel)", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    user_df = pd.read_csv(uploaded_file)
                else:
                    user_df = pd.read_excel(uploaded_file)

                st.success("✅ File uploaded successfully!")
                st.dataframe(user_df.head())

                csv_data = user_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇️ Download {degree} Data (CSV)",
                    data=csv_data,
                    file_name=f"{degree.lower()}_data.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error reading file: {e}")

   # Initialize session state
    if "degree_choice" not in st.session_state:
        st.session_state.degree_choice = None
    
    # Back button resets selection
    if st.button("⬅️ Back to Degree Menu"):
        st.session_state.degree_choice = None
    
    # If no dataset is chosen yet -> show dropdown
    if st.session_state.degree_choice is None:
        degree = st.selectbox("Select Degree Level", list(file_map.keys()))
        st.session_state.degree_choice = degree
    else:
        # Load data based on chosen degree
        try:
            df = load_data(file_map[st.session_state.degree_choice])
            st.subheader(f"📊 Showing data for: {st.session_state.degree_choice}")
            st.dataframe(df)
    
            # Your filters, summary, downloads go here
    
        except Exception as e:
            st.error(f"❌ Could not load dataset: {e}")

elif authentication_status is False:
    st.markdown("""
    <div style='color: black; font-weight: bold; background-color: #ffe6e6; 
                padding: 10px; border-radius: 8px; border-left: 6px solid red;'>
        ❌ Username or password is incorrect
    </div>
    """, unsafe_allow_html=True)



    st.markdown("""
    <div style='color: black; font-weight: bold; background-color: #ffe6e6; 
                padding: 10px; border-radius: 8px; border-left: 6px solid red;'>
        Please enter your username and password!
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Force all text to black */
    body, [class*="css"], .stMarkdown, .stText, .stDataFrame, .stSelectbox, .stRadio, .stCheckbox, .stTextInput label {
        color: black !important;
    }

    /* Also fix Streamlit default elements */
    .st-emotion-cache, .st-emotion-cache p, .st-emotion-cache span, .st-emotion-cache div {
        color: black !important;
    }

    /* Dataframe text */
    .stDataFrame div {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Default headers black */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* Custom teal headers */
    .custom-login-title, .custom-main-title {
        color: teal !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Force dataframe background white */
    .stDataFrame {
        background-color: white !important;
    }
    .stDataFrame div {
        color: black !important;   /* ✅ Make text black */
    }

    /* Force buttons white with teal border */
    div.stButton > button {
        background-color: white !important;
        color: teal !important;
        border: 2px solid teal !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    div.stButton > button:hover {
        background-color: teal !important;
        color: white !important;
    }

    /* File uploader white */
    .stFileUploader {
        background-color: white !important;
        color: black !important;
        border-radius: 8px !important;
        padding: 10px;
    }

    /* Search/filter inputs (inside dataframe) */
    .stTextInput input {
        background-color: white !important;
        color: black !important;   /* ✅ Make search text black */
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Force DataFrame (AgGrid) light theme */
    .stDataFrame [class^="ag-theme"] {
        background-color: white !important;
        color: black !important;
        font-weight: normal !important;
    }

    /* Row background white */
    .stDataFrame [class^="ag-row"] {
        background-color: white !important;
        color: black !important;
    }

    /* Header background light gray, text black */
    .stDataFrame [class^="ag-header"] {
        background-color: #f0f0f0 !important;
        color: black !important;
        font-weight: bold !important;
    }

    /* Search/filter inputs inside grid */
    .stDataFrame input {
        background-color: white !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* 🔹 Dataframe container */
    [data-testid="stDataFrame"] {
        background-color: white !important;
        color: black !important;
    }

    /* 🔹 Table cells */
    [data-testid="stDataFrameCell"] {
        background-color: white !important;
        color: black !important;
    }

    /* 🔹 Table headers */
    [data-testid="stDataFrameHeader"] {
        background-color: #f0f0f0 !important;
        color: black !important;
        font-weight: bold !important;
    }

    /* 🔹 Search/filter box */
    [data-testid="stDataFrame"] input {
        background-color: white !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* 🔹 Dataframe container */
[data-testid="stDataFrame"] {
    background-color: white !important;
    color: black !important;
}

/* 🔹 Table cells */
[data-testid="stDataFrameCell"] {
    background-color: white !important;
    color: black !important;
}

/* 🔹 Table headers */
[data-testid="stDataFrameHeader"] {
    background-color: #f0f0f0 !important;
    color: black !important;
    font-weight: bold !important;
}

/* 🔹 Search/filter box */
[data-testid="stDataFrame"] input {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* 🔹 Force white background for entire data container */
[data-testid="stDataFrame"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ddd !important;
    border-radius: 6px !important;
}

/* 🔹 Fix cell background and text */
[data-testid="stDataFrameCell"] {
    background-color: white !important;
    color: black !important;
}

/* 🔹 Header styling */
[data-testid="stDataFrameHeader"] {
    background-color: #f5f5f5 !important;
    color: black !important;
    font-weight: bold;
}

/* 🔹 Scrollbar visibility */
[data-testid="stDataFrame"]::-webkit-scrollbar {
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)
