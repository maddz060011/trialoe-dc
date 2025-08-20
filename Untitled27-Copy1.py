import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
from io import BytesIO

# -----------------------------
# 1. User Credentials (hashed)
# -----------------------------
# Hash plain-text passwords just once
hashed_passwords = stauth.Hasher(['123', '456', '789']).generate()

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
# 3. Login
# -----------------------------
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    # Sidebar
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome, {name} 👋")

    # -----------------------------
    # 4. Dashboard
    # -----------------------------
    st.title("📊 Company Dashboard")
    st.write("This dashboard lets you view company data, upload files, filter them, and download the results.")

    # Example demo data
    st.header("📋 Sample Data (Company Records)")
    data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Sales": [100, 120, 90, 150, 200],
        "Satisfaction": [3.2, 3.5, 3.1, 3.8, 4.0]
    }
    df = pd.DataFrame(data)

    st.subheader("Company Data Table")
    st.dataframe(df)

    # Charts
    st.subheader("📈 Sales Trend")
    st.line_chart(df.set_index("Month")["Sales"])

    st.subheader("⭐ Satisfaction Score")
    st.bar_chart(df.set_index("Month")["Satisfaction"])

    # -----------------------------
    # File Upload Section
    # -----------------------------
    st.header("📂 Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")
            st.subheader("🔍 Preview of Your Data")
            st.dataframe(user_df.head())

            # Filtering
            st.subheader("🔎 Filter Your Data")
            filterable_cols = user_df.select_dtypes(include=["object", "category"]).columns
            filters = {}
            if len(filterable_cols) > 0:
                for col in filterable_cols:
                    options = user_df[col].unique()
                    selected = st.multiselect(f"Filter {col}", options)
                    if selected:
                        filters[col] = selected
                for col, selected in filters.items():
                    user_df = user_df[user_df[col].isin(selected)]

            # Show filtered data
            st.subheader("📋 Filtered Data")
            st.dataframe(user_df)

            # Auto-generate charts
            numeric_cols = user_df.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                st.subheader("📊 Auto-generated Charts")
                for col in numeric_cols:
                    st.line_chart(user_df[col], use_container_width=True)
            else:
                st.warning("⚠️ No numeric columns found for charting.")

            # Download Buttons
            st.subheader("💾 Download Filtered Data")
            csv_data = user_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv_data,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                user_df.to_excel(writer, index=False, sheet_name="FilteredData")
            st.download_button(
                label="⬇️ Download as Excel",
                data=buffer,
                file_name="filtered_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif authentication_status is False:
    st.error("❌ Username or password is incorrect")

elif authentication_status is None:
    st.warning("⚠️ Please enter your username and password")
