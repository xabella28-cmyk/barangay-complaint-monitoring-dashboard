import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="BarangayPulse | Complaint Intelligence", page_icon="🏘️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(135deg,#f7f8ff 0%,#f4f7fb 55%,#f8f5ff 100%);}
.block-container {padding-top:1.6rem; padding-bottom:2.5rem; max-width:1500px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#111b3a,#202b58); }
[data-testid="stSidebar"] * {color:#f4f6ff !important;}
.hero {background:linear-gradient(115deg,#172554 0%,#3446a5 55%,#6d4bd1 100%);padding:27px 30px;border-radius:22px;color:white;margin-bottom:22px;box-shadow:0 12px 35px rgba(45,55,130,.18);}
.hero h1 {font-family:'Manrope';font-size:30px;font-weight:800;margin:0 0 5px 0;color:white;}
.hero p {color:#dce4ff;margin:0;font-size:14px;}
.kpi {background:white;border:1px solid #e9ecf5;border-radius:18px;padding:17px 18px;box-shadow:0 5px 18px rgba(31,45,90,.05);min-height:112px;}
.kpi-label {font-size:12px;color:#77809b;font-weight:700;text-transform:uppercase;letter-spacing:.6px;}
.kpi-value {font-family:'Manrope';font-size:29px;font-weight:800;color:#1d2854;margin-top:5px;}
.kpi-sub {font-size:11px;color:#929ab0;margin-top:2px;}
.section-title {font-family:'Manrope';font-size:19px;font-weight:800;color:#202b55;margin:15px 0 10px;}
div[data-testid="stMetric"] {background:white;padding:15px 17px;border:1px solid #e9ecf5;border-radius:16px;box-shadow:0 5px 18px rgba(31,45,90,.04);}
.stTabs [data-baseweb="tab"] {font-weight:700;}
div.stButton > button {border-radius:10px;font-weight:700;}
.small-note {color:#77809b;font-size:12px;}
</style>
""", unsafe_allow_html=True)

DEFAULT = "data/03_barangay_complaints_synthetic.csv"
st.sidebar.markdown("## 🏘️ BarangayPulse")
st.sidebar.caption("COMMUNITY COMPLAINT INTELLIGENCE")
uploaded = st.sidebar.file_uploader("Upload complaint CSV", type=["csv"])
try:
    df = pd.read_csv(uploaded if uploaded else DEFAULT)
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

required = ["Complaint_ID","Date_Reported","Complaint_Category","Zone","Submission_Channel","Priority","Status","Days_Open","Assigned_Office","Resident_Age_Group"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df["Date_Reported"] = pd.to_datetime(df["Date_Reported"], errors="coerce")
df["Days_Open"] = pd.to_numeric(df["Days_Open"], errors="coerce").fillna(0)
df = df.dropna(subset=["Date_Reported","Priority","Complaint_Category"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔎 Dashboard Filters")
zones = sorted(df["Zone"].dropna().unique())
selected_zones = st.sidebar.multiselect("Barangay zone", zones, default=zones)
priorities = ["Low","Medium","High"]
selected_priorities = st.sidebar.multiselect("Priority level", priorities, default=priorities)
statuses = sorted(df["Status"].dropna().unique())
selected_status = st.sidebar.multiselect("Complaint status", statuses, default=statuses)
date_min, date_max = df["Date_Reported"].min().date(), df["Date_Reported"].max().date()
date_range = st.sidebar.date_input("Reported date range", (date_min,date_max), min_value=date_min, max_value=date_max)
if isinstance(date_range, tuple) and len(date_range)==2:
    dstart, dend = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    dstart, dend = pd.Timestamp(date_min), pd.Timestamp(date_max)
f = df[(df["Zone"].isin(selected_zones)) & (df["Priority"].isin(selected_priorities)) & (df["Status"].isin(selected_status)) & (df["Date_Reported"].between(dstart,dend))]

st.markdown("""
<div class="hero">
  <h1>🏘️ BarangayPulse</h1>
  <p>Complaint Pattern Analysis & Priority Monitoring · Community service intelligence dashboard</p>
</div>
""", unsafe_allow_html=True)

total=len(f)
high=int((f["Priority"]=="High").sum())
open_count=int(f["Status"].isin(["Pending","In Progress","Escalated"]).sum())
resolved=int((f["Status"]=="Resolved").sum())
resolution_rate=(resolved/total*100) if total else 0
c1,c2,c3,c4=st.columns(4)
for col,label,value,sub in [
    (c1,"TOTAL COMPLAINTS",f"{total:,}","Records in selected filters"),
    (c2,"HIGH PRIORITY",f"{high:,}",f"{(high/total*100 if total else 0):.1f}% of complaints"),
    (c3,"ACTIVE CASES",f"{open_count:,}","Pending, in progress or escalated"),
    (c4,"RESOLVED RATE",f"{resolution_rate:.1f}%","Based on selected records")]:
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs(["📊 Overview","🧭 Complaint Patterns","🤖 Priority Prediction","📋 Records & Reports"])
with tab1:
    left,right=st.columns([1.55,1])
    with left:
        st.markdown('<div class="section-title">Complaint activity over time</div>',unsafe_allow_html=True)
        trend=f.assign(Month=f["Date_Reported"].dt.to_period("M").astype(str)).groupby("Month").size().reset_index(name="Complaints")
        fig=px.area(trend,x="Month",y="Complaints",markers=True,color_discrete_sequence=["#6558d3"])
        fig.update_traces(line_width=3,fillcolor="rgba(101,88,211,.13)")
        fig.update_layout(height=330,margin=dict(l=5,r=5,t=15,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white",font_color="#59627d")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Priority distribution</div>',unsafe_allow_html=True)
        p=f["Priority"].value_counts().reindex(["High","Medium","Low"]).dropna().reset_index()
        p.columns=["Priority","Count"]
        fig=px.pie(p,names="Priority",values="Count",hole=.65,color="Priority",color_discrete_map={"High":"#ef647c","Medium":"#f4b74f","Low":"#4db9a8"})
        fig.update_layout(height=330,margin=dict(l=5,r=5,t=15,b=5),paper_bgcolor="rgba(0,0,0,0)",legend_title="")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section-title">Status snapshot</div>',unsafe_allow_html=True)
    s=f["Status"].value_counts().reset_index(); s.columns=["Status","Cases"]
    fig=px.bar(s,x="Status",y="Cases",color="Status",color_discrete_sequence=["#6675dc","#38b6a0","#f0b44e","#e87991"])
    fig.update_layout(height=280,showlegend=False,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
    st.plotly_chart(fig,use_container_width=True)

with tab2:
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="section-title">Complaint categories</div>',unsafe_allow_html=True)
        cat=f["Complaint_Category"].value_counts().reset_index(); cat.columns=["Category","Count"]
        fig=px.bar(cat.sort_values("Count"),x="Count",y="Category",orientation="h",color="Count",color_continuous_scale=["#b8b7f4","#5548c8"])
        fig.update_layout(height=390,margin=dict(l=5,r=5,t=10,b=5),coloraxis_showscale=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        st.markdown('<div class="section-title">Complaints by zone and priority</div>',unsafe_allow_html=True)
        cross=f.groupby(["Zone","Priority"]).size().reset_index(name="Complaints")
        fig=px.bar(cross,x="Zone",y="Complaints",color="Priority",barmode="group",color_discrete_map={"High":"#ef647c","Medium":"#f4b74f","Low":"#4db9a8"})
        fig.update_layout(height=390,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="section-title">Average open days by category</div>',unsafe_allow_html=True)
    avg=f.groupby("Complaint_Category",as_index=False)["Days_Open"].mean().sort_values("Days_Open",ascending=False)
    fig=px.bar(avg,x="Complaint_Category",y="Days_Open",color="Days_Open",color_continuous_scale="Purples")
    fig.update_layout(height=300,margin=dict(l=5,r=5,t=10,b=70),coloraxis_showscale=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
    st.plotly_chart(fig,use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Machine learning: complaint priority classification</div>',unsafe_allow_html=True)
    st.caption("A Random Forest model estimates priority using complaint details. It is a decision-support aid, not a replacement for barangay assessment.")
    features=["Complaint_Category","Zone","Submission_Channel","Assigned_Office","Resident_Age_Group"]
    model_df=df.dropna(subset=features+["Priority"]).copy()
    X=model_df[features]; y=model_df["Priority"]
    if y.nunique()<2 or y.value_counts().min()<2:
        st.warning("Not enough examples in each priority class to train a stratified model.")
    else:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        categorical=features
        prep=ColumnTransformer([("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
        model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=180,random_state=42,class_weight="balanced"))])
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
        m1,m2,m3,m4=st.columns(4)
        for col,label,val in [(m1,"ACCURACY",accuracy_score(y_test,pred)),(m2,"PRECISION",precision_score(y_test,pred,average="weighted",zero_division=0)),(m3,"RECALL",recall_score(y_test,pred,average="weighted",zero_division=0)),(m4,"F1 SCORE",f1_score(y_test,pred,average="weighted",zero_division=0))]:
            col.metric(label,f"{val:.1%}")
        cm=confusion_matrix(y_test,pred,labels=["Low","Medium","High"])
        fig=px.imshow(cm,x=["Low","Medium","High"],y=["Low","Medium","High"],text_auto=True,color_continuous_scale="Purples",labels=dict(x="Predicted",y="Actual",color="Count"))
        fig.update_layout(height=330,title="Confusion matrix",margin=dict(l=5,r=5,t=45,b=5))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("#### Try a sample priority estimate")
        q1,q2,q3=st.columns(3)
        with q1: qcat=st.selectbox("Complaint category",sorted(df["Complaint_Category"].unique()))
        with q2: qzone=st.selectbox("Zone",sorted(df["Zone"].unique()))
        with q3: qchannel=st.selectbox("Submission channel",sorted(df["Submission_Channel"].unique()))
        q4,q5=st.columns(2)
        with q4: qoffice=st.selectbox("Assigned office",sorted(df["Assigned_Office"].unique()))
        with q5: qage=st.selectbox("Resident age group",sorted(df["Resident_Age_Group"].unique()))
        if st.button("Estimate complaint priority",type="primary"):
            sample=pd.DataFrame([{"Complaint_Category":qcat,"Zone":qzone,"Submission_Channel":qchannel,"Assigned_Office":qoffice,"Resident_Age_Group":qage}])
            result=model.predict(sample)[0]
            st.success(f"Estimated priority: {result}")
            st.caption("Please validate this estimate using actual complaint details and barangay guidelines.")

with tab4:
    st.markdown('<div class="section-title">Complaint register</div>',unsafe_allow_html=True)
    search=st.text_input("Search complaint ID, category, or zone")
    view=f.copy()
    if search:
        mask=view.astype(str).apply(lambda col: col.str.contains(search,case=False,na=False)).any(axis=1)
        view=view[mask]
    st.dataframe(view.sort_values("Date_Reported",ascending=False),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download filtered complaints (CSV)",view.to_csv(index=False).encode("utf-8"),"barangay_complaints_filtered.csv","text/csv",use_container_width=False)
    st.caption(f"Showing {len(view):,} records · Synthetic demonstration dataset")

st.markdown("---")
st.caption("BarangayPulse · Capstone 1 prototype · Demonstration data only")
