import os
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

DATA_FILE = Path(__file__).resolve().parent / "inbloom_participation.csv"


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        from inbloom_dataset_generator import generate_inbloom_dataset

        generate_inbloom_dataset(DATA_FILE, 250)
    return pd.read_csv(DATA_FILE)


def plot_event_trends(df):
    event_counts = df["event"].value_counts().sort_values(ascending=False)
    st.bar_chart(event_counts)

    day_counts = df["day"].value_counts().sort_index()
    st.line_chart(day_counts)

    college_counts = df["college"].value_counts().nlargest(10)
    st.bar_chart(college_counts)

    state_counts = df["state"].value_counts().nlargest(10)
    st.area_chart(state_counts)

    avg_rating = df.groupby("event")["event_rating"].mean().sort_values(ascending=False)
    st.bar_chart(avg_rating)


def text_analysis(df):
    st.subheader("Event-wise Word Cloud")
    selected_event = st.selectbox("Pick Event", sorted(df["event"].unique()))
    text = " ".join(df[df["event"] == selected_event]["feedback"].astype(str))

    if text:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array(), use_column_width=True)
    else:
        st.write("No feedback yet for this event.")

    st.subheader("Compare Feedback Term Frequencies")
    events_for_comp = st.multiselect("Select events to compare", sorted(df["event"].unique()), default=list(df["event"].unique())[:4])

    if len(events_for_comp) >= 2:
        combined_text = [" ".join(df[df["event"] == ev]["feedback"].astype(str)) for ev in events_for_comp]
        cv = CountVectorizer(stop_words="english", max_features=20)
        X = cv.fit_transform(combined_text)
        top_words = pd.DataFrame(X.toarray(), index=events_for_comp, columns=cv.get_feature_names_out()).T
        st.write(top_words.sort_values(by=events_for_comp, ascending=False).head(20))
    else:
        st.info("Select at least 2 events to compare.")


def image_processing_module():
    st.subheader("Day-wise Image Gallery & Processing")
    uploaded = st.file_uploader("Upload event images (PNG/JPEG)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

    if uploaded:
        entries = []
        for file in uploaded:
            day = st.selectbox(f"Assign day for {file.name}", [1, 2, 3, 4, 5], key=file.name)
            entries.append((file, day))

        selected_day = st.slider("Select day to view images", 1, 5, 1)
        st.write(f"Images for day {selected_day}")

        for file, day in entries:
            if day != selected_day:
                continue
            image = Image.open(file).convert("RGB")
            with st.expander(f"{file.name} (Day {day})"):
                st.image(image, caption="Original", width=300)

                gray = ImageOps.grayscale(image)
                edge = gray.filter(ImageFilter.FIND_EDGES)
                st.image([gray, edge], caption=["Grayscale", "Edge Detect"], width=300)

                if st.button(f"Download processed {file.name}", key=f"dl-{file.name}"):
                    out_path = Path("processed_images")
                    out_path.mkdir(exist_ok=True)
                    save_path = out_path / f"processed_{file.name}"
                    edge.save(save_path)
                    st.success(f"Saved {save_path}")
    else:
        st.info("Upload event photos to activate the day-wise gallery and processing.")


def render_dashboard(df):
    st.title("INBLOOM '25 Participation Analytics")

    st.sidebar.header("Filters")
    event_filter = st.sidebar.multiselect("Event", sorted(df["event"].unique()), default=sorted(df["event"].unique()))
    state_filter = st.sidebar.multiselect("State", sorted(df["state"].unique()), default=sorted(df["state"].unique()))
    college_filter = st.sidebar.multiselect("College", sorted(df["college"].unique()), default=sorted(df["college"].unique()))

    filtered = df[
    (df["event"].isin(event_filter)) &
    (df["state"].isin(state_filter)) &
    (df["college"].isin(college_filter))
    ]

    st.metric("Total Participants", filtered.shape[0], delta=filtered.shape[0] - df.shape[0])

    tab1, tab2, tab3 = st.tabs(["Participation", "Feedback Text Analysis", "Image Processing"])

    with tab1:
        st.header("Participation Trends")
        plot_event_trends(filtered)

        st.subheader("Detailed Tables")
        st.dataframe(filtered.head(30))

    with tab2:
        text_analysis(filtered)

    with tab3:
        image_processing_module()


def main():
    df = load_data()
    render_dashboard(df)


if __name__ == "__main__":
    main()
