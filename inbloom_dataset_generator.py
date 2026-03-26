import csv
import random
from pathlib import Path

from faker import Faker
import pandas as pd


def generate_inbloom_dataset(path="inbloom_participation.csv", n=250):
    fake = Faker()

    colleges = [
        "INBLOOM Academy", "Northgate College", "Eastwood Institute", "Sunrise University", "Apex College",
        "Riverbend College", "Lakeshore University", "Highland Arts College", "Zenith Institute", "Metro Cultural College"
    ]

    states = [
        "Karnataka", "Maharashtra", "Tamil Nadu", "Kerala", "Delhi", "Rajasthan", "Goa", "Telangana", "Uttar Pradesh", "West Bengal"
    ]

    events = [
        "Singing", "Dance", "Drama", "Stand-up Comedy", "Painting", "Poetry", "Fashion Show", "DJ Battle", "Rap", "Magic Show"
    ]

    categories = {
        "Singing": "Music", "Dance": "Performance", "Drama": "Theatre", "Stand-up Comedy": "Comedy", "Painting": "Art",
        "Poetry": "Literature", "Fashion Show": "Style", "DJ Battle": "Music", "Rap": "Music", "Magic Show": "Performance"
    }

    feedback_templates = [
        "Amazing vibe and super supportive crowd for {}!", 
        "Loved the organization and the judges' feedback after {}.",
        "Could improve the stage setup for {} sessions.",
        "The energy during {} was top-notch and unforgettable.",
        "I enjoyed meeting participants from different colleges during {}.",
        "The timing for {} was a little tight, but overall great experience.",
        "I wish there was more audience interaction in {}.",
        "Great sound system for {}. Please keep this up next year!"
    ]

    dataset = []
    for pid in range(1, n + 1):
        event = random.choice(events)
        row = {
            "participant_id": pid,
            "participant_name": fake.name(),
            "college": random.choice(colleges),
            "state": random.choice(states),
            "day": random.randint(1, 5),
            "event": event,
            "event_category": categories[event],
            "group_size": random.choice([1, 2, 3, 4, 5, 6]),
            "event_rating": round(random.uniform(3.0, 5.0), 1),
            "feedback": random.choice(feedback_templates).format(event)
        }
        dataset.append(row)

    df = pd.DataFrame(dataset)
    df.to_csv(path, index=False)
    print(f"Dataset generated: {path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "inbloom_participation.csv"
    generate_inbloom_dataset(out_path, 250)
