"""Idempotent database bootstrap for the simplified student demo.

Creates the schema and seeds the Question/Answer tables from the CSV files
committed under app/static/data/seed/. Safe to run on every container start:
on an ephemeral SQLite database it (re)creates everything; if the questions are
already present it does nothing.

User accounts are NOT seeded: they are created anonymously on the fly (see the
/start route in app/auth/routes.py).
"""

import os
from pathlib import Path

import pandas as pd

from app import create_app, db
from app.models import Answer, Question

SEED_DIR = Path(__file__).resolve().parent / "app" / "static" / "data" / "seed"


def populate_table(model, csv_file):
    df = pd.read_csv(csv_file, delimiter=";")
    df = df.astype(object).where(pd.notnull, None)
    for record in df.to_dict("records"):
        db.session.add(model(**record))
    db.session.commit()


def bootstrap():
    app = create_app()
    with app.app_context():
        db.create_all()
        if Question.query.first() is None:
            populate_table(Question, SEED_DIR / "question.csv")
            populate_table(Answer, SEED_DIR / "answer.csv")
            print(
                f"Seeded {Question.query.count()} questions and "
                f"{Answer.query.count()} answers."
            )
        else:
            print("Database already seeded, skipping.")


if __name__ == "__main__":
    bootstrap()
