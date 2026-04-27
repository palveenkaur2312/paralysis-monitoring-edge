"""
demo_seed.py — Seeds the database with fake alert events so you can
see the dashboard working immediately without running inference.

Run:  python demo_seed.py
Then: python main.py --mode dashboard
Then: open http://localhost:5000
"""

import sqlite3
import random
from datetime import datetime, timedelta

import config
import utils

utils.init_db()

event_types = ["inactivity", "distress_expression", "sustained_lip_activity"]
camera_ids  = ["CAM_01", "CAM_02"]

print("Seeding demo events into database...")

with sqlite3.connect(config.DB_PATH) as conn:
    # Clear old demo data
    conn.execute("DELETE FROM events")
    conn.commit()

    now = datetime.utcnow()
    for i in range(30):
        ts         = (now - timedelta(minutes=random.randint(0, 120))).isoformat(timespec="seconds") + "Z"
        event_type = random.choice(event_types)
        camera_id  = random.choice(camera_ids)
        track_id   = random.randint(1, 5)
        confidence = round(random.uniform(0.60, 0.99), 2)
        conn.execute(
            "INSERT INTO events (camera_id, event_type, track_id, confidence, timestamp) VALUES (?,?,?,?,?)",
            (camera_id, event_type, track_id, confidence, ts)
        )
    conn.commit()

print("Done! 30 demo events added.")
print("Now run:  python main.py --mode dashboard")
print("Then open: http://localhost:5000")
