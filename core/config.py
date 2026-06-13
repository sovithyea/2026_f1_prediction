"""
config.py
---------
All static constants for the 2026 F1 prediction project.
Import from here instead of hard-coding in any prediction script.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# 2026 Driver grid  (22 drivers, 11 teams)
# ---------------------------------------------------------------------------
DRIVERS_2026 = [
    # (DriverCode, FullName,            Team,           TeamCode)
    ("NOR", "Lando Norris",       "McLaren",       "MCL"),
    ("PIA", "Oscar Piastri",      "McLaren",       "MCL"),
    ("HAM", "Lewis Hamilton",     "Ferrari",       "FER"),
    ("LEC", "Charles Leclerc",    "Ferrari",       "FER"),
    ("VER", "Max Verstappen",     "Red Bull",      "RBR"),
    ("HAD", "Isack Hadjar",       "Red Bull",      "RBR"),
    ("RUS", "George Russell",     "Mercedes",      "MER"),
    ("ANT", "Kimi Antonelli",     "Mercedes",      "MER"),
    ("ALB", "Alex Albon",         "Williams",      "WIL"),
    ("SAI", "Carlos Sainz",       "Williams",      "WIL"),
    ("HUL", "Nico Hulkenberg",    "Audi",          "AUD"),
    ("BOR", "Gabriel Bortoleto",  "Audi",          "AUD"),
    ("ALO", "Fernando Alonso",    "Aston Martin",  "AMR"),
    ("STR", "Lance Stroll",       "Aston Martin",  "AMR"),
    ("GAS", "Pierre Gasly",       "Alpine",        "ALP"),
    ("COL", "Franco Colapinto",   "Alpine",        "ALP"),
    ("OCO", "Esteban Ocon",       "Haas",          "HAS"),
    ("BEA", "Oliver Bearman",     "Haas",          "HAS"),
    ("LAW", "Liam Lawson",        "Racing Bulls",  "RB"),
    ("LIN", "Arvid Lindblad",     "Racing Bulls",  "RB"),
    ("PER", "Sergio Perez",       "Cadillac",      "CAD"),
    ("BOT", "Valtteri Bottas",    "Cadillac",      "CAD"),
]

DRIVERS_DF = pd.DataFrame(
    DRIVERS_2026, columns=["DriverCode", "FullName", "Team", "TeamCode"]
)

# ---------------------------------------------------------------------------
# 2026 Race Calendar — 24 rounds
# Sprint weekends flagged separately
# ---------------------------------------------------------------------------
CALENDAR_2026 = [
    # (Round, Name,             Circuit,                        Country,        Date,         Sprint)
    (1,  "Australian GP",       "Albert Park",                  "Australia",    "2026-03-08", False),
    (2,  "Chinese GP",          "Shanghai International",       "China",        "2026-03-15", True),
    (3,  "Japanese GP",         "Suzuka",                       "Japan",        "2026-03-29", False),
    (4,  "Bahrain GP",          "Bahrain International",        "Bahrain",      "2026-04-12", False),
    (5,  "Saudi Arabian GP",    "Jeddah Corniche",              "Saudi Arabia", "2026-04-19", False),
    (6,  "Miami GP",            "Miami International",          "USA",          "2026-05-03", True),
    (7,  "Canadian GP",         "Circuit Gilles Villeneuve",    "Canada",       "2026-05-24", True),
    (8,  "Monaco GP",           "Circuit de Monaco",            "Monaco",       "2026-06-07", False),
    (9,  "Spanish GP",          "Circuit de Barcelona-Catalunya","Spain",       "2026-06-14", False),
    (10, "Austrian GP",         "Red Bull Ring",                "Austria",      "2026-06-28", False),
    (11, "British GP",          "Silverstone",                  "UK",           "2026-07-05", True),
    (12, "Belgian GP",          "Spa-Francorchamps",            "Belgium",      "2026-07-19", False),
    (13, "Hungarian GP",        "Hungaroring",                  "Hungary",      "2026-07-26", False),
    (14, "Dutch GP",            "Circuit Zandvoort",            "Netherlands",  "2026-08-23", True),
    (15, "Italian GP",          "Monza",                        "Italy",        "2026-09-06", False),
    (16, "Madrid GP",           "Circuito de Madrid",           "Spain",        "2026-09-13", False),
    (17, "Azerbaijan GP",       "Baku City Circuit",            "Azerbaijan",   "2026-09-20", False),
    (18, "Singapore GP",        "Marina Bay",                   "Singapore",    "2026-09-27", False),
    (19, "Austin GP",           "Circuit of the Americas",      "USA",          "2026-10-18", True),
    (20, "Mexico City GP",      "Autodromo Hermanos Rodriguez", "Mexico",       "2026-10-25", False),
    (21, "São Paulo GP",        "Interlagos",                   "Brazil",       "2026-11-08", True),
    (22, "Las Vegas GP",        "Las Vegas Strip Circuit",      "USA",          "2026-11-21", False),
    (23, "Qatar GP",            "Lusail International",         "Qatar",        "2026-11-29", True),
    (24, "Abu Dhabi GP",        "Yas Marina",                   "UAE",          "2026-12-06", False),
]

CALENDAR_DF = pd.DataFrame(
    CALENDAR_2026,
    columns=["Round", "Name", "Circuit", "Country", "Date", "Sprint"]
)

# ---------------------------------------------------------------------------
# Team constructor performance scores — 2026 new-era baseline
# Scale: 0–10  (from 2025 standings + new-regs reset estimate)
# ---------------------------------------------------------------------------
TEAM_PERFORMANCE_2026 = {
    "McLaren":      9.5,
    "Ferrari":      9.0,
    "Red Bull":     8.8,
    "Mercedes":     8.7,
    "Aston Martin": 7.8,
    "Williams":     7.5,
    "Racing Bulls": 7.2,
    "Haas":         6.8,
    "Alpine":       6.5,
    "Audi":         6.0,   # new manufacturer, regulation reset
    "Cadillac":     5.5,   # brand-new 11th constructor
}

# Per-driver modifier added on top of team score
# Accounts for experience, feedback quality, regulation development input
DRIVER_MODIFIER = {
    "NOR": +0.25, "PIA": +0.20,
    "HAM": +0.30, "LEC": +0.25,
    "VER": +0.40, "HAD": -0.10,   # rookie
    "RUS": +0.15, "ANT": -0.05,   # second season
    "ALB": +0.10, "SAI": +0.20,
    "HUL": +0.10, "BOR": -0.10,   # rookie
    "ALO": +0.20, "STR": +0.00,
    "GAS": +0.10, "COL": -0.05,
    "OCO": +0.05, "BEA": -0.05,
    "LAW": +0.05, "LIN": -0.20,   # rookie
    "PER": +0.05, "BOT": +0.05,
}

# FastF1 3-letter codes used in 2025 → 2026 equivalents
# (used to remap training data when a seat changed hands)
CODE_MAP_2025_TO_2026 = {
    "NOR": "NOR", "PIA": "PIA",
    "HAM": "HAM", "LEC": "LEC",
    "VER": "VER", "TSU": "HAD",   # Tsunoda out → Hadjar in
    "RUS": "RUS", "ANT": "ANT",
    "ALB": "ALB", "SAI": "SAI",
    "HUL": "HUL", "BOR": "BOR",
    "ALO": "ALO", "STR": "STR",
    "GAS": "GAS", "COL": "COL",
    "OCO": "OCO", "BEA": "BEA",
    "LAW": "LAW", "HAD": "LIN",   # Hadjar's 2025 RB slot → Lindblad
    "PER": "PER", "BOT": "BOT",
    # Drivers not continuing — map to None (rows are dropped)
    "MAG": None, "GIO": None, "ZHO": None,
}

# ---------------------------------------------------------------------------
# Team colours for plots
# ---------------------------------------------------------------------------
TEAM_COLORS = {
    "McLaren":      "#FF8000",
    "Ferrari":      "#E8002D",
    "Red Bull":     "#3671C6",
    "Mercedes":     "#27F4D2",
    "Williams":     "#64C4FF",
    "Aston Martin": "#358C75",
    "Alpine":       "#FF87BC",
    "Haas":         "#B6BABD",
    "Racing Bulls": "#6692FF",
    "Audi":         "#D0D0D0",
    "Cadillac":     "#B87333",
}

# ---------------------------------------------------------------------------
# GBR hyperparameters (same as mar-antaya reference repo)
# ---------------------------------------------------------------------------
GBR_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       5,
    "random_state":    39,
    "subsample":       0.8,
    "min_samples_leaf": 2,
}

# Feature columns the model trains and predicts on
FEATURE_COLS = [
    "QualifyingTime",           # gap from pole (normalized)
    "FP1Time",                  # FP1 best lap gap (normalized)
    "FP2Time",                  # FP2 best lap / long-run gap (normalized)
    "FP3Time",                  # FP3 best lap gap (normalized)
    "TeamPerformanceScore",
    "RainProbability",
    "Temperature",
    "AverageSeasonPerformance",
]