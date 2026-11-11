"""SQLite schema, embedded as a string so it always ships with the package."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    route TEXT NOT NULL,
    policy_version INTEGER NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    rating INTEGER,
    issue_type TEXT NOT NULL,
    required_terms_json TEXT,
    preferred_format TEXT,
    ideal_response TEXT,
    correction TEXT,
    notes TEXT,
    source TEXT NOT NULL DEFAULT 'human',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patch_type TEXT NOT NULL,
    target TEXT NOT NULL,
    content_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    source_issue_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    score_before REAL,
    score_after REAL,
    route TEXT,
    details_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    promoted_at TEXT
);

CREATE TABLE IF NOT EXISTS policy_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_text TEXT NOT NULL,
    source_patch_id INTEGER REFERENCES patches(id) ON DELETE SET NULL,
    active INTEGER NOT NULL DEFAULT 0,
    route TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TEXT
);

CREATE TABLE IF NOT EXISTS few_shot_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    route TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    source_patch_id INTEGER REFERENCES patches(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    route TEXT NOT NULL,
    source TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    source_patch_id INTEGER REFERENCES patches(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mistake_triage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_type TEXT NOT NULL,
    verdict TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    reasons_json TEXT,
    case_count INTEGER NOT NULL,
    route TEXT,
    max_feedback_id INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_issue_type ON feedback(issue_type);
CREATE INDEX IF NOT EXISTS idx_feedback_interaction_id ON feedback(interaction_id);
CREATE INDEX IF NOT EXISTS idx_patches_status ON patches(status);
CREATE INDEX IF NOT EXISTS idx_policy_rules_active ON policy_rules(active);
"""

# Columns added after 0.4.0. Databases created by older versions are upgraded
# in place the first time they are opened.
MIGRATIONS = [
    ("feedback", "ideal_response", "TEXT"),
    ("feedback", "source", "TEXT NOT NULL DEFAULT 'human'"),
    ("patches", "route", "TEXT"),
    ("patches", "details_json", "TEXT"),
    ("policy_rules", "route", "TEXT"),
    ("few_shot_examples", "source_patch_id", "INTEGER"),
    ("mistake_triage", "route", "TEXT"),
    ("mistake_triage", "max_feedback_id", "INTEGER"),
]
