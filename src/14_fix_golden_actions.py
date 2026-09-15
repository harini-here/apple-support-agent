import pandas as pd


# ============================================================
# LOAD GOLDEN SET
# ============================================================

INPUT_FILE = "evaluation/golden_set.csv"
OUTPUT_FILE = "evaluation/golden_set.csv"

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} Golden Set examples.")


# ============================================================
# ESCALATION RULES
# ============================================================

ESCALATION_PATTERNS = {

    "refund_or_charge": [
        "refund",
        "charged twice",
        "charged me",
        "wrong charge",
        "unauthorized charge",
        "money back",
        "payment dispute",
        "billing dispute",
        "billing issue",
        "payment issue"
    ],

    "account_security": [
        "hacked",
        "account stolen",
        "someone accessed",
        "security breach",
        "unauthorized access",
        "account compromised"
    ],

    "legal_or_serious": [
        "lawsuit",
        "legal action",
        "court",
        "police",
        "fraud",
        "scam"
    ],

    "hardware_service": [
        "genius bar",
        "repair",
        "replacement",
        "physical damage",
        "broken screen",
        "broken phone",
        "broken iphone"
    ]
}


def find_escalation_reason(text):

    text = str(text).lower()

    for reason, patterns in ESCALATION_PATTERNS.items():

        for pattern in patterns:

            if pattern in text:
                return reason

    return None


# ============================================================
# ASSIGN EXPECTED ACTION
# ============================================================

actions = []
reasons = []

for text in df["clean_text"]:

    reason = find_escalation_reason(text)

    if reason is not None:

        actions.append("ESCALATE")
        reasons.append(reason)

    else:

        actions.append("AUTO_HANDLE")
        reasons.append("none")


df["expected_action"] = actions
df["escalation_reason"] = reasons


# ============================================================
# UPDATE REVIEW STATUS
# ============================================================

df["review_status"] = "VERIFIED_RULE_ASSISTED"


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("GOLDEN SET ACTION LABELS")
print("================================")

print("\nAction distribution:")

print(
    df["expected_action"].value_counts()
)

print("\nEscalation reasons:")

print(
    df["escalation_reason"].value_counts()
)

print("\nSaved:")
print(OUTPUT_FILE)