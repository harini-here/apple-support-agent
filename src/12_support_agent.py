import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(
    "data/processed/apple_training.csv"
)

train_df["clean_text"] = train_df["clean_text"].fillna("")

X_train = train_df["clean_text"]
y_train = train_df["intent"]


# ============================================================
# 2. TRAIN TF-IDF + LOGISTIC REGRESSION
# ============================================================

print("Training support intent classifier...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

classifier = LogisticRegression(
    class_weight="balanced",
    max_iter=2000
)

classifier.fit(
    X_train_tfidf,
    y_train
)

print("Classifier ready.")


# ============================================================
# 3. ESCALATION RULES
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
        "billing dispute"
    ],

    "account_security": [
        "hacked",
        "account stolen",
        "account compromised",
        "someone accessed",
        "security breach",
        "unauthorized access"
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


def detect_risk(text):

    text_lower = text.lower()

    for reason, patterns in ESCALATION_PATTERNS.items():

        for pattern in patterns:

            if pattern in text_lower:
                return True, reason

    return False, None


# ============================================================
# 4. MULTI-ISSUE DETECTION
# ============================================================

ISSUE_GROUPS = {

    "battery": [
        "battery",
        "charging",
        "charge",
        "charger"
    ],

    "performance": [
        "slow",
        "lag",
        "freeze",
        "freezing",
        "crash",
        "hang"
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "internet",
        "signal"
    ],

    "account": [
        "apple id",
        "icloud",
        "login",
        "sign in",
        "password"
    ],

    "apps": [
        "app",
        "application",
        "app store",
        "itunes"
    ],

    "device": [
        "iphone",
        "ipad",
        "macbook",
        "mac",
        "screen",
        "keyboard"
    ]
}


def detect_multiple_issue_groups(text):

    text_lower = text.lower()

    matched_groups = []

    for group, keywords in ISSUE_GROUPS.items():

        for keyword in keywords:

            if keyword in text_lower:

                matched_groups.append(group)
                break

    return list(set(matched_groups))


# ============================================================
# 5. RESPONSE TEMPLATES
# ============================================================

RESPONSE_TEMPLATES = {

    "ios_update":
        "I can help troubleshoot your iOS update issue. Please share your device model and current iOS version so we can narrow down the problem.",

    "battery_charging":
        "I can help troubleshoot the battery or charging issue. Please share your device model, whether it charges normally, and when you first noticed the problem.",

    "device_hardware":
        "I can help narrow down the device issue. Please share your device model and describe what is happening and when it started.",

    "app_issues":
        "I can help troubleshoot the app issue. Please tell me which app is affected and what happens when you try to use it.",

    "performance":
        "I can help troubleshoot the performance issue. Please share your device model, iOS version, and what kind of slowdown, freezing, or crashing you're experiencing.",

    "screen_display_keyboard":
        "I can help with the display or keyboard issue. Please share your device model and describe exactly what happens.",

    "connectivity":
        "I can help troubleshoot the connectivity issue. Please let me know whether the problem is with Wi-Fi, Bluetooth, mobile data, or another connection.",

    "apple_id_icloud":
        "I can help with your Apple ID or iCloud issue. Please describe the problem you're seeing, such as sign-in, syncing, or account access.",

    "app_store_itunes":
        "I can help with the App Store or iTunes issue. Please tell me what you're trying to download or access and what error or behavior you're seeing.",

    "apple_music_media":
        "I can help troubleshoot the Apple Music or media issue. Please tell me what you're trying to play and what happens when you try.",

    "billing_payment_purchase":
        "This request may involve billing or payment details, so I'll route it to a support specialist for further assistance.",

    "orders_products_warranty":
        "I can help with the product, order, or warranty issue. Please share the relevant product or order details so the support team can assist you.",

    "other":
        "I want to make sure this is handled correctly, so I'll route your request to a support specialist for further assistance."
}


# ============================================================
# 6. SUPPORT AGENT
# ============================================================

def support_agent(message, confidence_threshold=0.55):

    message = str(message).strip()

    if not message:

        return {
            "intent": "other",
            "confidence": 0.0,
            "decision": "ESCALATE",
            "reason": "empty_message",
            "reply": RESPONSE_TEMPLATES["other"]
        }

    # --------------------------------------------------------
    # Intent prediction
    # --------------------------------------------------------

    X = vectorizer.transform([message])

    probabilities = classifier.predict_proba(X)[0]

    best_index = np.argmax(probabilities)

    predicted_intent = classifier.classes_[best_index]

    confidence = float(probabilities[best_index])

    # --------------------------------------------------------
    # Risk detection
    # --------------------------------------------------------

    risky, risk_reason = detect_risk(message)

    # --------------------------------------------------------
    # Multi-issue detection
    # --------------------------------------------------------

    issue_groups = detect_multiple_issue_groups(message)

    is_multi_issue = len(issue_groups) >= 3

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if risky:

        decision = "ESCALATE"
        reason = risk_reason

    elif is_multi_issue:

        decision = "ESCALATE"
        reason = "multiple_issue_groups"

    elif confidence < confidence_threshold:

        decision = "ESCALATE"
        reason = "low_confidence"

    else:

        decision = "AUTO_HANDLE"
        reason = "high_confidence"

    # --------------------------------------------------------
    # Reply
    # --------------------------------------------------------

    reply = RESPONSE_TEMPLATES.get(
        predicted_intent,
        RESPONSE_TEMPLATES["other"]
    )

    return {
        "intent": predicted_intent,
        "confidence": round(confidence, 4),
        "decision": decision,
        "reason": reason,
        "issue_groups": issue_groups,
        "reply": reply
    }


# ============================================================
# 7. DEMO
# ============================================================

if __name__ == "__main__":

    test_messages = [

        "My iPhone battery is draining really fast",

        "I can't sign into my iCloud account",

        "The App Store won't download my apps",

        "My screen keeps freezing",

        "I was charged twice for the same purchase",

        "My iPhone is very slow after the latest update",

        "WiFi keeps disconnecting from my iPhone",

        "My iPhone is slow, WiFi doesn't work and I can't sign into iCloud"
    ]

    print("\n================================")
    print("APPLE SUPPORT AI AGENT")
    print("================================")

    for message in test_messages:

        result = support_agent(message)

        print("\nCustomer:")
        print(message)

        print("\nIntent:")
        print(result["intent"])

        print("Confidence:")
        print(result["confidence"])

        print("Issue groups:")
        print(result["issue_groups"])

        print("Decision:")
        print(result["decision"])

        print("Reason:")
        print(result["reason"])

        print("Reply:")
        print(result["reply"])

        print("-" * 70)