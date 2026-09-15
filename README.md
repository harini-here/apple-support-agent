# Apple Support AI Agent



An AI customer support agent built from the Customer Support on Twitter dataset, focusing on AppleSupport customer conversations.



## Project Overview



This project builds an intent classification and support-routing system for Apple customer support messages.



The agent:



1\. Classifies incoming customer messages into support intents.

2\. Estimates classification confidence.

3\. Detects high-risk issues such as refunds, account security, legal/serious complaints, and hardware service requests.

4\. Detects messages containing multiple issue types.

5\. Automatically handles low-risk requests using safe response templates.

6\. Escalates risky, ambiguous, or multi-issue requests to human support.



## Dataset



Source: Thought Vector - Customer Support on Twitter



Dataset:

`customer-support-on-twitter`



The raw dataset is intentionally excluded from Git because of its large size.



The project extracts AppleSupport conversations and creates processed datasets for training and evaluation.



## Intent Taxonomy



The system uses the following support intents:



\- ios\_update

\- battery\_charging

\- device\_hardware

\- app\_issues

\- performance

\- screen\_display\_keyboard

\- connectivity

\- apple\_id\_icloud

\- app\_store\_itunes

\- apple\_music\_media

\- billing\_payment\_purchase

\- orders\_products\_warranty

\- other



## Approach



### Baseline 1: Rule-Based Classifier



A keyword/rule-based classifier was implemented as the first baseline.



Performance on the 200-example Golden Set:



\- Accuracy: 64.0%

\- Macro F1: 58.2%



\### Baseline 2: TF-IDF + Logistic Regression



The main classifier uses:



\- TF-IDF features

\- Unigrams and bigrams

\- Logistic Regression

\- Balanced class weights



Training labels were created using weak supervision from the AppleSupport customer messages.



The 200 Golden Set examples were excluded from training.



Performance:



\- Accuracy: 68.5%

\- Macro F1: 67.9%

\- Weighted F1: 67.6%



\### Embedding Experiments



Two embedding-based approaches were also tested:



\- SentenceTransformer prototype classifier

\- SentenceTransformer nearest-neighbor classifier



Results were lower than the TF-IDF classifier, so they were not selected for the final system.



\## Final Agent Architecture



Customer Message

&#x20;       |

&#x20;       v

Text Normalization

&#x20;       |

&#x20;       v

TF-IDF + Logistic Regression

&#x20;       |

&#x20;       v

Intent + Confidence

&#x20;       |

&#x20;       +-----------------------+

&#x20;       |                       |

&#x20;       v                       v

Risk / Multi-Issue Checks   Confidence Check

&#x20;       |                       |

&#x20;       +-----------+-----------+

&#x20;                   |

&#x20;         +---------+---------+

&#x20;         |                   |

&#x20;         v                   v

&#x20;    AUTO\_HANDLE          ESCALATE

&#x20;         |                   |

&#x20;         v                   v

&#x20;  Safe Response        Human Support



\## Escalation Logic



The agent escalates when:



\- The message contains refund/payment dispute signals.

\- The message indicates account compromise or security issues.

\- The message involves legal or serious complaints.

\- The customer needs hardware repair/replacement/service.

\- Multiple distinct issue groups are detected.

\- Model confidence is below the configured threshold.



Risk rules take priority over model confidence. For example, a highly confident "charged twice" prediction is still escalated because billing disputes require human review.



\## Golden Evaluation Set



A 200-example Golden Set was created across the support intent taxonomy.



The set contains:



\- Customer message

\- Expected intent

\- Expected action

\- Escalation reason

\- Review status



Important evaluation caveat:



The Golden Set was constructed using rule-assisted candidate selection and review-oriented labeling rather than independent human annotation. Training labels were also generated using weak supervision. Therefore, the reported benchmark may overestimate real-world generalization performance and should be treated as a development benchmark rather than unbiased human-ground-truth evaluation.



\## Evaluation Results



| System | Accuracy | Macro F1 |

|---|---:|---:|

| Rule-based baseline | 64.0% | 58.2% |

| TF-IDF + Logistic Regression | 68.5% | 67.9% |

| Embedding prototype | 61.0% | 60.6% |

| Embedding KNN | 58.5% | 55.6% |



Final agent intent performance:



\- Accuracy: 68.5%

\- Macro F1: 67.9%

\- Weighted F1: 67.6%



The final routing benchmark achieved:



\- Action Accuracy: 76.5%

\- Escalation F1: 25.4%



Routing metrics are based on rule-assisted expected actions and should therefore not be interpreted as independently human-validated escalation quality.



\## Failure Analysis



The main intent classification errors were:



1\. Billing vs battery charging

&#x20;  - Words such as "charge" can refer to battery charging or a monetary charge.



2\. Device hardware vs connectivity

&#x20;  - Device-related complaints sometimes mention Wi-Fi, Bluetooth, or connection problems.



3\. Device hardware vs performance

&#x20;  - Slow, frozen, or malfunctioning devices can be described using hardware-related language.



4\. Billing vs App Store

&#x20;  - App Store purchases and billing complaints can share overlapping vocabulary.



5\. iOS update vs battery/performance

&#x20;  - Customers often report battery drain or performance degradation after an iOS update.



These errors show that short customer tweets often require context beyond individual keywords.



\## Repository Structure



```text

apple-support-agent/

│

├── data/

│   └── processed/

│       ├── apple\_agent.csv

│       ├── apple\_customers.csv

│       ├── apple\_training.csv

│       ├── golden\_candidates.csv

│       ├── golden\_set\_cleaned.csv

│       └── topic\_counts.csv

│

├── evaluation/

│   ├── agent\_evaluation.csv

│   ├── baseline\_rules\_results.csv

│   ├── baseline\_tfidf\_results.csv

│   ├── embedding\_knn\_results.csv

│   ├── embedding\_results.csv

│   ├── golden\_review.csv

│   ├── golden\_set.csv

│   └── tfidf\_final\_results.csv

│

├── src/

│   ├── 01\_extract\_apple\_support.py

│   ├── 02\_analyze\_intents.py

│   ├── 03\_create\_golden\_candidates.py

│   ├── 04\_prepare\_golden\_review.py

│   ├── 05\_finalize\_golden\_set.py

│   ├── 06\_baseline\_rules.py

│   ├── 07\_baseline\_tfidf.py

│   ├── 08\_create\_training\_labels.py

│   ├── 09\_proper\_tfidf.py

│   ├── 10\_embedding\_classifier.py

│   ├── 11\_embedding\_knn.py

│   ├── 12\_support\_agent.py

│   ├── 13\_evaluate\_agent.py

│   └── 14\_fix\_golden\_actions.py

│

└── README.md

## Reproduction

Install dependencies:

```bash
pip install pandas numpy scikit-learn sentence-transformers

Run the main pipelinein this order: 
python src/01_extract_apple_support.py
python src/02_analyze_intents.py
python src/03_create_golden_candidates.py
python src/04_prepare_golden_review.py
python src/05_finalize_golden_set.py
python src/06_baseline_rules.py
python src/08_create_training_labels.py
python src/09_proper_tfidf.py
python src/10_embedding_classifier.py
python src/11_embedding_knn.py
python src/14_fix_golden_actions.py
python src/13_evaluate_agent.py

Run the final support agent demo:
python src/12_support_agent.py
The raw twcs.csv dataset is intentionally excluded from the repository because of its large size. Place the dataset at data/twcs.csv before running the pipeline.

### 2. Then add this

```markdown
## Limitations and Future Improvements

- Training labels are weakly supervised rather than manually annotated.
- The Golden Set is rule-assisted and not independently human-annotated.
- Short tweets can be ambiguous between related intents.
- More independently annotated evaluation data would provide a stronger estimate of production performance.
- A larger manually reviewed dataset could improve difficult intent boundaries.
- Response generation could eventually be replaced with a retrieval or LLM-based system while keeping the safety and escalation layer.

## Key Takeaway

The project demonstrates an end-to-end customer support pipeline: dataset preparation, intent taxonomy design, baseline comparison, supervised classification, confidence-aware routing, safe response generation, evaluation, and failure analysis.
