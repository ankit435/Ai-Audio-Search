"""Hand-authored labeled query set for the golden dataset (Task 5 /
Evaluation Plan §2).

Each entry references `expected_turn_indices` — indices into that
file's `turns.json` ground truth (see `golden_conversations.py`) —
rather than exact transcript text, because real Whisper transcription
introduces minor wording differences (e.g. "back-and-services" for
"backend services"). `scripts/golden_eval_utils.py` resolves each
turn_index to the actual persisted `chunk_id`(s) via time-range overlap
against the real ingested data, which is robust to those ASR quirks and
to however the merge-and-split chunker grouped turns.

`query_type` is either "keyword" (answerable by near-exact phrase match)
or "semantic" (a paraphrase/conceptual question that does not reuse the
transcript's exact wording) — split out for per-type recall@k reporting
per the Evaluation Plan.

Every query below was authored by a human directly against the known
script content (no LLM was used to draft these, consistent with the
plan's "human-verified" requirement — see AGENT_LOG.md).
"""

from __future__ import annotations

QUERY_SET: list[dict] = [
    # ---- hiring_interview.wav ----
    {"file": "hiring_interview.wav", "query": "token bucket algorithm", "query_type": "keyword", "expected_turn_indices": [11]},
    {"file": "hiring_interview.wav", "query": "rate limiter for a public API", "query_type": "keyword", "expected_turn_indices": [10]},
    {"file": "hiring_interview.wav", "query": "message queues payments company", "query_type": "keyword", "expected_turn_indices": [1]},
    {"file": "hiring_interview.wav", "query": "one design doc per quarter", "query_type": "keyword", "expected_turn_indices": [17]},
    {"file": "hiring_interview.wav", "query": "two to three weeks scheduling", "query_type": "keyword", "expected_turn_indices": [9]},
    {"file": "hiring_interview.wav", "query": "system design round hiring manager", "query_type": "keyword", "expected_turn_indices": [7]},
    {"file": "hiring_interview.wav", "query": "how do you troubleshoot a service that is intermittently slow", "query_type": "semantic", "expected_turn_indices": [4, 5]},
    {"file": "hiring_interview.wav", "query": "what makes this company's engineering culture appealing", "query_type": "semantic", "expected_turn_indices": [16]},
    {"file": "hiring_interview.wav", "query": "how would you keep request limits consistent across regions", "query_type": "semantic", "expected_turn_indices": [14]},
    {"file": "hiring_interview.wav", "query": "what does the interview process look like end to end", "query_type": "semantic", "expected_turn_indices": [7]},
    {"file": "hiring_interview.wav", "query": "how would you find the root cause of a slow request across services", "query_type": "semantic", "expected_turn_indices": [6]},
    {"file": "hiring_interview.wav", "query": "why is this candidate a good fit for the platform team", "query_type": "semantic", "expected_turn_indices": [3]},

    # ---- product_roadmap.wav ----
    {"file": "product_roadmap.wav", "query": "caching layer in front of the database", "query_type": "keyword", "expected_turn_indices": [2]},
    {"file": "product_roadmap.wav", "query": "migrate billing off the deprecated framework", "query_type": "keyword", "expected_turn_indices": [10]},
    {"file": "product_roadmap.wav", "query": "public changelog page", "query_type": "keyword", "expected_turn_indices": [12]},
    {"file": "product_roadmap.wav", "query": "thirty percent since the last release", "query_type": "keyword", "expected_turn_indices": [1]},
    {"file": "product_roadmap.wav", "query": "retention is the metric leadership cares about", "query_type": "keyword", "expected_turn_indices": [8]},
    {"file": "product_roadmap.wav", "query": "why is the mobile app redesign a high priority", "query_type": "semantic", "expected_turn_indices": [3]},
    {"file": "product_roadmap.wav", "query": "how will the old billing framework risk be addressed", "query_type": "semantic", "expected_turn_indices": [10, 11]},
    {"file": "product_roadmap.wav", "query": "what happened to the sales team's dashboard request", "query_type": "semantic", "expected_turn_indices": [6]},
    {"file": "product_roadmap.wav", "query": "what is the plan to fix slow search performance", "query_type": "semantic", "expected_turn_indices": [0, 2]},
    {"file": "product_roadmap.wav", "query": "what gets deprioritized this quarter", "query_type": "semantic", "expected_turn_indices": [16]},

    # ---- customer_support_call.wav ----
    {"file": "customer_support_call.wav", "query": "billing retry job ran twice", "query_type": "keyword", "expected_turn_indices": [5]},
    {"file": "customer_support_call.wav", "query": "refund within three to five business days", "query_type": "keyword", "expected_turn_indices": [8]},
    {"file": "customer_support_call.wav", "query": "forward the fee notice from your bank", "query_type": "keyword", "expected_turn_indices": [12]},
    {"file": "customer_support_call.wav", "query": "disabled the faulty retry job", "query_type": "keyword", "expected_turn_indices": [14]},
    {"file": "customer_support_call.wav", "query": "case number for your records", "query_type": "keyword", "expected_turn_indices": [10]},
    {"file": "customer_support_call.wav", "query": "why was I charged twice this month", "query_type": "semantic", "expected_turn_indices": [1]},
    {"file": "customer_support_call.wav", "query": "how will the bank overdraft fee be handled", "query_type": "semantic", "expected_turn_indices": [3, 12]},
    {"file": "customer_support_call.wav", "query": "will this billing problem happen again next month", "query_type": "semantic", "expected_turn_indices": [13, 14]},
    {"file": "customer_support_call.wav", "query": "was this problem caused by something the customer did wrong", "query_type": "semantic", "expected_turn_indices": [6, 7]},

    # ---- architecture_review.wav ----
    {"file": "architecture_review.wav", "query": "message broker instead of direct HTTP calls", "query_type": "keyword", "expected_turn_indices": [3, 4]},
    {"file": "architecture_review.wav", "query": "idempotency keys on each notification", "query_type": "keyword", "expected_turn_indices": [14]},
    {"file": "architecture_review.wav", "query": "shard by user id or time series store", "query_type": "keyword", "expected_turn_indices": [10]},
    {"file": "architecture_review.wav", "query": "alerting on queue depth and consumer lag", "query_type": "keyword", "expected_turn_indices": [15]},
    {"file": "architecture_review.wav", "query": "email push notifications and SMS subscribe to events", "query_type": "keyword", "expected_turn_indices": [2]},
    {"file": "architecture_review.wav", "query": "why avoid direct service to service calls", "query_type": "semantic", "expected_turn_indices": [4]},
    {"file": "architecture_review.wav", "query": "how does the system avoid losing notifications when a consumer is down", "query_type": "semantic", "expected_turn_indices": [12]},
    {"file": "architecture_review.wav", "query": "how are duplicate notifications prevented", "query_type": "semantic", "expected_turn_indices": [14]},
    {"file": "architecture_review.wav", "query": "how easy is it to add a brand new notification channel", "query_type": "semantic", "expected_turn_indices": [5]},
    {"file": "architecture_review.wav", "query": "what happens if notification traffic grows a lot", "query_type": "semantic", "expected_turn_indices": [9, 10]},

    # ---- onboarding_session.wav ----
    {"file": "onboarding_session.wav", "query": "thirty days from your start date to enroll", "query_type": "keyword", "expected_turn_indices": [6]},
    {"file": "onboarding_session.wav", "query": "once every six weeks on call rotation", "query_type": "keyword", "expected_turn_indices": [9]},
    {"file": "onboarding_session.wav", "query": "review in under twenty minutes", "query_type": "keyword", "expected_turn_indices": [14]},
    {"file": "onboarding_session.wav", "query": "at least one approval before merging", "query_type": "keyword", "expected_turn_indices": [12]},
    {"file": "onboarding_session.wav", "query": "access to the internal engineering wiki", "query_type": "keyword", "expected_turn_indices": [1, 2]},
    {"file": "onboarding_session.wav", "query": "how do the two health insurance plans differ", "query_type": "semantic", "expected_turn_indices": [4]},
    {"file": "onboarding_session.wav", "query": "what should a new hire expect from being on call", "query_type": "semantic", "expected_turn_indices": [7, 9]},
    {"file": "onboarding_session.wav", "query": "how severe are the incidents engineers get paged for", "query_type": "semantic", "expected_turn_indices": [11]},
    {"file": "onboarding_session.wav", "query": "what is expected during code review", "query_type": "semantic", "expected_turn_indices": [12, 14]},
]
