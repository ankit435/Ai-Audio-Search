"""Hand-authored source conversations for the synthetic golden dataset.

Each conversation is a list of (speaker_id, text) turns. Content is
original, written for this project — not copied from any copyrighted
source — so it is safe to synthesize into audio and redistribute.

Two speakers per conversation, matching the plan's "unique speaker
pairs" requirement. Turns are deliberately varied in length and phrasing
so the merge-and-split chunker has realistic material to work with, and
each conversation covers a distinct topic so keyword vs. semantic
queries can be meaningfully distinguished during evaluation.
"""

from __future__ import annotations

Turn = tuple[str, str]

HIRING_INTERVIEW: list[Turn] = [
    ("SPEAKER_00", "Thanks for joining today. Could you start by telling me about your experience with distributed systems?"),
    ("SPEAKER_01", "Sure. I spent the last four years building backend services for a payments company, mostly working with event driven architectures and message queues."),
    ("SPEAKER_00", "That's great. We're actually hiring a senior backend engineer for our platform team right now, and message queue experience is exactly what we need."),
    ("SPEAKER_01", "I saw that in the job posting. I'm particularly interested in the platform team because of the focus on reliability and observability."),
    ("SPEAKER_00", "Let's talk about a specific scenario. Suppose a downstream service starts timing out intermittently. How would you debug that?"),
    ("SPEAKER_01", "First I'd check the service's latency dashboards and error rates, then look at whether the timeouts correlate with deploys or traffic spikes."),
    ("SPEAKER_01", "If nothing obvious shows up, I'd add distributed tracing to follow a single request across services and find where the delay actually happens."),
    ("SPEAKER_00", "Good answer. Our hiring process includes a technical interview like this one, followed by a system design round, and then a conversation with the hiring manager."),
    ("SPEAKER_01", "That sounds reasonable. How long does the whole process usually take from first interview to an offer?"),
    ("SPEAKER_00", "Typically two to three weeks, assuming scheduling goes smoothly on both sides."),
    ("SPEAKER_00", "Now let's do a quick system design question. How would you design a rate limiter for a public API?"),
    ("SPEAKER_01", "I'd use a token bucket algorithm backed by a fast in-memory store like Redis, with a small amount of local caching to reduce round trips."),
    ("SPEAKER_01", "Each client would get a bucket that refills at a fixed rate, and requests are rejected once the bucket is empty."),
    ("SPEAKER_00", "What would you do differently if the rate limiter needed to work across multiple data centers?"),
    ("SPEAKER_01", "I'd probably accept some looseness in the global limit and use per region buckets with periodic synchronization, rather than trying to enforce a perfectly exact global count."),
    ("SPEAKER_00", "That's a solid tradeoff. One more thing, what interests you most about our engineering culture?"),
    ("SPEAKER_01", "I really like that the team writes public design docs before big changes. It makes it easier to give feedback early instead of after the code is already written."),
    ("SPEAKER_00", "We take that seriously. Every senior engineer here is expected to write at least one design doc per quarter for anything non trivial."),
    ("SPEAKER_01", "That matches how I like to work. Thanks again for the detailed walkthrough of the hiring process."),
    ("SPEAKER_00", "Thanks for your time today. We'll follow up within a week with next steps for the system design round."),
]

PRODUCT_ROADMAP: list[Turn] = [
    ("SPEAKER_00", "Let's kick off the quarterly roadmap review. Our top priority for next quarter is improving search latency across the platform."),
    ("SPEAKER_01", "Agreed. Support tickets about slow search results have gone up almost thirty percent since the last release."),
    ("SPEAKER_00", "Right, and the engineering team already proposed adding a caching layer in front of the database to cut down repeated queries."),
    ("SPEAKER_01", "Caching should help a lot, but we also need to prioritize the mobile app redesign that customers have been asking about for months."),
    ("SPEAKER_00", "That's fair. Let's timebox the redesign to six weeks and make sure it doesn't block the search latency work."),
    ("SPEAKER_01", "Sounds workable. What about the analytics dashboard the sales team requested last quarter?"),
    ("SPEAKER_00", "I'd like to push that to the following quarter unless someone has a strong argument for doing it sooner."),
    ("SPEAKER_01", "Sales has been vocal about it, but I think search performance and the mobile redesign are more urgent for retention."),
    ("SPEAKER_00", "Agreed, retention is the metric leadership cares about most this cycle."),
    ("SPEAKER_00", "Let's also talk about technical debt. The billing service is still running on the old framework version."),
    ("SPEAKER_01", "Yes, we should allocate at least one engineer for two weeks to migrate billing off the deprecated framework before it loses support."),
    ("SPEAKER_00", "Makes sense, security has flagged that as a risk a couple of times already."),
    ("SPEAKER_01", "One more item, the product roadmap review from marketing wants a public changelog page so customers can see what shipped each week."),
    ("SPEAKER_00", "I like that idea. It's low effort and it directly addresses customer complaints about not knowing what changed between releases."),
    ("SPEAKER_01", "I'll draft a one page proposal for the changelog page and share it by Friday."),
    ("SPEAKER_00", "Great. To summarize, our priorities are search latency improvements, the mobile redesign, the billing framework migration, and the changelog page."),
    ("SPEAKER_01", "The analytics dashboard and anything else not mentioned gets pushed to next quarter unless something urgent comes up."),
    ("SPEAKER_00", "Perfect, let's get this roadmap into the planning doc and share it with the wider team tomorrow morning."),
]

CUSTOMER_SUPPORT_CALL: list[Turn] = [
    ("SPEAKER_00", "Thanks for calling support, I understand you're seeing an unexpected charge on your account. Can you tell me more?"),
    ("SPEAKER_01", "Yes, I was charged twice this month for the same subscription plan, and I only have one active account."),
    ("SPEAKER_00", "I'm sorry about that. Let me pull up your billing history and take a look at what happened."),
    ("SPEAKER_01", "I really need this resolved quickly because the double charge overdrew my account and caused a bank fee."),
    ("SPEAKER_00", "I completely understand, and I'll make sure we fix both the duplicate charge and reimburse the bank fee if it was caused by our error."),
    ("SPEAKER_00", "Looking at the history, it seems a billing retry job ran twice due to a temporary system issue on our end."),
    ("SPEAKER_01", "So this is a known bug, not something I did wrong on my side?"),
    ("SPEAKER_00", "Correct, this is an internal billing system issue, not anything related to your account settings."),
    ("SPEAKER_00", "I'm processing a refund for the duplicate charge right now, and it should appear back in your account within three to five business days."),
    ("SPEAKER_01", "Thank you, that's a relief. Can you also send me an email confirming the refund and the reason for the double charge?"),
    ("SPEAKER_00", "Absolutely, I'll send that confirmation email as soon as we hang up, along with a case number for your records."),
    ("SPEAKER_01", "I appreciate that. What about the bank fee, will that also be reimbursed automatically?"),
    ("SPEAKER_00", "For the bank fee, please forward us the fee notice from your bank and we'll issue a separate reimbursement for that amount."),
    ("SPEAKER_01", "Okay, I'll send that over today. Is there anything I need to do to prevent this from happening again next month?"),
    ("SPEAKER_00", "No action needed on your side, our engineering team has already disabled the faulty retry job that caused the duplicate billing."),
    ("SPEAKER_01", "That's good to hear. Thank you for explaining everything so clearly."),
    ("SPEAKER_00", "You're very welcome, and again I apologize for the inconvenience caused by the duplicate charge."),
]

ARCHITECTURE_REVIEW: list[Turn] = [
    ("SPEAKER_00", "Today we're reviewing the proposed architecture for the new notification service. Walk us through the design."),
    ("SPEAKER_01", "The core idea is a queue based system where every notification event gets published to a message broker before being delivered."),
    ("SPEAKER_01", "Downstream consumers, like email, push notifications, and SMS, each subscribe to the events relevant to their channel."),
    ("SPEAKER_00", "Why did you choose a message broker instead of having services call each other directly over HTTP?"),
    ("SPEAKER_01", "Direct HTTP calls would tightly couple every producer to every consumer, so adding a new notification channel would require changes everywhere."),
    ("SPEAKER_01", "With a broker in the middle, we can add a new consumer, like a Slack integration, without touching the producers at all."),
    ("SPEAKER_00", "That makes sense for extensibility. What database are you planning to use for storing notification history?"),
    ("SPEAKER_01", "We're proposing Postgres, since we already run it in production and the notification volume doesn't require a specialized store yet."),
    ("SPEAKER_00", "Have you thought about how this system will scale if notification volume grows tenfold?"),
    ("SPEAKER_01", "Yes, the message broker itself can scale horizontally by adding more partitions, and consumers can scale independently based on their own load."),
    ("SPEAKER_01", "Postgres could become a bottleneck eventually, but we can shard by user id or move to a time series store later if needed."),
    ("SPEAKER_00", "What about failure handling? What happens if the email consumer goes down for an hour?"),
    ("SPEAKER_01", "Since it's queue based, events just accumulate in the broker until the consumer comes back online, so nothing gets lost."),
    ("SPEAKER_00", "Good, that's an important reliability property. How are you handling duplicate delivery if a consumer crashes mid processing?"),
    ("SPEAKER_01", "We're using at least once delivery semantics with idempotency keys on each notification, so retries don't create duplicate emails."),
    ("SPEAKER_00", "This architecture looks solid overall. My main concern is monitoring, make sure we have alerting on queue depth and consumer lag."),
    ("SPEAKER_01", "Agreed, I'll add dashboards for queue depth, consumer lag, and delivery latency before we roll this out to production."),
]

ONBOARDING_SESSION: list[Turn] = [
    ("SPEAKER_00", "Welcome to the team. Today we'll walk through onboarding, starting with your equipment and account setup."),
    ("SPEAKER_01", "Great, I already have my laptop, but I haven't been able to log into the internal engineering wiki yet."),
    ("SPEAKER_00", "I'll get IT to grant you access to the wiki right after this call. It has all our engineering guidelines and architecture docs."),
    ("SPEAKER_01", "Perfect. Can you also tell me about the benefits package, especially the health insurance options?"),
    ("SPEAKER_00", "Sure, we offer two health insurance plans, a standard plan with no premium and a premium plan with lower deductibles for a monthly fee."),
    ("SPEAKER_01", "How do I choose between the two plans, and is there a deadline to enroll?"),
    ("SPEAKER_00", "You have thirty days from your start date to enroll, and HR will send you a comparison sheet showing the coverage differences."),
    ("SPEAKER_00", "Let's also cover the on call rotation, since your team participates in production support."),
    ("SPEAKER_01", "I saw that mentioned in the offer letter. How often would I be on call?"),
    ("SPEAKER_00", "Engineers rotate on call roughly once every six weeks, for a full week at a time, with a senior engineer as backup."),
    ("SPEAKER_01", "That sounds manageable. What's the typical severity of incidents I might get paged for?"),
    ("SPEAKER_00", "Most pages are low severity, like a slow batch job, but occasionally you'll get paged for something customer facing that needs quick attention."),
    ("SPEAKER_00", "One more important thing, our code review process. Every pull request needs at least one approval before merging."),
    ("SPEAKER_01", "Is there a preferred size for pull requests, or is that left up to the individual engineer?"),
    ("SPEAKER_00", "We generally recommend keeping pull requests small enough to review in under twenty minutes, since large diffs tend to get less thorough review."),
    ("SPEAKER_01", "That's good guidance, thank you. I think I have a much clearer picture of my first few weeks now."),
    ("SPEAKER_00", "Great, feel free to message me anytime this week if anything about benefits, on call, or the wiki access is unclear."),
]

# name -> (turns, voice_a, voice_b)
CONVERSATIONS: dict[str, list[Turn]] = {
    "hiring_interview": HIRING_INTERVIEW,
    "product_roadmap": PRODUCT_ROADMAP,
    "customer_support_call": CUSTOMER_SUPPORT_CALL,
    "architecture_review": ARCHITECTURE_REVIEW,
    "onboarding_session": ONBOARDING_SESSION,
}

VOICE_BY_SPEAKER = {"SPEAKER_00": "Samantha", "SPEAKER_01": "Daniel"}
