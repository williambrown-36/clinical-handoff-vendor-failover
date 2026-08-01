"""Run a vendor-routed clinical handoff example."""

from privacy_safe_handoff import create_client, summarize_handoff


NOTE = """Patient reports a penicillin allergy. Medication reconciliation is pending.
Follow-up is scheduled for tomorrow. Escalate new breathing difficulty immediately."""


if __name__ == "__main__":
    print(summarize_handoff(create_client(), NOTE))
