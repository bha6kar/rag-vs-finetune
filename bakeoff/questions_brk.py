"""Eval set: 15 questions against Berkshire Hathaway 2024 Annual Report (FYE Dec 31, 2024).

Ground truth was hand-extracted from the PDF. The Berkshire AR is a richer
test for PageIndex than a clean 10-K because it bundles a discursive
shareholder letter, a 10-K, and appendices in one document.
"""

QUESTIONS = [
    # ---------- LOOKUP (5) ----------
    {
        "id": "BL1", "bucket": "lookup",
        "question": "Who are Berkshire Hathaway's named executive officers per the 2024 10-K, and what is each one's role?",
        "ground_truth": (
            "Warren E. Buffett — Chairman and CEO; Gregory E. Abel — Vice Chairman, "
            "Non-Insurance Operations; Ajit Jain — Vice Chairman, Insurance Operations; "
            "Marc D. Hamburg — Senior Vice-President and Chief Financial Officer."
        ),
    },
    {
        "id": "BL2", "bucket": "lookup",
        "question": "When did Berkshire last declare a cash dividend?",
        "ground_truth": "Berkshire has not declared a cash dividend since 1967.",
    },
    {
        "id": "BL3", "bucket": "lookup",
        "question": "What is the minimum cash and Treasury holdings threshold below which Berkshire will not repurchase its common stock?",
        "ground_truth": (
            "Berkshire will not repurchase its common stock if the repurchases would reduce "
            "the value of its consolidated cash, cash equivalents and U.S. Treasury Bills "
            "holdings to less than $30 billion."
        ),
    },
    {
        "id": "BL4", "bucket": "lookup",
        "question": "What were Berkshire Hathaway's total revenues for the year ended December 31, 2024?",
        "ground_truth": "$371,433 million in total revenues.",
    },
    {
        "id": "BL5", "bucket": "lookup",
        "question": "What was the year-end 2024 aggregate market value of Berkshire's five Japanese trading-company investments, and what was the cost basis?",
        "ground_truth": "Year-end 2024 market value of $23.5 billion against an aggregate cost (in dollars) of $13.8 billion.",
    },

    # ---------- MULTI-SECTION (6) ----------
    {
        "id": "BM1", "bucket": "multi_section",
        "question": "What insurance subsidiaries make up Berkshire's underwriting business, and what is the long-term underwriting goal?",
        "ground_truth": (
            "Berkshire's insurance and reinsurance businesses are GEICO, Berkshire Hathaway "
            "Primary Group (BH Primary) and Berkshire Hathaway Reinsurance Group (BHRG). "
            "BH Primary itself includes BHSI, RSUI/CapSpecialty, BHHC, MedPro, GUARD, NICO "
            "Primary, BH Direct and USLI. The long-term goal is to generate pre-tax "
            "underwriting earnings (premiums earned less losses/benefits incurred and "
            "underwriting expenses) in all business categories, except in the retroactive "
            "reinsurance and periodic-payment-annuity businesses."
        ),
    },
    {
        "id": "BM2", "bucket": "multi_section",
        "question": "How did GEICO's underwriting earnings change in 2024 vs 2023, and what were the main drivers?",
        "ground_truth": (
            "GEICO's pre-tax underwriting earnings rose to $7,813 million in 2024 from "
            "$3,635 million in 2023. The improvement reflected higher average premiums per "
            "auto policy, lower claims frequencies (property damage and collision), and "
            "improved operating efficiencies — partially offset by less favorable prior-year "
            "claim development, increases in average claims severities, and catastrophe "
            "losses (~$360 million from Hurricanes Helene and Milton)."
        ),
    },
    {
        "id": "BM3", "bucket": "multi_section",
        "question": "What significant catastrophe events affected Berkshire's 2024 insurance results, and what subsequent event is disclosed?",
        "ground_truth": (
            "2024 significant events included Hurricanes Milton and Helene. As a subsequent "
            "event, several wildfires broke out in Southern California in January 2025; "
            "Berkshire preliminarily estimated its insurance group could incur pre-tax losses "
            "of approximately $1.3 billion from these wildfires."
        ),
    },
    {
        "id": "BM4", "bucket": "multi_section",
        "question": "How does Berkshire govern cybersecurity risk at both the holding-company and subsidiary level?",
        "ground_truth": (
            "The Audit Committee of Berkshire's Board has overall oversight of the "
            "cybersecurity risk-management program, receives periodic reports on incidents "
            "and trends, and approves the internal-audit IT/cyber workplan (penetration "
            "testing, attack simulations, vulnerability assessments). Senior management of "
            "each Business Group is responsible for day-to-day protection of their systems, "
            "and each Business Group must report significant cybersecurity events up to "
            "Berkshire. Berkshire also notes exposure to cyberattacks on third-party service providers."
        ),
    },
    {
        "id": "BM5", "bucket": "multi_section",
        "question": "What does the 2024 chairman's letter disclose about the five Japanese trading-company investments and the matched yen-denominated borrowings?",
        "ground_truth": (
            "Berkshire began purchasing the five Japanese trading companies in July 2019. "
            "Year-end 2024 cost was $13.8 billion vs market value of $23.5 billion. Berkshire "
            "originally agreed to keep ownership below 10% per company, but the five companies "
            "agreed to moderately relax that ceiling. To approximate currency-neutrality, "
            "Berkshire has consistently increased its yen-denominated fixed-rate borrowings "
            "(no floaters), and recognized $2.3 billion of after-tax gains from yen "
            "borrowings cumulatively, including $850 million in 2024. Expected 2025 dividend "
            "income from the Japanese stakes is about $812 million vs ~$135 million of yen interest cost."
        ),
    },
    {
        "id": "BM6", "bucket": "multi_section",
        "question": "What regulatory and coal-revenue exposures does Berkshire disclose for BNSF?",
        "ground_truth": (
            "BNSF is subject to many laws and regulations covering rates, taxes, railroad "
            "operations, health/safety, labor, environment, etc., and changes can require "
            "significant capital expenditure and operating risk. BNSF derives significant "
            "revenue from transporting energy-related commodities — particularly coal — so "
            "policy changes that limit/restrict coal usage or displace coal could adversely "
            "affect revenue and earnings. As a common carrier, BNSF is also required to "
            "transport toxic-inhalation-hazard chemicals and other hazardous materials, "
            "exposing it to significant claims, losses and environmental remediation obligations."
        ),
    },

    # ---------- NUMERIC / TABLE (4) ----------
    {
        "id": "BN1", "bucket": "numeric_table",
        "question": "What were GEICO's pre-tax underwriting earnings in 2024?",
        "ground_truth": "$7,813 million.",
    },
    {
        "id": "BN2", "bucket": "numeric_table",
        "question": "What were Berkshire Hathaway's total assets as of December 31, 2024?",
        "ground_truth": "$1,153,881 million (about $1.154 trillion).",
    },
    {
        "id": "BN3", "bucket": "numeric_table",
        "question": "What was net earnings attributable to Berkshire shareholders in 2024 (and how did it compare with 2023)?",
        "ground_truth": "$88,995 million in 2024 vs $96,223 million in 2023.",
    },
    {
        "id": "BN4", "bucket": "numeric_table",
        "question": "What were Berkshire's short-term investments in U.S. Treasury Bills at December 31, 2024 vs December 31, 2023?",
        "ground_truth": "$286,472 million at December 31, 2024 vs $129,619 million at December 31, 2023 (more than doubled).",
    },
]

assert len(QUESTIONS) == 15
assert sum(1 for q in QUESTIONS if q["bucket"] == "lookup") == 5
assert sum(1 for q in QUESTIONS if q["bucket"] == "multi_section") == 6
assert sum(1 for q in QUESTIONS if q["bucket"] == "numeric_table") == 4
