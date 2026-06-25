"""Eval set: 25 questions against the Apple FY2024 (FYE Sep 28, 2024) 10-K.

Ground truth was extracted by hand from the PDF. `expected_pages` references
the document's printed page numbers (which differ from PDF page indices).
"""

QUESTIONS = [
    # ---------- LOOKUP (8) — single, mostly verbatim facts ----------
    {
        "id": "L1", "bucket": "lookup",
        "question": "What was Apple's total net sales for fiscal year 2024?",
        "ground_truth": "$391,035 million (i.e. about $391.0 billion).",
    },
    {
        "id": "L2", "bucket": "lookup",
        "question": "What was Apple's net income in fiscal year 2024?",
        "ground_truth": "$93,736 million.",
    },
    {
        "id": "L3", "bucket": "lookup",
        "question": "Where is Apple's corporate headquarters located?",
        "ground_truth": "One Apple Park Way, Cupertino, California 95014.",
    },
    {
        "id": "L4", "bucket": "lookup",
        "question": "As of October 18, 2024, how many shareholders of record did Apple report?",
        "ground_truth": "23,301 shareholders of record.",
    },
    {
        "id": "L5", "bucket": "lookup",
        "question": "What was the size of the new share repurchase program Apple authorized in May 2024?",
        "ground_truth": "Up to $110 billion.",
    },
    {
        "id": "L6", "bucket": "lookup",
        "question": "What is Apple's quarterly cash dividend per share as of fiscal year-end 2024?",
        "ground_truth": "$0.25 per share (raised from $0.24 in May 2024).",
    },
    {
        "id": "L7", "bucket": "lookup",
        "question": "How does the 10-K describe Apple Intelligence?",
        "ground_truth": "A personal intelligence system that uses generative models, announced in the third quarter of fiscal 2024 alongside iOS 18 / macOS Sequoia / iPadOS 18 / watchOS 11 / visionOS 2 / tvOS 18.",
    },
    {
        "id": "L8", "bucket": "lookup",
        "question": "Who has overall responsibility at Apple for managing cybersecurity risks?",
        "ground_truth": "The Head of Corporate Information Security, who leads Apple's Information Security team and reports under the Cybersecurity governance structure.",
    },

    # ---------- MULTI-SECTION (10) — synthesis across parts of the doc ----------
    {
        "id": "M1", "bucket": "multi_section",
        "question": "What major antitrust actions in the U.S. and EU did Apple disclose in fiscal 2024, and what is at stake in each?",
        "ground_truth": (
            "Three threads. (1) EU Digital Markets Act: the European Commission opened "
            "noncompliance investigations under Article 5(4) and Article 6(3) on Mar 25, 2024, "
            "issued preliminary findings on Article 5(4) and opened a third investigation on "
            "Jun 24, 2024; potential fines of up to 10% of annual worldwide net sales. "
            "(2) U.S. DOJ antitrust lawsuit filed Mar 21, 2024 alleging monopolization in "
            "performance smartphones / smartphones markets, seeking equitable relief. "
            "(3) Epic Games — California District Court found certain App Store Review "
            "Guidelines violated California UCL and issued an injunction; Apple implemented a "
            "compliance plan Jan 16, 2024 and on Sep 30, 2024 moved to narrow or vacate the injunction."
        ),
    },
    {
        "id": "M2", "bucket": "multi_section",
        "question": "How did Greater China and Japan segment net sales perform in 2024 vs 2023, and what drove the difference?",
        "ground_truth": (
            "Greater China net sales fell 8% (from $72,559M to $66,952M), driven primarily by "
            "lower iPhone and iPad sales, with the weak renminbi vs the U.S. dollar an unfavorable "
            "year-over-year impact. Japan net sales rose 3% (from $24,257M to $25,052M), driven "
            "primarily by higher iPhone sales, despite an unfavorable currency impact from the weak yen."
        ),
    },
    {
        "id": "M3", "bucket": "multi_section",
        "question": "How did R&D and SG&A operating expenses change in 2024 vs 2023, and what drove the changes?",
        "ground_truth": (
            "R&D rose 5% to $31,370M (from $29,915M), driven primarily by increases in "
            "headcount-related expenses. SG&A rose 5% to $26,097M (from $24,932M), an increase "
            "of $1.2 billion. Total operating expenses rose 5% to $57,467M."
        ),
    },
    {
        "id": "M4", "bucket": "multi_section",
        "question": "What risks does Apple disclose specifically related to artificial intelligence and machine learning?",
        "ground_truth": (
            "Two angles. Product/safety: introducing complex new technologies such as AI features "
            "can increase safety risks, including exposing users to harmful, inaccurate, or other "
            "negative content and experiences, and Apple may not detect or fix all issues. "
            "Legal/regulatory: machine learning and artificial intelligence are listed among the "
            "areas of law to which Apple's business is subject (alongside intellectual property, "
            "digital platforms, telecommunications, content, etc.)."
        ),
    },
    {
        "id": "M5", "bucket": "multi_section",
        "question": "What new hardware products did Apple announce in the fourth quarter of fiscal 2024?",
        "ground_truth": "iPhone 16, iPhone 16 Plus, iPhone 16 Pro, iPhone 16 Pro Max, Apple Watch Series 10, and AirPods 4.",
    },
    {
        "id": "M6", "bucket": "multi_section",
        "question": "What drove the year-over-year growth in Services revenue in fiscal 2024?",
        "ground_truth": (
            "Services net sales rose 13% to $96,169M (from $85,200M), driven primarily by higher "
            "net sales from advertising, the App Store, and cloud services. Services revenue also "
            "includes amortization of the deferred value of services bundled in the sales price of certain products."
        ),
    },
    {
        "id": "M7", "bucket": "multi_section",
        "question": "Why was Apple's effective tax rate in fiscal 2024 higher than in 2023?",
        "ground_truth": (
            "The effective tax rate was 24.1% in 2024 vs 14.7% in 2023. The increase was driven "
            "primarily by a one-time net income tax charge of $10.2 billion related to the State "
            "Aid Decision, plus a higher effective tax rate on foreign earnings and lower tax "
            "benefits from share-based compensation."
        ),
    },
    {
        "id": "M8", "bucket": "multi_section",
        "question": "Describe Apple's State Aid Decision tax obligation and how it was funded as of fiscal year-end 2024.",
        "ground_truth": (
            "As of September 28, 2024, Apple had an obligation to pay €14.2 billion (~$15.8 "
            "billion) to Ireland in connection with the State Aid Decision, all of which was "
            "expected to be paid within 12 months. The funds necessary to settle the obligation "
            "were held in escrow as of September 28, 2024 and were restricted from general use."
        ),
    },
    {
        "id": "M9", "bucket": "multi_section",
        "question": "What new accounting standards updates will Apple adopt, and when?",
        "ground_truth": (
            "ASU 2023-09 (Income Taxes — improved disclosures): Apple will adopt in Q4 of fiscal "
            "2026 using a prospective transition method. ASU 2023-07 (Segment Reporting — improved "
            "reportable segment disclosures): Apple will adopt in Q4 of fiscal 2025 using a "
            "retrospective transition method."
        ),
    },
    {
        "id": "M10", "bucket": "multi_section",
        "question": "How does Apple measure foreign exchange risk on its derivative positions, and what is the disclosed maximum one-day loss?",
        "ground_truth": (
            "Apple uses a value-at-risk (VAR) model based on Monte Carlo simulation applied to "
            "its foreign currency derivative positions. With 95% confidence, the estimated maximum "
            "one-day loss in fair value was $538 million as of September 28, 2024 (vs $669 million "
            "as of September 30, 2023)."
        ),
    },

    # ---------- NUMERIC / TABLE (7) ----------
    {
        "id": "N1", "bucket": "numeric_table",
        "question": "What were iPhone net sales in fiscal 2024, and how did they compare to fiscal 2023?",
        "ground_truth": "iPhone net sales were $201,183 million in 2024 vs $200,583 million in 2023 — essentially flat (0% change).",
    },
    {
        "id": "N2", "bucket": "numeric_table",
        "question": "What were Services net sales in fiscal 2024 and fiscal 2023?",
        "ground_truth": "Services net sales were $96,169 million in 2024 and $85,200 million in 2023 (up 13%).",
    },
    {
        "id": "N3", "bucket": "numeric_table",
        "question": "What was Apple's total cost of sales in fiscal 2024 vs fiscal 2023?",
        "ground_truth": "Total cost of sales was $210,352 million in 2024 vs $214,137 million in 2023.",
    },
    {
        "id": "N4", "bucket": "numeric_table",
        "question": "What was the Products gross margin percentage for fiscal years 2024, 2023, and 2022?",
        "ground_truth": "Products gross margin percentage was 37.2% in 2024, 36.5% in 2023, and 36.3% in 2022.",
    },
    {
        "id": "N5", "bucket": "numeric_table",
        "question": "What was Apple's diluted earnings per share in fiscal 2024 and fiscal 2023?",
        "ground_truth": "Diluted EPS was $6.08 in fiscal 2024 and $6.13 in fiscal 2023.",
    },
    {
        "id": "N6", "bucket": "numeric_table",
        "question": "What was cash generated by operating activities in fiscal 2024?",
        "ground_truth": "$118,254 million.",
    },
    {
        "id": "N7", "bucket": "numeric_table",
        "question": "What were Apple's total assets as of September 28, 2024?",
        "ground_truth": "$364,980 million.",
    },
]

assert len(QUESTIONS) == 25
assert sum(1 for q in QUESTIONS if q["bucket"] == "lookup") == 8
assert sum(1 for q in QUESTIONS if q["bucket"] == "multi_section") == 10
assert sum(1 for q in QUESTIONS if q["bucket"] == "numeric_table") == 7
