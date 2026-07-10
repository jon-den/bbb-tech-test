# Agentic AI Investment System — 10-Minute Pitch

**BB Biotech — Task 3**

TODO: Add limitations and address in agentic ai
---

## The Investment Problem

Investment analysts spend hours doing what machines do poorly: synthesising heterogeneous, partially-contradictory signals across data sources that speak different languages — ICD-10 codes, ATC codes, PubMed MeSH terms, SEC filings, clinical trial endpoints, and payer policy documents. The bottleneck is not data access — it is structured reasoning across ontologically incompatible sources.

Our agentic AI system turns this around: a multi-agent pipeline that ingests all available evidence, resolves ontological conflicts, and produces a continuously-updated investment signal for a drug's adoption trajectory. The analyst's job shifts from synthesis to oversight.

---

## System Architecture

```
                      ┌──────────────────────────────────┐
                      │        INVESTMENT SIGNAL          │
                      │  adoption curve · TAM · risk      │
                      └──────────────┬───────────────────┘
                                     │
                      ┌──────────────▼───────────────────┐
                      │        SYNTHESIS AGENT            │
                      │  Reconciles conflicting sources   │
                      │  Flags low-confidence claims      │
                      │  Generates IC-ready summaries     │
                      └──────────┬────────────────────────┘
            ┌───────────────────┬┴──────────────────────┐
            ▼                   ▼                       ▼
   ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐
   │  CLAIMS AGENT  │  │  EVIDENCE AGENT│  │  MARKET AGENT        │
   │  patient-level │  │  PubMed + trial│  │  epidemiology · TAM  │
   │  modelling     │  │  evidence      │  │  payer policy        │
   └────────┬───────┘  └────────┬───────┘  └────────┬─────────────┘
            │                   │                   │
   ┌────────▼──────────────────▼───────────────────▼────────────┐
   │                    TOOL LAYER                                │
   │  Patient predictor · Code resolver · TAM estimator          │
   │  PubMed MCP · Clinicaltrials.gov · SEC filings              │
   └────────────────────────────────────────────────────────────┘
            │                   │                   │
   ┌────────▼───────┐  ┌────────▼───────┐  ┌───────▼────────────┐
   │ DATA CONNECTORS│  │ ONTOLOGY LAYER │  │ EXTERNAL SOURCES   │
   │ MarketScan     │  │ OMOP CDM       │  │ PubMed MCP         │
   │ IQVIA          │  │ ICD-10-CM      │  │ ClinicalTrials.gov │
   │ Truveta        │  │ ATC/NDC/RxNorm │  │ FDA label API      │
   │ TriNetX        │  │ SNOMED CT      │  │ SEC EDGAR          │
   │ Komodo/Veeva   │  └────────────────┘  └────────────────────┘
   └────────────────┘
```

---

## Component 1: Data Connectors

Claims data from a single payer is the single biggest limitation of our current analysis. The system integrates:

**MarketScan (IBM):** Largest commercial claims database (200M+ patient-years). Enables the ~100× sample size increase needed to support richer models — 5,000–10,000 Camzyos initiators instead of 149.

**IQVIA Longitudinal Patient Data (LPD):** Prescription data covering 70% of dispensed US prescriptions. Time-to-refill and days-supply data enables persistence modelling (the gap in our current analysis). Matches NPI-level prescribers to patients.

**Truveta:** Multi-system EHR network with structured clinical data. Key gap filler: LVOT gradient, NYHA class, echo parameters, and physician notes — the actual decision variables that billing codes cannot capture.

**TriNetX:** Research network aggregating EHR data from academic medical centres. Critical for rare diseases like oHCM where specialist centres dominate prescribing and routine claims miss the clinical context.

**Komodo Health / Veeva Pulse:** Prescriber-level longitudinal data. NPI-matched prescribing patterns, specialty, hospital affiliation, and REMS certification status. Closes the provider-adoption channel that patient-level claims cannot model.

**Integration challenge:** Each source uses different patient identifiers, date granularities, and coding conventions. The ontology layer (below) resolves this; the connector layer handles authentication, access agreements, and format normalisation.

---

## Component 2: Ontology Layer

Claims data is polyglot. A single drug appears as a brand name in ICD billing, a generic name in NDC (National Drug Code), an ATC code in pharmacovigilance, and a MeSH term in PubMed. Without resolution, a query for "beta-blocker exposure" retrieves different patients in each database.

The **Ontology Agent** maintains a live mapping across:

- **OMOP CDM (Common Data Model):** Standard vocabulary for harmonising claims data. Maps ICD-10-CM diagnoses, CPT procedures, NDC/RxNorm drugs, and LOINC labs to a unified concept space. Our current `atc.py` module is a simplified prototype of this.
- **ICD-10-CM hierarchy:** Disease code relationships (I421 is a leaf under I42 — Cardiomyopathy — under I30–I52 — Other heart diseases). Enables principled breadth-vs-specificity trade-offs in cohort definitions.
- **ATC hierarchy:** Anatomical Therapeutic Chemical classification for drugs. Level 2 = therapeutic class (C07 = beta-blockers), Level 5 = molecular entity. Enables drug-class features without hard-coding brand names.
- **RxNorm/NDC:** Resolves brand names (Toprol-XL) → generic (metoprolol succinate) → ingredient (metoprolol). Critical for prescription feature engineering across data sources.
- **SNOMED CT:** Clinical concept hierarchy for EHR linkage. Maps to ICD-10 for cross-source harmonisation.

The agent answers queries like: "Which patients have ever taken a cardiac myosin inhibitor?" and returns the correct concept IDs across all connected data sources, accounting for historical synonyms, re-classifications, and market withdrawals.

---

## Component 3: Real-World Evidence Agent (PubMed MCP)

The claims model is purely observational — it cannot distinguish "this drug combination is prescribed less because it's contraindicated" from "patients on this combination are systematically different." Literature grounds the model:

**PubMed MCP integration:**
- Live search against PubMed/MEDLINE for the drug + indication combination
- Extracts: study design, N, endpoint, HR/OR with CI, follow-up duration
- Flags contradictions between claims evidence and RCT evidence (e.g., if our model finds age is not predictive but EXPLORER-HCM showed age-stratified effects)
- Extracts prescribing contraindications, drug interactions, and monitoring requirements from FDA labels and clinical guidelines

**Use in our Camzyos example:**
- EXPLORER-HCM (N=251, Phase 3 RCT): primary endpoint, LVOT gradient reduction. The agent knows we cannot observe this in claims and flags the limitation automatically.
- MAVA-LTE: Long-term extension; informs persistence modelling priors.
- Competing drug trials (SEQUOIA-HCM, aficamten): The agent monitors trial registrations for competitor updates, alerting the investment team when new data changes the competitive landscape.

---

## Component 4: Tools

Individual callable tools that agents invoke to answer specific questions:

**Patient risk predictor:** Given a new patient's treatment history snapshot (as-of date, medication list, procedure history), returns the predicted 6/12/24-month Camzyos initiation probability using the fitted discrete-time hazard model. Updated monthly as new data arrives. Used by:
- Market sizing: sum predicted hazards across the untreated pool → expected new initiations per month
- Portfolio monitoring: flag patients approaching high-probability initiation window

**TAM estimator:** Inputs epidemiological estimates (oHCM prevalence × treatment-eligible fraction × payer coverage fraction × REMS reach fraction) and outputs a distribution over addressable patients. Explicitly propagates uncertainty — the IC receives a credible interval, not a point estimate. Automatically pulls updated US census and prevalence literature when recalculating.

**Code resolver:** Natural-language to code translation. "Patients who had an echocardiogram showing LVOT obstruction in the prior 6 months" → correct CPT/ICD code combinations across all connected databases, including historical codes and cross-walk mappings. Reduces the risk of manually omitting a synonym.

**Update predictor:** When new clinical data is published (trial readout, label update, guideline revision), recalculates the adoption model with updated feature engineering rules and generates a delta report: which patients' risk scores changed materially, and in which direction.

---

## What the System Cannot Do (and Why Honesty Matters)

The system is only as good as the data it ingests and the questions it is allowed to answer. We flag three systematic blind spots:

**1. Prescriber behaviour is not fully captured by claims.** The REMS certification effect — which cardiologists can prescribe Camzyos — is the dominant adoption driver in practice. Even with NPI-level data, prescriber decision logic (why a specific cardiologist chose Camzyos for this patient on this visit) is not recoverable from administrative data. Provider features reduce this gap; they don't close it.

**2. The model cannot predict adoption disruptions.** A competitor label expansion, a black-box warning addition, a payer prior-authorisation policy change — these shift the adoption curve discontinuously. The model extrapolates the current regime; it cannot model regime changes. These events are monitored by the Evidence Agent but cannot be quantified prospectively.

**3. Rare subgroups are undermodelled.** Paediatric HCM, patients without commercial insurance, and patients at non-specialist centres have insufficient sample size for reliable predictions. Predictions for these subgroups should be flagged as extrapolations, not estimates.

**4. Synthetic data cannot validate real-world performance.** The current model was developed on synthetic claims data modelled on realistic patterns, but it has not been validated against real payer data. All model outputs should be treated as directional until prospective validation is complete.

---

## Why This Investment Team Should Build It

**Current process:** Analyst reads 40-page label, manually codes a cohort, queries a database, builds a spreadsheet, presents to IC. 2–3 weeks per drug. Information decays between analysis and decision.

**With the system:** Analyst specifies the drug and indication. The system builds the cohort, runs the model, cross-references the literature, estimates the TAM, and flags the key uncertainties — in hours. The analyst reviews the output, interrogates the flagged uncertainties, and focuses their expertise on the judgment calls the system cannot make.

**Competitive moat:** The value is not the model — it is the integrated ontology layer and connected data sources. Each new data connection compounds the value of all existing connections. An analyst who has spent 18 months building OMOP mappings for 15 therapeutic areas has built a moat that takes competitors 18 months to replicate. The system becomes more valuable as BB Biotech develops more drug coverage, because every past drug informs the prior for the next one.

**Investment in the thesis:** We are already running a version of this. The current Camzyos analysis is Module 0 — a prototype with synthetic data. The production system replaces the synthetic data with real payer databases and adds the agentic orchestration layer. Timeline to prototype: 3–6 months. Cost: data access agreements ($100–500k/year for MarketScan/IQVIA), compute, and 2 ML engineers embedded in the investment team.
