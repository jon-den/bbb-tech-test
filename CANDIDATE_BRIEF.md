# ML Take-Home Assessment

## Clinical Context

Obstructive hypertrophic cardiomyopathy (oHCM) is a condition in which the heart muscle thickens abnormally, obstructing blood flow out of the left ventricle. Patients experience progressive symptoms including shortness of breath, chest pain, syncope, and exercise intolerance. Management typically involves medical therapy with escalation based on symptom burden, and in refractory cases, invasive interventions.

**Camzyos (mavacamten)** is a first-in-class cardiac myosin inhibitor approved by the FDA in April 2022 for the treatment of symptomatic oHCM. It is distributed through a restricted program (REMS) and is primarily prescribed at academic medical centres and specialised HCM clinics. Initiation of Camzyos is directly observable in prescription claims data.

## The Data

You are provided with claims data on approximately 30,000 cardiac patients observed over several years. The data includes diagnoses, procedures, prescriptions, and enrollment information. A subset of these patients have oHCM. Camzyos launched during the observation period — you can identify who initiated it in the prescription data.

This dataset is drawn from a US commercial claims database. The ~30,000 cardiac patients here represent all patients in that database with at least one cardiac-related diagnosis during the observation period. The selection criteria for the underlying database are not specified.

The data is synthetic but modelled on realistic clinical patterns. The dataset is described in `DATA_README.md`. All clinical codes are documented in `code_dictionary.csv`.

## Tasks

As part of an investment case, we want to understand Camzyos adoption — who is initiating, what characterises them, and how uptake is likely to evolve.

1. **Model Camzyos adoption.** Which patients are likely to initiate, and when? How is uptake evolving over time? Quantify your uncertainty.

2. **Estimate the total addressable US patient population for Camzyos -- today and over time.** Several imperfect information sources are relevant here — this dataset is one of them. Be explicit about how you combine sources, how you weight them when they disagree, and how this shapes the uncertainty in your final estimate.

3. **The future.** We're building an agentic AI investment system. A US claims dataset of this kind — roughly 30 million insured lives — would be one input among several. How would it fit into the system, what other datasets would you combine it with, and what capabilities would matter most? Give us your 10 minute pitch!

## Deliverables

- **Code**: Reproducible analysis (Python preferred). Include a `requirements.txt` or equivalent. Your code should run end-to-end from the provided data files.
- **Write-up**: Explain your methodology, assumptions, results, and limitations. Communicates your reasoning clearly. Tell us what you would change with more time or more data.
- **Any material supporting your pitch**

## What Happens Next

If selected for the next round we will invite you to discuss your technical work and hear your pitch. 

## A Note on LLMs

You're welcome to use LLMs in your work, as you would in the actual role. We're interested in the decisions you make and the reasoning behind them, not whether you typed every line yourself. We are also interested in how you use LLMs and agents to do yoru work. 
