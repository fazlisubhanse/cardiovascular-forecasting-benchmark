# Data access and provenance

## What is known

The study used a supplied annual table with 54 rows spanning 1970-2023. Its fields are sex, year, age group, age-standardized prevalence of hypertension (ASPH, represented as a proportion), and mean total cholesterol (MTC, mmol/L). The manuscript describes the supplied series as applying to men aged 30-79 years. The validated analysis file has SHA-256 `376ccd858ea6a759c08d4ee5fb8e68f1fd4b8b3e66334b586195014facf316cd`; the originally supplied CSV has SHA-256 `337cec7bc9e6e6029c8654c34bfc1cab84a76c815de6bf68fac6dc3df8aa4e55`.

## What remains unresolved

Local numerical matching found that ASPH values for 1990-2019 correspond essentially exactly to the identified China NCD-RisC series, and MTC values for 1980-2018 correspond exactly to the identified China NCD-RisC series. These overlapping matches are strong numerical clues, but they do not by themselves establish the complete dataset's geography, source chain, or processing history.

The supplied files do not establish a complete, independently verifiable chain from an original data provider to the processed study series. In particular, provenance outside the overlapping periods above could not be independently reconstructed. The original source URL and filename, download date, provider release/version, variable-selection procedure, transformations, interpolation or smoothing, uncertainty intervals, and whether recent values are observations, estimates, projections, or manually derived values remain unresolved. Numerical similarity to a candidate source is not proof of provenance for the nonoverlapping years.

## Why the data are absent

Neither the complete 1970-2023 study dataset nor any candidate third-party NCD-RisC files are redistributed here. Their redistribution rights and exact provenance have not been established. The hashes and schema above support local identity checks but do not grant access or a license.

The repository's MIT and CC BY 4.0 licenses do not apply to or grant rights in the complete study dataset, candidate NCD-RisC files, third-party downloaded data, or other externally owned material. No such data file is included in the repository.

## Reasonable-request route

Researchers may request the processed study dataset from the corresponding author, Fazli Subhan, at `fazlisubhanse@outlook.com`. Access is subject to author review, confirmation that sharing is legally and contractually permitted, and any restrictions imposed by the original data provider. A request may therefore be declined or redirected to the original provider. This repository does not imply that the authors own redistribution rights.

If an authorized copy is obtained, place it at `data/raw/input_series.csv` and run the validation command in `REPRODUCIBILITY.md`. A matching hash confirms file identity only; it does not resolve the provenance limitations above.
