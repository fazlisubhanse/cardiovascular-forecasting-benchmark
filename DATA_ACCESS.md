# Data access and local reconstruction

## Why the numerical data are absent

NCD-RisC source numerical files are **not redistributed in this release**. The official NCD-RisC download pages provide public access, but no explicit dataset-level redistribution licence was identified for the separately hosted CSV files. Extracted annual NCD-RisC estimates and uncertainty bounds are therefore also withheld.

The WHO hypertension CSV is an independent cross-check rather than the primary analysis source. It is omitted under this package's extra-conservative policy even though the WHO indicator page identifies the dataset as CC BY 4.0 subject to the WHO data terms.

Users should retrieve all three files directly from their official hosts, verify their hashes, and reconstruct the processed dataset locally.

## Required official files

### Primary ASPH source

- Organization: NCD Risk Factor Collaboration (NCD-RisC)
- Dataset: Lancet 2021 hypertension age-standardised country estimates
- Publication: *Worldwide trends in hypertension prevalence and progress in treatment and control from 1990 to 2019*
- Official download page: https://www.ncdrisc.org/data-downloads-hypertension.html
- Direct download: https://www.ncdrisc.org/downloads/hypertension/NCD-RisC_Lancet_2021_Hypertension_age_standardised_countries.csv
- Expected filename: `NCD-RisC_Lancet_2021_Hypertension_age_standardised_countries.csv`
- Expected SHA-256: `5b93b95db281acddce4b80feec4523ddc928cff6b236d93fa449566d070e5517`
- Filter: China (`ISO=CHN`), men, age standardised for ages 30–79 years
- Years: 1990–2019, 30 annual observations
- Unit: proportion
- Processing: central estimate and lower/upper 95% uncertainty bounds are retained by identity

### Primary MTC source

- Organization: NCD Risk Factor Collaboration (NCD-RisC)
- Dataset: Nature 2020 cholesterol age-standardised country estimates
- Publication: *Repositioning of the global epicentre of non-optimal cholesterol*
- Official download page: https://www.ncdrisc.org/data-downloads-cholesterol.html
- Direct download: https://www.ncdrisc.org/downloads/chol/NCD_RisC_Nature_2020_Cholesterol_age_standardised_countries.csv
- Expected filename: `NCD_RisC_Nature_2020_Cholesterol_age_standardised_countries.csv`
- Expected SHA-256: `e24ac13cebf9e80ecc3bd42c8fec0a8e5eb63da34e464d1fb3cc64f5cb18b05b`
- Filter: China (`ISO=CHN`), men, adults aged 18 years and older
- Years: 1980–2018, 39 annual observations
- Unit: mmol/L
- Processing: mean total cholesterol and its lower/upper 95% uncertainty bounds are retained by identity; HDL and non-HDL are not selected targets

### Independent ASPH cross-check

- Organization: World Health Organization (WHO)
- Indicator: 608DE39 / NCD_HYP_PREVALENCE_A
- Indicator page and terms: https://data.who.int/indicators/i/7DA4E68/608DE39
- Direct download: https://srhdpeuwpubsa.blob.core.windows.net/whdh/DATADOT/INDICATOR/608DE39_ALL_LATEST.csv
- Expected filename: `608DE39_ALL_LATEST.csv`
- Expected SHA-256: `767d6fe20ad8116c2ca856dccb504ca3c55668d2316bbdb1082a7d4b9d10bffd`
- Filter: China, male, ages 30–79 years, 1990–2019
- Unit: percent, reported to one decimal percentage point
- Role: rounding cross-check only; it does not replace the full-precision primary ASPH source

## Expected local paths

Place the downloaded files without renaming them at:

```text
data/raw/NCD-RisC_Lancet_2021_Hypertension_age_standardised_countries.csv
data/raw/NCD_RisC_Nature_2020_Cholesterol_age_standardised_countries.csv
data/raw/608DE39_ALL_LATEST.csv
```

## Reconstruction

From the repository root, run:

```shell
python scripts/reconstruct_round2_official_data.py
```

The script verifies all three raw hashes before reading them, applies the documented filters, performs the WHO rounding cross-check, validates the 30/39 annual counts and units, and writes:

```text
data/processed/china_male_official_cardiovascular_series.csv
```

The expected reconstructed dataset contains 69 rows and has SHA-256:

```text
4575b29feee41e3832c9c2e51c9c0f8f0e312d5e7f4df9b87ce75e13e705c3e2
```

The hash is identity metadata and does not disclose the annual values. No interpolation, extrapolation, imputation, smoothing, splicing, or synthetic extension is performed.

## Licensing boundary

The MIT License in this package applies to original project software only. It does not apply to or relicense either NCD-RisC dataset or the WHO dataset. Users are responsible for reading and complying with the terms shown by each official source. Public availability should not be interpreted as a grant of redistribution rights.
