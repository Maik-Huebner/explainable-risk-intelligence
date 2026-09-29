# Data Card

## Source and licence

This project uses I-Cheng Yeh's *Default of Credit Card Clients* dataset from the UCI Machine Learning
Repository (dataset 350), DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H). UCI lists the dataset
under Creative Commons Attribution 4.0. The original project code is separately licensed under MIT.

The checked source archive URL and SHA-256 are versioned in the packaged
`explainable_risk_intelligence/resources/default.yaml`. The downloader writes a local
`provenance.json` containing the retrieval time, source, size, and verified digest.

## Population and target

The dataset contains 30,000 Taiwanese credit-card clients, with credit limit, demographics, repayment
status, bill amounts, and payment amounts covering April through September 2005. The binary target records
default payment in the following month. UCI reports no missing values in the source.

## Modelling fields

The model uses credit limit, repayment statuses, bill amounts, payment amounts, and deterministic aggregate
features. `ID` is excluded because it is an identifier. Sex, age, education, and marital status are excluded
from model inputs and retained only for audit slices.

This choice reduces direct demographic dependence but does not eliminate proxies, structural bias, or
disparate impact. It must not be interpreted as a fairness guarantee.

## Known limitations

- Historical observations from one market are not representative of current applicants or other regions.
- No suitable event timestamp supports a defensible temporal split.
- Demographic fields are binary/coarse and reflect the source's dated schema.
- The observed target is not a causal or exhaustive measure of creditworthiness.
- Dataset documentation does not establish how every operational field was collected or governed.

## Citation

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H
