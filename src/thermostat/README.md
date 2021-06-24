# Details on the Thermostat package

In this README, we describe our implementational details.
More to follow.

---

### Config files
jsonnet config files are used for generating the explanation datasets and have the following naming convention:
`<DATASET_ID>/<EXPLAINER_ID>/<MODEL_ID>.jsonnet` where
* `<DATASET_ID>` corresponds to a dataset (from `datasets` package by default, but can be any other locally stored dataset),
* `<EXPLAINER_ID>` corresponds to an explainability method (usually provided through the `captum` package) and
* `<MODEL_ID>` corresponds to a model (from `transformers` package by default)