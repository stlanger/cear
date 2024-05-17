# CEAR: Creating a knowledge graph of chemical entities and roles in scientific literature

> this file will be updated in the next days

The corresponding paper described the following steps:

## Chemical Entity and Chemical Role Recognition

A [Google Electra model](https://huggingface.co/google/electra-base-discriminator) is fine-tuned on different datasets for NER:

- The [BC5CDR](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) dataset consists of human annotations of chemicals, diseases and their
interactions from 1,500 PubMed articles
- The [NLM-Chem](https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/) contains 150 full-text articles on biomedical literature, carefully
selected for containing chemical entities which are difficult to find for NER tools. Ten
domain experts annotated the chemical entities in three annotation rounds.
- [CRAFT](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) contains 97 full-text open access articles from the PubMed Central
Open Access subset. It identifies all mentions of nearly all concepts from nine prominent
biomedical ontologies, including ChEBI

Both NLM-Chem and BC5CDR are not annotated with chemical roles (for example: solvent, catalyst, drug). We use all chemical roles in the ChEBI ontology to annotate them using a simple lexical approach.