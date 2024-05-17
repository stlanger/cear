# CEAR: Creating a knowledge graph of chemical entities and roles in scientific literature

> this file will be updated in the next days

## How it works
The corresponding paper described the following steps, that are used to create the knowledge graph (KG):

### 1. Text Extraction
We have downloaded 8,000 chemistry research papers from [ChemRxiv](https://chemrxiv.org/). The text extraction process is conducted in a separate NestJS project, which uses `pdf2txt` and creates a JSON file which includes:
- the papers' metadata (downloaded from ChemRxiv)
- the papers' full text
- de-duplication information
- the PDF file's pages with the full text

### 2. Chemical Entity and Chemical Role Recognition
The `ner-chem-trainer notebook` fine-tunes a [Google Electra model](https://huggingface.co/google/electra-base-discriminator) on different datasets for NER:

- The [BC5CDR](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) dataset consists of human annotations of chemicals, diseases and their
interactions from 1,500 PubMed articles
- The [NLM-Chem](https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/) contains 150 full-text articles on biomedical literature, carefully
selected for containing chemical entities which are difficult to find for NER tools. Ten
domain experts annotated the chemical entities in three annotation rounds.
- [CRAFT](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) contains 97 full-text open access articles from the PubMed Central
Open Access subset. It identifies all mentions of nearly all concepts from nine prominent
biomedical ontologies, including ChEBI

Both NLM-Chem and BC5CDR lack annotations for chemical roles (for example: solvent, catalyst, drug). We annotate them using a lexical approach for all chemical roles defined in the [ChEBI ontology](https://www.ebi.ac.uk/chebi/). This is accomplished in the corresponding loader classes BC5CDRLoader and NLMChemLoader.

### 3. Link Validation
The `llama2-role-validator` notebook uses a [Llama-2-7b-chat-hf] to check for all co-occurences of chemical entities and roles in a sentence, whether the chemical entity has the mentioned chemical role.

### 4. Knowledge Graph Creation
The `kg_data_construction` notebook links all confirmed pairs of chemical entities and roles to ChEBI. After grouping and counting these pairs, a hyperparameter `minRef` is applied to filter out any pairs of entities and relations based on how frequent they are found inside the literature set. The knowledge graph consists of the described relations. It is stored using the Terse RDF Triple Language (Turtle). Each contained chemical entity (obo:CHEBI_24431) and role (obo:CHEBI_50906) is defined by its ChEBI identifier. Chemical entities or roles that are unknown to ChEBI are defined using the `@prefix cear:<https://wwwiti.cs.uni-magdeburg.de/iti_dke/cear/> .` namespace. The `obo:RO_0000087` is used in ChEBI to define roles of chemical entities. The following listing shows an example for two chemical entities, `ethylene glycol bis(2-aminoethyl)tetraacetate` and `PBS`, both of which have the chemical role `buffer`:

```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix obo: <http://purl.obolibrary.org/obo/> .
@prefix cear: <https://wwwiti.cs.uni-magdeburg.de/iti_dke/cear/> .
obo:CHEBI_35225 rdf:type obo:CHEBI_50906 .
obo:CHEBI_35225 rdfs:label "buffer" .
obo:CHEBI_30741 rdf:type obo:CHEBI_24431 .
obo:CHEBI_30741 rdfs:label "ethylene glycol bis(2-aminoethyl)tetraacetate" .
obo:CHEBI_30741 obo:RO_0000087 obo:CHEBI_35225 .
cear:chem_4023 rdf:type obo:CHEBI_24431 .
cear:chem_4023 rdfs:label "PBS" .
cear:chem_4023 obo:RO_0000087 obo:CHEBI_35225 .
```

Depending on the minRef hyperparameter specification, different KGs are created. They can all be accessed at [https://wwwiti.cs.uni-magdeburg.de/iti_dke/cear/](https://wwwiti.cs.uni-magdeburg.de/iti_dke/cear/)


Additionally we create a `nodes.json` and an `edges` JSON file, which contain chemical entities and roles as nodes and the :hasRole relationship between them as edges. These files are used in a separate VueJS project using v-network-graph to visualize the KG.


## Setup
This section will contain detailed information of how to setup this project
