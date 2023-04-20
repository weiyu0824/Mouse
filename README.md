# Mouse

## Overview

## Installation
1. Download the query.csv file 
```
http://api.brain-map.org/api/v2/data/query.csv?criteria=
model::Gene,
rma::criteria,products[abbreviation$eq'DevMouse'],
rma::options,[tabular$eq'genes.id','genes.acronym+as+gene_symbol','genes.name+as+gene_name','genes.entrez_id+as+entrez_gene_id','genes.homologene_id+as+homologene_group_id'],
[order$eq'genes.acronym']
&num_rows=all&start_row=0
```
2. Download the dataset:
    python3 download_dataset.py [-t threads] [-q query file path] [-d dataset dir]
## Usage
