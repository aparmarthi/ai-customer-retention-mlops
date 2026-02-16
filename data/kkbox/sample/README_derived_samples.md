# Derived Samples (Spine + Model Table)
        These files are **small samples** of the pipeline-derived tables.
        ## Purpose
        - Provide a quick look at the engineered feature tables
        - Allow reviewers to inspect the modeling schema without downloading full datasets
        - Not intended for training real models

        ## Sampling strategy
        - Sampled **1000** users (`msno`) from `model_table.parquet`
        - Subset both `spine.parquet` and `model_table.parquet` to that same `msno` list

        ## Files
        - spine_sample.csv (<= 1000 users; 1 row per msno)
        - model_table_sample.csv (<= 1000 users; 1 row per msno)

        ## Generate
        Run from project root (with venv activated):
        ```bash
        python src/data/07_create_derived_tables.py
        