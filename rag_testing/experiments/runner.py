import argparse
from typing import Dict, Any

import yaml

from rag_testing.pipelines import create_pipeline_from_config
from rag_testing.experiments.configs import get_experiment_config
from rag_testing.experiments.formatters import get_formatter
from rag_testing.experiments.data import (
    get_standard_test_queries
)
from rag_testing.experiments.experiment_runner import (
    run_retrieval_experiment
)

def run_experiment(
    config_type: str,
    type: str,
    **kwargs
) -> Dict[str, Any]:
    print(f"Running {config_type} experiment")
    
    config = get_experiment_config(config_type, **kwargs)
    pipeline = create_pipeline_from_config(config)
    test_queries = get_standard_test_queries()
    
    result_formatter = get_formatter(config_type)
    
    return run_retrieval_experiment(
        pipeline=pipeline,
        test_queries=test_queries,
        result_formatter=result_formatter,
        **kwargs
    )

def main():
   
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    
    parser.add_argument(
        "--config-path", 
        action="store",
        help="Path to the configuration file",
        required=True
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    for experiment in params["experiments"]:
        assert "type" in experiment, "Experiment must have a type"
        run_experiment(**experiment)


if __name__ == "__main__":
    main() 