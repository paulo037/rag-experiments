from typing import Dict
from rag_testing.config.models import RAGConfig
from rag_testing.core.base import RAGPipeline
from rag_testing.pipelines.pipeline import SimpleRAGPipeline
from rag_testing.pipelines.rada_pipeline import RadaPipeline


pipeline_from_type: Dict[str, type[RAGPipeline]] = {
    SimpleRAGPipeline.type: SimpleRAGPipeline,
    RadaPipeline.type: RadaPipeline
}

def create_pipeline_from_config(config: RAGConfig) -> RAGPipeline:

    assert config.type in pipeline_from_type, f"Pipeline type `{config.type}` not supported"
    pipe = pipeline_from_type[config.type]
    return pipe.build(config)