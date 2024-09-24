from graphrag.index.cli import index_cli
from graphrag.index.emit.types import TableEmitterType
from graphrag.index.progress.types import ReporterType
index_cli(
    root_dir="ragdirs/ragdir_6",
    verbose=True,
    resume="",
    update_index_id=None,
    memprofile=False,
    nocache=False,
    reporter=ReporterType.RICH,
    config_filepath=None,
    emit=[TableEmitterType.Parquet.value],
    dryrun=False,
    init=False,
    skip_validations=False,
    output_dir=None,
    local=True
)


# from graphrag.query.cli import run_local_search

# run_local_search(
#     config_filepath=None,
#     data_dir="output",
#     root_dir="ragdirs/ragdir_4",
#     community_level=2,
#     response_type="Multiple Paragraphs",
#     streaming=False,
#     query="What is Graphrag?",  # Replace with the actual query string
# )