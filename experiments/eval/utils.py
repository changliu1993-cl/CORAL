import yaml


def load_config(benchmark, dataset, config_path="config/benchmark_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if benchmark not in config:
        raise ValueError(f"Unknown benchmark '{benchmark}'")

    bench_conf = config[benchmark]
    if dataset not in bench_conf["datasets"]:
        raise ValueError(f"Dataset '{dataset}' is not supported for benchmark '{benchmark}'")

    question_path = bench_conf["question_path"].format(dataset=dataset)
    image_dir = bench_conf["image_dir"].format(dataset=dataset)

    return question_path, image_dir