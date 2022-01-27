import argparse
from argparse import Namespace

from transformers import AutoModelForCausalLM, AutoTokenizer


def define_arg_parser() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="A string, the model id of a pretrained model configuration hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like bert-base-uncased, or namespaced under a user or organization name, like dbmdz/bert-base-german-cased."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so revision can be any identifier allowed by git."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    return parser.parse_args()


def main(config: Namespace):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model_name_or_path,
            revision=config.revision,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, revision=config.revision)
        model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
    except:
        exit(1)


if __name__ == "__main__":
    config = define_arg_parser()
    main(config)
