# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import HfFolder, Repository, whoami

from .pipeline_utils import DiffusionPipeline
from .utils import is_modelcards_available, logging


if is_modelcards_available():
    from modelcards import CardData, ModelCard


logger = logging.get_logger(__name__)


MODEL_CARD_TEMPLATE_PATH = Path(__file__).parent / "utils" / "model_card_template.md"


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def init_git_repo(args, at_init: bool = False):
    """
    Args:
    Initializes a git repo in `args.hub_model_id`.
        at_init (`bool`, *optional*, defaults to `False`):
            Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is `True`
            and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped out.
    """
    if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
        return
    hub_token = args.hub_token if hasattr(args, "hub_token") else None
    use_auth_token = True if hub_token is None else hub_token
    if not hasattr(args, "hub_model_id") or args.hub_model_id is None:
        repo_name = Path(args.output_dir).absolute().name
    else:
        repo_name = args.hub_model_id
    if "/" not in repo_name:
        repo_name = get_full_repo_name(repo_name, token=hub_token)

    try:
        repo = Repository(
            args.output_dir,
            clone_from=repo_name,
            use_auth_token=use_auth_token,
            private=args.hub_private_repo,
        )
    except EnvironmentError:
        if args.overwrite_output_dir and at_init:
            # Try again after wiping output_dir
            shutil.rmtree(args.output_dir)
            repo = Repository(
                args.output_dir,
                clone_from=repo_name,
                use_auth_token=use_auth_token,
            )
        else:
            raise

    repo.git_pull()

    # By default, ignore the checkpoint folders
    if not os.path.exists(os.path.join(args.output_dir, ".gitignore")):
        with open(os.path.join(args.output_dir, ".gitignore"), "w", encoding="utf-8") as writer:
            writer.writelines(["checkpoint-*/"])

    return repo


def push_to_hub(
    args,
    pipeline: DiffusionPipeline,
    repo: Repository,
    commit_message: Optional[str] = "End of training",
    blocking: bool = True,
    **kwargs,
) -> str:
    """
    Parameters:
    Upload *self.model* and *self.tokenizer* to the 🤗 model hub on the repo *self.args.hub_model_id*.
        commit_message (`str`, *optional*, defaults to `"End of training"`):
            Message to commit while pushing.
        blocking (`bool`, *optional*, defaults to `True`):
            Whether the function should return only when the `git push` has finished.
        kwargs:
            Additional keyword arguments passed along to [`create_model_card`].
    Returns:
        The url of the commit of your model in the given repository if `blocking=False`, a tuple with the url of the
        commit and an object to track the progress of the commit if `blocking=True`
    """

    if not hasattr(args, "hub_model_id") or args.hub_model_id is None:
        model_name = Path(args.output_dir).name
    else:
        model_name = args.hub_model_id.split("/")[-1]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving pipeline checkpoint to {output_dir}")
    pipeline.save_pretrained(output_dir)

    # Only push from one node.
    if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
        return

    # Cancel any async push in progress if blocking=True. The commits will all be pushed together.
    if (
        blocking
        and len(repo.command_queue) > 0
        and repo.command_queue[-1] is not None
        and not repo.command_queue[-1].is_done
    ):
        repo.command_queue[-1]._process.kill()

    git_head_commit_url = repo.push_to_hub(commit_message=commit_message, blocking=blocking, auto_lfs_prune=True)
    # push separately the model card to be independent from the rest of the model
    create_model_card(args, model_name=model_name)
    try:
        repo.push_to_hub(commit_message="update model card README.md", blocking=blocking, auto_lfs_prune=True)
    except EnvironmentError as exc:
        logger.error(f"Error pushing update to the model card. Please read logs and retry.\n${exc}")

    return git_head_commit_url


def create_model_card(args, model_name):
    if not is_modelcards_available:
        raise ValueError(
            "Please make sure to have `modelcards` installed when using the `create_model_card` function. You can"
            " install the package with `pip install modelcards`."
        )

    if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
        return

    hub_token = args.hub_token if hasattr(args, "hub_token") else None
    repo_name = get_full_repo_name(model_name, token=hub_token)

    model_card = ModelCard.from_template(
        card_data=CardData(  # Card metadata object that will be converted to YAML block
            language="en",
            license="apache-2.0",
            library_name="diffusers",
            tags=[],
            datasets=args.dataset_name,
            metrics=[],
        ),
        template_path=MODEL_CARD_TEMPLATE_PATH,
        model_name=model_name,
        repo_name=repo_name,
        dataset_name=args.dataset_name if hasattr(args, "dataset_name") else None,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
        if hasattr(args, "gradient_accumulation_steps")
        else None,
        adam_beta1=args.adam_beta1 if hasattr(args, "adam_beta1") else None,
        adam_beta2=args.adam_beta2 if hasattr(args, "adam_beta2") else None,
        adam_weight_decay=args.adam_weight_decay if hasattr(args, "adam_weight_decay") else None,
        adam_epsilon=args.adam_epsilon if hasattr(args, "adam_epsilon") else None,
        lr_scheduler=args.lr_scheduler if hasattr(args, "lr_scheduler") else None,
        lr_warmup_steps=args.lr_warmup_steps if hasattr(args, "lr_warmup_steps") else None,
        ema_inv_gamma=args.ema_inv_gamma if hasattr(args, "ema_inv_gamma") else None,
        ema_power=args.ema_power if hasattr(args, "ema_power") else None,
        ema_max_decay=args.ema_max_decay if hasattr(args, "ema_max_decay") else None,
        mixed_precision=args.mixed_precision,
    )

    card_path = os.path.join(args.output_dir, "README.md")
    model_card.save(card_path)
