"""
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in vidgen.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use vidgen as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import os

from vidgen.config import get_cfg
from vidgen.engine import default_argument_parser, default_setup, launch, Trainer, comm
from vidgen.evaluation import verify_results, DatasetEvaluators, VTSampler
from vidgen.evaluation.bits_evaluation import BitsEvaluator
from vidgen.evaluation.codes_extractor import CodesExtractor
from vidgen.evaluation.mse_evaluation import MSEEvaluator


class MyTrainer(Trainer):
    """
    We use the "Trainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        if "CodesExtractor" in cfg.TEST.EVALUATORS:
            evaluator_list.append(CodesExtractor(dataset_name, distributed=True, output_dir=output_folder))
        if "MSEEvaluator" in cfg.TEST.EVALUATORS:
            evaluator_list.append(MSEEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if "BitsEvaluator" in cfg.TEST.EVALUATORS:
            evaluator_list.append(BitsEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        if "VTSampler" in cfg.TEST.EVALUATORS:
            evaluator_list.append(VTSampler(cfg, dataset_name, distributed=True, output_dir=output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        _, checkpointers = model.configure_optimizers_and_checkpointers()
        for item in checkpointers:
            item["checkpointer"].resume_or_load(item["pretrained"], resume=False)
        res = MyTrainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
