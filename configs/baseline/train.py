imports = ["base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
transform = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            ["transform_input", transform_input],
            [
                "transform_code",
                mlprogram.utils.transform.action_sequence.GroundTruthToActionSequence(
                    parser=parser,
                ),
            ],
            ["transform_action_sequence", transform_action_sequence],
            [
                "transform_ground_truth",
                mlprogram.utils.transform.action_sequence.EncodeActionSequence(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                ),
            ],
        ],
    ),
)
collate_fn = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "transform",
                mlprogram.functools.Map(
                    func=transform,
                ),
            ],
            ["collate", collate.collate],
        ],
    ),
)
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
main = mlprogram.entrypoint.train_supervised(
    workspace_dir=os.path.join(args=[base_output_dir, "workspace"]),
    output_dir=output_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                ["loss", mlprogram.nn.action_sequence.Loss()],
                [
                    "pick",
                    mlprogram.nn.Function(
                        f=Pick(
                            key="output@action_sequence_loss",
                        ),
                    ),
                ],
            ],
        ),
    ),
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
            synthesizer=synthesizer,
            timeout_sec=params.train_timeout_sec,
        ),
        metrics=metrics,
        top_n=params.metric_top_n,
        n_process=params.n_evaluate_process,
    ),
    metric=params.metric,
    threshold=params.metric_threshold,
    collate=collate_fn,
    batch_size=params.batch_size,
    length=mlprogram.entrypoint.train.Iteration(
        n=params.n_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=params.eval_interval,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=params.snapshot_interval,
    ),
    device=device,
)
