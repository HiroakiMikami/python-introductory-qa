imports = ["base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
main = mlprogram.entrypoint.evaluate(
    workspace_dir=os.path.join(args=[base_output_dir, "workspace"]),
    input_dir=output_dir,
    output_dir=output_dir,
    valid_dataset=valid_dataset,
    model=model,
    synthesizer=synthesizer,
    metrics=metrics,
    top_n=params.metric_top_n,
    device=device,
    n_process=params.n_evaluate_process,
)
