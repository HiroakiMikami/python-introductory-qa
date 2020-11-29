imports = ["../dataset.py", "../io.py"]

params = {
    "word_threshold": 5,
    "token_threshold": 5,
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
    "batch_size": 32,
    "n_iteration":      10000,
    "eval_interval":     2000,
    "snapshot_interval": 1000,
    "beam_size": 50,
    "max_step_size": 100,
    "metric_top_n": [1, 5, 10],
    "metric_threshold": 1.0,
    "metric": "bleu@5",
    "n_evaluate_process": None,
    "train_timeout_sec": 30,
    "evaluate_timeout_sec": 120,
}

extract_reference = TokenizeQuery()
parser = mlprogram.languages.python.Parser(split_value=SplitValue())
encoder = {
    "word_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "word_encoder.pt"],
        ),
        config=torchnlp.encoders.LabelEncoder(
            sample=mlprogram.utils.data.get_words(
                dataset=dataset_for_encoder,
                extract_reference=extract_reference,
            ),
            min_occurrences=params.word_threshold,
        ),
    ),
    "action_sequence_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "action_sequence_encoder.pt"],
        ),
        config=mlprogram.encoders.ActionSequenceEncoder(
            samples=mlprogram.utils.data.get_samples(
                dataset=dataset_for_encoder,
                parser=parser,
            ),
            token_threshold=params.token_threshold,
        ),
    ),
}
action_sequence_reader = mlprogram.nn.nl2code.ActionSequenceReader(
    num_rules=encoder.action_sequence_encoder._rule_encoder.vocab_size,
    num_tokens=encoder.action_sequence_encoder._token_encoder.vocab_size,
    num_node_types=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
    node_type_embedding_size=params.node_type_embedding_size,
    embedding_size=params.embedding_size,
)
model = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "encoder",
                mlprogram.nn.nl2code.NLReader(
                    num_words=encoder.word_encoder.vocab_size,
                    embedding_dim=params.embedding_size,
                    hidden_size=params.hidden_size,
                    dropout=params.dropout,
                ),
            ],
            [
                "decoder",
                torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            ["action_sequence_reader", action_sequence_reader],
                            [
                                "decoder",
                                mlprogram.nn.nl2code.Decoder(
                                    query_size=params.hidden_size,
                                    input_size=add(
                                        x=mul(
                                            x=2,
                                            y=params.embedding_size,
                                        ),
                                        y=params.node_type_embedding_size,
                                    ),
                                    hidden_size=params.hidden_size,
                                    att_hidden_size=params.attr_hidden_size,
                                    dropout=params.dropout,
                                ),
                            ],
                            [
                                "predictor",
                                mlprogram.nn.nl2code.Predictor(
                                    reader=action_sequence_reader,
                                    embedding_size=params.embedding_size,
                                    query_size=params.hidden_size,
                                    hidden_size=params.hidden_size,
                                    att_hidden_size=params.attr_hidden_size,
                                ),
                            ],
                        ],
                    ),
                ),
            ],
        ],
    ),
)
collate = mlprogram.utils.data.Collate(
    device=device,
    word_nl_query=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    nl_query_features=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    reference_features=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    previous_actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    previous_action_rules=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    history=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=1,
        padding_value=0,
    ),
    hidden_state=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    state=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    ground_truth_actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
)
transform_input = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "extract_reference",
                mlprogram.utils.transform.text.ExtractReference(
                    extract_reference=extract_reference,
                ),
            ],
            [
                "encode_word_query",
                mlprogram.utils.transform.text.EncodeWordQuery(
                    word_encoder=encoder.word_encoder,
                ),
            ],
        ],
    ),
)
transform_action_sequence = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_previous_action",
                mlprogram.utils.transform.action_sequence.AddPreviousActions(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_action",
                mlprogram.utils.transform.action_sequence.AddActions(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_state",
                mlprogram.utils.transform.action_sequence.AddStateForRnnDecoder(),
            ],
            [
                "add_history",
                mlprogram.utils.transform.action_sequence.AddHistoryState(),
            ],
        ],
    ),
)
synthesizer = mlprogram.synthesizers.BeamSearch(
    beam_size=params.beam_size,
    max_step_size=params.max_step_size,
    sampler=mlprogram.samplers.transform(
        sampler=mlprogram.samplers.ActionSequenceSampler(
            encoder=encoder.action_sequence_encoder,
            is_subtype=is_subtype,
            transform_input=transform_input,
            transform_action_sequence=transform_action_sequence,
            collate=collate,
            module=model,
        ),
        transform=parser.unparse,
    ),
)
