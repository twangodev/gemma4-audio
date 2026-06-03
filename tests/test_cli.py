from gemma4_audio.cli import parse_args


def test_batch_size_flag():
    cfg = parse_args(["eval", "--model", "m", "--batch-size", "32"])
    assert cfg.batch_size == 32


def test_batch_size_defaults_to_16():
    cfg = parse_args(["eval", "--model", "m"])
    assert cfg.batch_size == 16


def test_output_dir_flag_is_wired_through():
    cfg = parse_args(["eval", "--model", "m", "--output-dir", "somewhere"])
    assert cfg.output_dir == "somewhere"
