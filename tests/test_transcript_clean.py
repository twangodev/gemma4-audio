from gemma4_audio.backends.vllm import clean_transcript


def test_passes_through_normal_text():
    t = "Little Pearl, who was as greatly pleased."
    assert clean_transcript(t) == t


def test_strips_full_channel_loop_to_empty():
    loop = "<|channel>thought\n<channel|>" * 5
    assert clean_transcript(loop) == ""


def test_strips_leading_channel_block_keeps_answer():
    assert clean_transcript("<|channel>thought\n<channel|>The answer.") == "The answer."


def test_strips_unclosed_trailing_channel():
    assert clean_transcript("The answer.<|channel>thought\nrambling") == "The answer."


def test_keeps_the_word_thought_in_real_text():
    t = "I thought it was fine."
    assert clean_transcript(t) == t
