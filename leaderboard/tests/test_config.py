from aletheia_runner.config import dataset_task, dataset_model_lora, dataset_label


def test_task_parsing_and_model_lora():
    cases = {
        "aletheias-quest/dev-test-instructed-deception-Qwen3.5-27B-None":
            ("instructed-deception", "Qwen/Qwen3.5-27B", None),
        "aletheias-quest/dev-test-instructed-deception-gemma-3-27b-it-abliterated-collusion-gemma3-27b-v1":
            ("instructed-deception", "google/gemma-3-27b-it", "abliterated-collusion-gemma3-27b-v1"),
        "aletheias-quest/validation-soft-trigger-gemma-3-27b-it-gemma-3-27b-it-lora-greeting":
            ("soft-trigger", "google/gemma-3-27b-it", "gemma-3-27b-it-lora-greeting"),
        "aletheias-quest/validation-insider-trading-gemma-3-27b-it-None":
            ("insider-trading", "google/gemma-3-27b-it", None),
        "aletheias-quest/validation-soft-trigger-NVIDIA-Nemotron-3-Super-120B-A12B-BF16-lora-greeting-filtered-r64":
            ("soft-trigger", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", "lora-greeting-filtered-r64"),
    }
    for name, (task, model, lora) in cases.items():
        assert dataset_task(name) == task, name
        assert dataset_model_lora(name) == (model, lora), name


def test_codename_grouped_by_task():
    # Same task (different models/loras) -> one codename; different tasks -> different.
    soft = [
        "aletheias-quest/validation-soft-trigger-Qwen3.5-27B-qwen3.5-27b-lora-greeting",
        "aletheias-quest/validation-soft-trigger-gemma-3-27b-it-gemma-3-27b-it-lora-time",
    ]
    instructed = [
        "aletheias-quest/dev-test-instructed-deception-Qwen3.5-27B-None",
        "aletheias-quest/dev-test-instructed-deception-gemma-3-27b-it-None",
    ]
    assert len({dataset_label(n) for n in soft}) == 1
    assert len({dataset_label(n) for n in instructed}) == 1
    assert dataset_label(soft[0]) != dataset_label(instructed[0])


def test_unknown_name_falls_back():
    # A non-conforming id (e.g. a test fixture) still gets a stable codename.
    assert dataset_model_lora("dummy") == (None, None)
    assert dataset_label("dummy") == dataset_label("dummy")
    assert dataset_label("dummy").startswith("Dataset ")
