from types import SimpleNamespace

from dev.schema_llm.explore import head_fv_phi_cache


def test_stable_phi_rotary_pins_and_restores_without_generation_source_check(monkeypatch):
    original = object()
    previous_inv = object()
    previous_long = object()
    rotary = SimpleNamespace(
        original_inv_freq=original,
        inv_freq=previous_inv,
        long_inv_freq=previous_long,
    )
    model = object()
    monkeypatch.setattr(
        head_fv_phi_cache, "_checked_rotary", lambda _lm: (model, rotary)
    )

    with head_fv_phi_cache.stable_phi_rotary(object()):
        assert rotary.inv_freq is original
        assert rotary.long_inv_freq is original

    assert rotary.inv_freq is previous_inv
    assert rotary.long_inv_freq is previous_long


def test_stable_phi_rotary_restores_absent_long_frequency(monkeypatch):
    original = object()
    previous_inv = object()
    rotary = SimpleNamespace(original_inv_freq=original, inv_freq=previous_inv)
    monkeypatch.setattr(
        head_fv_phi_cache, "_checked_rotary", lambda _lm: (object(), rotary)
    )

    with head_fv_phi_cache.stable_phi_rotary(object()):
        assert rotary.inv_freq is original
        assert rotary.long_inv_freq is original

    assert rotary.inv_freq is previous_inv
    assert not hasattr(rotary, "long_inv_freq")
