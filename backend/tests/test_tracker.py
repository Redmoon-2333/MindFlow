from mindflow.collector.tracker import get_active_window_info, is_user_idle


def test_get_active_window_info_returns_dict_or_none():
    result = get_active_window_info()
    if result is not None:
        assert isinstance(result, dict)
        assert "process_name" in result
        assert "window_title" in result
        assert "timestamp" in result


def test_is_user_idle_returns_bool():
    result = is_user_idle(60)
    assert isinstance(result, bool)


def test_is_user_idle_with_custom_threshold():
    result = is_user_idle(1)
    assert isinstance(result, bool)
