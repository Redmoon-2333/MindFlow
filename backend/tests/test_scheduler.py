from mindflow.collector.scheduler import CollectorScheduler


def test_scheduler_initial_state():
    scheduler = CollectorScheduler()
    assert scheduler.is_running is False


def test_scheduler_start_stop():
    scheduler = CollectorScheduler()
    scheduler.start()
    assert scheduler.is_running is True
    scheduler.stop()
    assert scheduler.is_running is False


def test_scheduler_double_start():
    scheduler = CollectorScheduler()
    scheduler.start()
    scheduler.start()
    assert scheduler.is_running is True
    scheduler.stop()


def test_scheduler_stop_when_not_running():
    scheduler = CollectorScheduler()
    scheduler.stop()
    assert scheduler.is_running is False
