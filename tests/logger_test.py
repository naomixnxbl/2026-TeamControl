from pathlib import Path

from TeamControl.utils.Logger import LogSaver


def test_log_saver_writes_messages_to_configured_directory(tmp_path):
    logs = LogSaver(log_dir=tmp_path, process_name="logger_test")

    logs.info("This is an info message.")
    logs.debug("This is a debug message.")
    logs.warning("This is a warning message.")
    logs.error("This is an error message.")
    logs.critical("This is a critical message.")

    content = Path(logs.log_file).read_text(encoding="utf-8")

    assert "START OF LOG FOR: logger_test" in content
    assert "This is an info message." in content
    assert "This is a critical message." in content
