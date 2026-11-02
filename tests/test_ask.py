from types import SimpleNamespace

from app.ask import ask


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="answer",
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.ask.genai.Client", FakeClient)

    assert ask("question") == "answer"
    assert capsys.readouterr().out == "[トークン数] 入力: 12, 出力: 34\n"


def test_ask_reports_missing_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="answer")

    class FakeModels:
        def generate_content(self, **kwargs):
            return response

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    monkeypatch.setattr("app.ask.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.ask.genai.Client", FakeClient)

    ask("question")
    assert capsys.readouterr().out == "[トークン数] 入力: 不明, 出力: 不明\n"
