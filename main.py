import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2 import service_account
from openai import OpenAI

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


BASE_URL = "https://education-demo-app.services.ai.azure.com/openai/v1"
MODEL = "gpt-5.4-mini"
WORDS_PATH = Path(__file__).resolve().parent.parent / "terminal" / "words.json"


def get_client() -> OpenAI | None:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url=BASE_URL, api_key=api_key)


def ensure_words_file() -> None:
    if not WORDS_PATH.exists() or not WORDS_PATH.read_text(encoding="utf-8").strip():
        WORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        WORDS_PATH.write_text('{"words": []}', encoding="utf-8")


def load_words_db() -> dict[str, Any]:
    ensure_words_file()
    try:
        data = json.loads(WORDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {"words": []}
    if not isinstance(data, dict) or "words" not in data or not isinstance(data["words"], list):
        data = {"words": []}
    return data


def save_words_db(db: dict[str, Any]) -> None:
    WORDS_PATH.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def build_words_updates(selected_word: str | None = None):
    data = load_words_db()
    choices = [item.get("word", "") for item in data["words"] if item.get("word")]
    current = selected_word if selected_word in choices else (choices[0] if choices else None)
    rows = [[item.get("word", ""), len(item.get("entries", [])), item.get("updated_at", "")] for item in data["words"]]
    return (
        rows,
        gr.update(value=current, choices=choices),
        gr.update(value=current, choices=choices),
        show_word_detail(current) if current else "単語を選択してください。",
    )


def extract_json(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.replace("```json", "").replace("```", "").strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON形式で応答が得られませんでした。")
    return json.loads(payload[start : end + 1])


def call_llm_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    client = get_client()
    if client is None:
        raise RuntimeError("環境変数 AZURE_OPENAI_API_KEY が設定されていません。")

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = res.choices[0].message.content or ""
    return extract_json(content)


def format_dictionary_result(result: dict[str, Any]) -> str:
    lines = [f"## {result.get('word', '')}"]
    entries = result.get("entries", [])
    for i, entry in enumerate(entries, start=1):
        pron = entry.get("pronunciation", {})
        lines.extend(
            [
                f"### その{i}",
                f"- 品詞: {entry.get('pos', '')}",
                f"- 意味: {entry.get('meaning', '')}",
                f"- 発音記号: {pron.get('us', '')}（米） / {pron.get('uk', '')}（英）",
                f"- アクセント: {entry.get('accent', '')}",
                f"- 例文: {entry.get('example', '')}",
                f"- 例文訳: {entry.get('translation', '')}",
            ]
        )
    return "\n".join(lines)


def format_dictionary_summary(result: dict[str, Any]) -> str:
    word = result.get("word", "")
    entries = result.get("entries", [])
    if not entries:
        return f"## {word}\n\n紹介情報を取得しました。"

    first_entry = entries[0]
    pron = first_entry.get("pronunciation", {})
    summary_lines = [
        f"## {word}",
        f"- 品詞: {first_entry.get('pos', '')}",
        f"- 意味: {first_entry.get('meaning', '')}",
        f"- 発音記号: {pron.get('us', '')}（米） / {pron.get('uk', '')}（英）",
    ]
    if len(entries) > 1:
        summary_lines.append(f"- 語義数: {len(entries)}")
    return "\n".join(summary_lines)


def build_tts_client():
    if texttospeech is None:
        raise RuntimeError("google-cloud-texttospeech がインストールされていません。")

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        credential_file = Path(credentials_path).expanduser()
        if credential_file.exists():
            credentials = service_account.Credentials.from_service_account_file(str(credential_file))
            return texttospeech.TextToSpeechClient(credentials=credentials)
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS に指定されたファイルが見つかりません: {credential_file}"
        )

    try:
        return texttospeech.TextToSpeechClient()
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google Cloud TTS の認証情報が見つかりません。"
            " サービスアカウントJSONを用意して GOOGLE_APPLICATION_CREDENTIALS にパスを設定してください。"
        ) from exc


def synthesize_fallback_tts(text: str):
    if gTTS is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            gTTS(text=text, lang="en").write_to_fp(tmp)
            return tmp.name
    except Exception:
        return None


def lookup_word(term: str):
    if not term.strip():
        return "単語（または熟語）を入力してください。", None, None, ""
    system_prompt = (
        "あなたは英語辞書アシスタントです。必ずJSONのみで返答してください。"
        "スキーマ: {word: string, entries: [{pos, meaning, pronunciation:{us,uk}, accent, example, translation}]}"
    )
    user_prompt = f"""
単語または熟語: {term}
複数の意味があれば複数entriesで返してください。
例文は学習者向けに短く自然なものにしてください。
"""
    try:
        result = call_llm_json(system_prompt, user_prompt)
        audio_path = synthesize_pronunciation(result.get("word", term))
        return format_dictionary_summary(result), result, audio_path, format_dictionary_result(result)
    except Exception as exc:
        return f"検索に失敗しました: {exc}", None, None, ""


def synthesize_pronunciation(text: str):
    if not text:
        return None

    try:
        tts_client = build_tts_client()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(response.audio_content)
            return tmp.name
    except Exception:
        return synthesize_fallback_tts(text)


def upsert_word(result: dict[str, Any] | None):
    if not result:
        return "先に単語検索を実行してください。", *build_words_updates()

    db = load_words_db()
    words = db["words"]
    target = (result.get("word") or "").strip()
    if not target:
        return "検索結果に単語情報がありません。", *build_words_updates()

    now = datetime.now().isoformat(timespec="seconds")
    new_record = {
        "word": target,
        "entries": result.get("entries", []),
        "updated_at": now,
        "created_at": now,
    }

    for idx, item in enumerate(words):
        if item.get("word", "").lower() == target.lower():
            new_record["created_at"] = item.get("created_at", now)
            words[idx] = new_record
            save_words_db(db)
            return f"単語帳を更新しました: {target}", *build_words_updates(target)

    words.append(new_record)
    save_words_db(db)
    return f"単語帳に追加しました: {target}", *build_words_updates(target)


def list_words_table():
    db = load_words_db()
    rows = []
    choices = []
    for item in db["words"]:
        word = item.get("word", "")
        choices.append(word)
        rows.append([word, len(item.get("entries", [])), item.get("updated_at", "")])
    selected = choices[0] if choices else None
    return rows, gr.update(choices=choices, value=selected), gr.update(choices=choices, value=selected)


def show_word_detail(word: str):
    db = load_words_db()
    for item in db["words"]:
        if item.get("word") == word:
            return format_dictionary_result(item)
    return "単語を選択してください。"


def delete_word(word: str):
    if not word:
        return "削除する単語を選択してください。", *list_words_table()
    db = load_words_db()
    before = len(db["words"])
    db["words"] = [item for item in db["words"] if item.get("word") != word]
    save_words_db(db)
    after = len(db["words"])
    status = f"削除しました: {word}" if after < before else "対象が見つかりませんでした。"
    rows, detail_choices, listen_choices = list_words_table()
    return status, rows, detail_choices, listen_choices


def synthesize_tts(word: str, speak_target: str):
    if not word:
        return None, "単語を選択してください。"

    db = load_words_db()
    target = next((item for item in db["words"] if item.get("word") == word), None)
    if target is None:
        return None, "単語が単語帳に見つかりません。"

    entries = target.get("entries", [])
    if speak_target == "例文" and entries:
        text = entries[0].get("example") or target.get("word")
    else:
        text = target.get("word")

    try:
        tts_client = build_tts_client()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(response.audio_content)
            return tmp.name, f"Google TTS で再生しました: {text}"
    except Exception as exc:
        fallback_audio = synthesize_fallback_tts(text)
        if fallback_audio:
            return fallback_audio, f"Google TTS の認証に失敗したため、代替音声を再生しました: {text}"
        return None, f"Google TTS 生成に失敗しました: {exc}"


def translate_text(direction: str, text: str):
    if not text.strip():
        return "翻訳する文章を入力してください。"

    system_prompt = (
        "あなたは英語学習向け翻訳アシスタントです。必ずJSONのみで返答してください。"
        "スキーマ: {translation: string, comment: string}"
    )
    if direction == "英→日":
        user_prompt = f"以下を自然な日本語に訳してください。\n{text}"
    else:
        user_prompt = f"以下を自然な英語に訳してください。\n{text}"

    try:
        result = call_llm_json(system_prompt, user_prompt)
        return f"## 翻訳\n{result.get('translation', '')}\n\n## コメント\n{result.get('comment', '')}"
    except Exception as exc:
        return f"翻訳に失敗しました: {exc}"


def pick_word_for_test() -> str:
    db = load_words_db()
    words = [item.get("word", "") for item in db["words"] if item.get("word")]
    return random.choice(words) if words else "study"


def create_writing_prompt():
    keyword = pick_word_for_test()
    system_prompt = (
        "あなたは英作文テスト作成者です。必ずJSONのみで返答してください。"
        "スキーマ: {prompt_jp: string, keyword: string, target_level: string}"
    )
    user_prompt = f"単語 '{keyword}' を使う英作文課題を1つ作ってください。"
    try:
        result = call_llm_json(system_prompt, user_prompt)
        state = {
            "prompt_jp": result.get("prompt_jp", ""),
            "keyword": result.get("keyword", keyword),
            "target_level": result.get("target_level", "高校生"),
        }
        view = (
            f"## 課題\n{state['prompt_jp']}\n\n"
            f"- 指定語: {state['keyword']}\n"
            f"- 目安レベル: {state['target_level']}"
        )
        return view, state
    except Exception as exc:
        return f"課題生成に失敗しました: {exc}", None


def grade_writing(answer: str, state: dict[str, Any] | None):
    if state is None:
        return "先に課題を生成してください。"
    if not answer.strip():
        return "答案を入力してください。"

    system_prompt = (
        "あなたは英作文採点者です。文法・語法を厳密に判定してください。"
        "必ずJSONのみで返答し、スキーマは"
        "{is_correct: boolean, score: number, corrected_answer: string, comment: string}"
    )
    user_prompt = f"""
課題: {state.get('prompt_jp', '')}
指定語: {state.get('keyword', '')}
受験者回答: {answer}

判定方針:
- 文法・語法が不自然なら is_correct=false
- score は 0-100
"""
    try:
        result = call_llm_json(system_prompt, user_prompt)
        if result.get("is_correct"):
            return f"## 判定: 正解\n- Score: {result.get('score', 0)}\n\n{result.get('comment', '')}"
        return (
            f"## 判定: 要修正\n- Score: {result.get('score', 0)}\n\n"
            f"### 訂正文\n{result.get('corrected_answer', '')}\n\n"
            f"### コメント\n{result.get('comment', '')}"
        )
    except Exception as exc:
        return f"採点に失敗しました: {exc}"


def create_translation_test(direction: str):
    db = load_words_db()
    vocab = [item.get("word", "") for item in db["words"] if item.get("word")]
    sample_words = random.sample(vocab, k=min(2, len(vocab))) if vocab else ["challenge"]

    system_prompt = (
        "あなたは翻訳テスト作成者です。必ずJSONのみで返答してください。"
        "スキーマ: {direction: string, source_sentence: string, expected_translation: string, used_words: [string]}"
    )
    mode = "en_to_ja" if direction == "英→日" else "ja_to_en"
    user_prompt = (
        f"方向: {mode}\n"
        f"必須語彙: {', '.join(sample_words)}\n"
        "1文だけ作成し、必須語彙をsource_sentenceに含めてください。"
    )
    try:
        result = call_llm_json(system_prompt, user_prompt)
        state = {
            "direction": result.get("direction", mode),
            "source_sentence": result.get("source_sentence", ""),
            "expected_translation": result.get("expected_translation", ""),
            "used_words": result.get("used_words", sample_words),
        }
        return (
            f"## 問題 ({'英→日' if state['direction'] == 'en_to_ja' else '日→英'})\n"
            f"{state['source_sentence']}\n\n"
            f"使用語彙: {', '.join(state['used_words'])}",
            state,
        )
    except Exception as exc:
        return f"問題生成に失敗しました: {exc}", None


def check_translation_test(answer: str, state: dict[str, Any] | None):
    if state is None:
        return "先に問題を生成してください。"
    if not answer.strip():
        return "解答を入力してください。"

    system_prompt = (
        "あなたは翻訳答案の採点者です。必ずJSONのみで返答してください。"
        "スキーマ: {is_correct: boolean, corrected_translation: string, comment: string}"
    )
    user_prompt = f"""
方向: {state.get('direction')}
問題文: {state.get('source_sentence')}
想定訳: {state.get('expected_translation')}
受験者解答: {answer}

判定ルール:
- 意味が十分一致し、文法が自然なら is_correct=true
- 不適切なら corrected_translation を提示
"""
    try:
        result = call_llm_json(system_prompt, user_prompt)
        if result.get("is_correct"):
            return f"## 判定: 正解\n{result.get('comment', '')}"
        return (
            "## 判定: 要修正\n"
            f"### 訂正案\n{result.get('corrected_translation', '')}\n\n"
            f"### コメント\n{result.get('comment', '')}"
        )
    except Exception as exc:
        return f"採点に失敗しました: {exc}"


with gr.Blocks(title="英語勉強補助アプリ") as app:
    gr.Markdown("# 英語勉強補助アプリ\n単語検索・単語帳・リスニング・翻訳・テストを1つに統合しています。")

    dictionary_state = gr.State(None)
    writing_state = gr.State(None)
    translation_test_state = gr.State(None)

    with gr.Tab("辞書/熟語検索"):
        term_input = gr.Textbox(label="単語または熟語")
        search_btn = gr.Button("検索")
        search_result = gr.Markdown()
        search_audio = gr.Audio(label="読み上げ", type="filepath")
        with gr.Accordion("詳細な紹介を表示", open=False):
            search_detail = gr.Markdown()
        add_btn = gr.Button("単語帳に保存")
        add_status = gr.Textbox(label="保存結果", interactive=False)

    with gr.Tab("単語帳"):
        reload_btn = gr.Button("単語帳を再読み込み")
        word_table = gr.Dataframe(
            headers=["word", "entry_count", "updated_at"],
            datatype=["str", "number", "str"],
            label="登録単語一覧",
            interactive=False,
        )
        word_select = gr.Dropdown(label="詳細表示する単語", choices=[])
        word_detail = gr.Markdown()
        delete_btn = gr.Button("選択単語を削除")
        delete_status = gr.Textbox(label="削除結果", interactive=False)

    with gr.Tab("リスニング (Google TTS)"):
        listen_reload_btn = gr.Button("単語一覧を更新")
        listen_word = gr.Dropdown(label="再生単語", choices=[])
        speak_target = gr.Radio(["単語", "例文"], value="単語", label="再生対象")
        tts_btn = gr.Button("音声を生成")
        tts_audio = gr.Audio(label="再生", type="filepath")
        tts_status = gr.Textbox(label="状態", interactive=False)

    with gr.Tab("翻訳"):
        direction = gr.Radio(["英→日", "日→英"], value="英→日", label="翻訳方向")
        source_text = gr.Textbox(label="入力文", lines=4)
        trans_btn = gr.Button("翻訳する")
        trans_output = gr.Markdown()

    with gr.Tab("Writingテスト"):
        make_writing_btn = gr.Button("課題を作る")
        writing_prompt_md = gr.Markdown()
        writing_answer = gr.Textbox(label="あなたの英作文", lines=5)
        grade_writing_btn = gr.Button("採点")
        writing_result_md = gr.Markdown()

    with gr.Tab("英日/日英 文章テスト"):
        test_direction = gr.Radio(["英→日", "日→英"], value="英→日", label="出題方向")
        make_test_btn = gr.Button("問題を作る")
        test_question_md = gr.Markdown()
        test_answer = gr.Textbox(label="あなたの解答", lines=4)
        grade_test_btn = gr.Button("判定")
        test_feedback_md = gr.Markdown()

    search_btn.click(
        lookup_word,
        inputs=[term_input],
        outputs=[search_result, dictionary_state, search_audio, search_detail],
    )
    add_btn.click(
        upsert_word,
        inputs=[dictionary_state],
        outputs=[add_status, word_table, word_select, listen_word, word_detail],
    )

    reload_btn.click(list_words_table, outputs=[word_table, word_select, listen_word])
    listen_reload_btn.click(list_words_table, outputs=[word_table, word_select, listen_word])
    word_select.change(show_word_detail, inputs=[word_select], outputs=[word_detail])
    delete_btn.click(
        delete_word,
        inputs=[word_select],
        outputs=[delete_status, word_table, word_select, listen_word],
    )

    tts_btn.click(synthesize_tts, inputs=[listen_word, speak_target], outputs=[tts_audio, tts_status])
    trans_btn.click(translate_text, inputs=[direction, source_text], outputs=[trans_output])

    make_writing_btn.click(create_writing_prompt, outputs=[writing_prompt_md, writing_state])
    grade_writing_btn.click(grade_writing, inputs=[writing_answer, writing_state], outputs=[writing_result_md])

    make_test_btn.click(create_translation_test, inputs=[test_direction], outputs=[test_question_md, translation_test_state])
    grade_test_btn.click(check_translation_test, inputs=[test_answer, translation_test_state], outputs=[test_feedback_md])

    app.load(list_words_table, outputs=[word_table, word_select, listen_word])


if __name__ == "__main__":
    app.launch(share=True, auth=("student", "edu2026"))