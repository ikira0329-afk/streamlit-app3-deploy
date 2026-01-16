from dotenv import load_dotenv
load_dotenv()


import streamlit as st

# ChatOpenAI を明示的に固定インポートします（Option A）
try:
    from langchain_openai.chat_models.base import ChatOpenAI
except Exception:
    st.warning("`langchain_openai.chat_models.base` から ChatOpenAI をインポートできませんでした。環境を確認してください。")
    st.stop()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ConversationBufferMemory の互換インポート（環境によりモジュール名が異なるため）
ConversationBufferMemory = None
try:
    from langchain.memory import ConversationBufferMemory as _CBM
    ConversationBufferMemory = _CBM
except Exception:
    try:
        from langchain_core.memory import ConversationBufferMemory as _CBM2
        ConversationBufferMemory = _CBM2
    except Exception:
        ConversationBufferMemory = None

# 会話履歴を保持するためのメモリを初期化（ConversationBufferMemory が無ければ簡易実装にフォールバック）
class SimpleMemory:
    def __init__(self, key="chat_history"):
        self.key = key
        if self.key not in st.session_state:
            st.session_state[self.key] = []

    def load_memory_variables(self, _=None):
        chat = st.session_state.get(self.key, [])
        lines = []
        for m in chat:
            role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) or "user"
            content = (m.get("content") if isinstance(m, dict) else getattr(m, "content", None)) or ""
            prefix = "User" if str(role).lower().startswith("user") else "Assistant"
            lines.append(f"{prefix}: {content}")
        return {self.key: "\n".join(lines)}

    def save_context(self, inputs, outputs):
        user_text = inputs.get("input") if isinstance(inputs, dict) else ""
        out_text = outputs.get("output") if isinstance(outputs, dict) else str(outputs)
        st.session_state[self.key].append({"role": "user", "content": user_text})
        st.session_state[self.key].append({"role": "assistant", "content": out_text})

if ConversationBufferMemory is not None:
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = st.session_state["memory"]
else:
    # フォールバックはファイル永続化する SimpleMemory を使う
    memory = SimpleMemory("chat_history")

# 表示レイアウトの設定
st.title("専門家になんでも聞いて！")
st.write(
    "この質問コーナーは、健康維持について、理学療法士と管理栄養士の専門家に質問できるサービスです。")

selected_item = st.radio(
    "2人の専門家の中から、質問したい専門家を選んでください。",
    ["理学療法士","管理栄養士"],
)
st.divider()

st.write(f"選択された専門家:{selected_item}")

# 会話履歴を入力欄の上に表示する（ユーザー要望）
st.subheader("これまでの会話")

# 履歴をチャット風に表示するヘルパー
def render_chat():
    mem_vars = memory.load_memory_variables({})  # -> {'chat_history': [...] or "string"}
    chat = mem_vars.get("chat_history")
    if not chat:
        st.info("履歴はまだありません。")
        return

    # case A: list (メッセージオブジェクトや辞書のリスト)
    if isinstance(chat, list):
        for m in chat:
            if isinstance(m, dict):
                role = m.get("role") or m.get("type") or "user"
                content = m.get("content") or m.get("text") or ""
            else:
                role = getattr(m, "role", None) or getattr(m, "type", None) or "user"
                content = getattr(m, "content", None) or getattr(m, "text", "") or ""
            sender = "user" if str(role).lower().startswith("user") else "assistant"
            with st.chat_message(sender):
                st.markdown(content)
        return

    # case B: 文字列（"User: ...\nAssistant: ..." のような形式）
    chat_str = str(chat)
    for line in chat_str.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("user:"):
            text = line.split(":", 1)[1].strip()
            with st.chat_message("user"):
                st.markdown(text)
        elif line.lower().startswith("assistant:") or line.lower().startswith("system:"):
            text = line.split(":", 1)[1].strip()
            with st.chat_message("assistant"):
                st.markdown(text)
        else:
            with st.chat_message("assistant"):
                st.markdown(line)

render_chat()

# 最新の回答を入力欄の直上に表示（セッションに保存されている場合）
if st.session_state.get("last_answer"):
    st.write("### 最新の回答")
    st.info(st.session_state.get("last_answer"))

st.markdown("---")

input_message = st.text_input(
    label="健康に関するお悩みやその他、質問を下記の欄に入力してください。"
)

# LLMに渡す役割の内容
system_message = {
    "理学療法士":(
        "あなたは理学療法士です。お客様の健康維持のため、"
        "適切なアドバイスで、身体的なサポートを提案し、"
        "リハビリテーションに関する専門家です。"
        "専門用語を使用せず、日本語で中学生でも理解できるように"
        "分かりやすく説明してください。"
        "200文字以内で回答してください。"
     ),
    "管理栄養士":(
        "あなたは管理栄養士です。お客様の健康維持のため、"
        "適切なアドバイスで、栄養バランスの取れた食事プランを提案し、"
        "健康的なライフスタイルに関する専門家です。"
        "専門用語を使用せず、日本語で中学生でも理解できるように"
        "分かりやすく説明してください。"
        "200文字以内で回答してください。"
    ),
}
# LLMのグレード指定と出力パーサーの設定（文字列として出力）
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5)
parser = StrOutputParser()

 

# プロンプト（履歴は明示的なメッセージ型ではなく human テンプレート内の変数として渡す）
# langchain の ChatPromptTemplate は 'history' のような未定義のメッセージ種別を受け付けないため、
# 履歴を human メッセージに埋め込んで渡します。
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message[selected_item]),
        ("human", "{chat_history}\n\n{input}"),
    ]
)
answer_chain = answer_prompt | llm | parser

point_prompt = ChatPromptTemplate.from_template(
    """あなたは質問に対して、専門的な知識がある専門家です。
    以下の質問が「健康」「身体」「栄養」「運動」「リハビリ」「メンタル」等に関する内容なら「はい」、
    全く関係ない内容であれば、「いいえ」とだけ答えてください。

    質問:{question}
    """
    )
point_chain = point_prompt | llm | parser




if st.button("実行"):
    st.divider()
    if not input_message:
        st.error("質問を入力してください。")
    else:
        # まずは関連性チェック（既存の point_chain を使用）
        point_result = point_chain.invoke({"question": input_message})
        if "いいえ" in point_result:
            st.write("この質問は健康や栄養に関する内容ではないです。質問内容を変更してください。")
        else:
            # メモリから履歴（list or str）を取得して一緒に渡す
            mem_vars = memory.load_memory_variables({})               # -> {'chat_history': ...}
            invoke_vars = {"input": input_message, **mem_vars}
            try:
                result = answer_chain.invoke(invoke_vars)
            except Exception as e:
                st.error("回答の取得に失敗しました。時間を置いて再度お試しください。")
            else:
                # 成功したら履歴に保存（ここで初めて保存）
                memory.save_context({"input": input_message}, {"output": result})
                # 最新の回答をセッションに保存して上部で表示可能にする
                st.session_state["last_answer"] = str(result)

                # 回答を表示
                st.write("### 回答:")
                st.write(result)

                # （履歴は画面上部のみで表示するため、下部の再表示は行いません）