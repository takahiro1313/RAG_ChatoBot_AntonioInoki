import streamlit as st
from openai import OpenAI
import numpy as np
import PyPDF2
import os

# PDF読み込み用ライブラリ(オプショナル)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# OpenAI APIの設定
# Streamlit Cloud用とローカル用の両方に対応
try:
    # Streamlit Cloudの場合
    api_key = st.secrets["OPENAI_API_KEY"]
except (FileNotFoundError, KeyError):
    # ローカル環境の場合
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEYが設定されていません。環境変数を確認してください。")
    st.info("""
    **Streamlit Cloudの場合:**
    Settings → Secrets で以下を設定してください:
```
    OPENAI_API_KEY = "your_key_here"
```
    
    **ローカルの場合:**
    .envファイルを作成してください:
```
    OPENAI_API_KEY=your_key_here
```
    """)
    st.stop()

client = OpenAI(api_key=api_key)

# ページ設定
st.set_page_config(
    page_title="アントニオ猪木FAQチャットボット",
    page_icon="💪",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 18px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# PDF読み込み関数(複数ライブラリ対応)
@st.cache_data
def load_pdf(file_path="faq.pdf"):
    """PDFファイルを読み込んでテキストを抽出(複数ライブラリ対応)"""

    # 方法1: PyMuPDF(最強)
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            if text and text.count('�') < 10:  # 文字化けチェック
                st.success("✅ PyMuPDF で読み込み成功")
                return text
        except Exception as e:
            st.warning(f"PyMuPDF読み込み失敗: {str(e)}")
    else:
        st.info("💡 PyMuPDF未インストール: pip install PyMuPDF")

    # 方法2: pdfplumber(日本語に強い)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if text and text.count('�') < 10:  # 文字化けチェック
                st.success("✅ pdfplumber で読み込み成功")
                return text
            else:
                st.warning("⚠️ pdfplumberで文字化けを検出")
        except Exception as e:
            st.warning(f"pdfplumber読み込み失敗: {str(e)}")
    else:
        st.info("💡 pdfplumber未インストール: pip install pdfplumber")

    # 方法3: PyPDF2(フォールバック)
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        if text and text.count('�') < 20:  # PyPDF2は文字化けしやすいので緩い基準
            st.warning("⚠️ PyPDF2で読み込み(文字化けの可能性あり)")
            return text
    except FileNotFoundError:
        st.error(f"❌ {file_path} が見つかりません。faq.pdfをプロジェクトフォルダに配置してください。")
        return ""
    except Exception as e:
        st.error(f"❌ PyPDF2読み込み失敗: {str(e)}")

    # 全て失敗
    st.error("❌ 全ての方法でPDF読み込みに失敗しました")
    st.info("""
    💡 解決策:
    1. pip install PyMuPDF pdfplumber
    2. PDFを作り直す(Word → PDF)
    3. テキストファイルで作成
    """)
    return ""

# テキストファイル読み込み(文字化け対策の代替案)
@st.cache_data
def load_txt_fallback(file_path="faq.txt"):
    """文字化け対策: テキストファイルから読み込み"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# FAQ読み込み・チャンク分割
@st.cache_data
def load_faq():
    """FAQ読み込み(PDF→テキストファイルの順で試行)"""
    # まずPDFを試す
    content = load_pdf("faq.pdf")

    # PDFが読めなかったらテキストファイルを試す
    if not content:
        content = load_txt_fallback("faq.txt")
        if content:
            st.info("📝 faq.txtから読み込みました")
    else:
        st.success("📄 faq.pdfから読み込みました")

    if not content:
        st.error("❌ faq.pdf または faq.txt を配置してください")
        return []

    # Q&A単位で分割
    chunks = []
    lines = content.split("\n")
    current_chunk = ""

    for line in lines:
        if line.startswith("Q") and current_chunk:
            # 新しいQが始まったら、前のチャンクを保存
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 空のチャンクを除外
    chunks = [c for c in chunks if c and len(c) > 20]

    # 長すぎるチャンクを分割(1チャンク最大1500文字)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 1500:
            # 1500文字ごとに分割
            for i in range(0, len(chunk), 1500):
                final_chunks.append(chunk[i:i+1500])
        else:
            final_chunks.append(chunk)

    return final_chunks

# Embedding取得(バッチ処理対応)
@st.cache_data
def get_embeddings(texts):
    """テキストリストをEmbeddingに変換(バッチ処理)"""
    if not texts:
        return []

    all_embeddings = []
    batch_size = 50  # 一度に処理する数(トークン制限対策)

    # テキストをバッチに分割して処理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

# コサイン類似度計算
def cosine_similarity(vec1, vec2):
    """2つのベクトルのコサイン類似度を計算"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 関連チャンク検索
def search_relevant_chunks(query, chunks, embeddings, top_k=3):
    """質問に最も関連するチャンクを検索"""
    if not chunks or not embeddings:
        return [], []

    # 質問をEmbedding化
    query_embedding = get_embeddings([query])[0]

    # 各チャンクとの類似度を計算
    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    # 類似度が高い順にソート
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_chunks = [chunks[i] for i in top_indices]
    scores = [similarities[i] for i in top_indices]

    return relevant_chunks, scores

# RAG回答生成
def generate_rag_response(query, relevant_chunks):
    """関連チャンクをコンテキストとして回答生成"""

    # コンテキスト構築
    context = "\n\n".join(relevant_chunks)

    # プロンプト作成
    prompt = f"""あなたはアントニオ猪木について詳しい専門家です。
以下の参考情報をもとに、ユーザーの質問に答えてください。

【参考情報】
{context}

【重要な指示】
- 参考情報に記載されている内容をもとに回答してください
- 参考情報に無い内容は推測せず、「参考情報には記載がありません」と伝えてください
- 回答は丁寧で分かりやすい日本語で
- 必要に応じて、猪木の名言や精神を交えて回答してください

【ユーザーの質問】
{query}"""

    # ChatGPT APIで回答生成
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたはアントニオ猪木についての質問に答える専門家です。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faq_chunks" not in st.session_state:
    with st.spinner("📚 FAQを読み込み中..."):
        st.session_state.faq_chunks = load_faq()

        if not st.session_state.faq_chunks:
            st.error("⚠️ FAQの読み込みに失敗しました。")
            st.info("💡 faq.pdf または faq.txt を配置してください")
            st.stop()

if "faq_embeddings" not in st.session_state:
    with st.spinner("🔍 FAQをベクトル化中..."):
        st.session_state.faq_embeddings = get_embeddings(st.session_state.faq_chunks)

# UIレイアウト
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-title">💪 アントニオ猪木 FAQチャットボット</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">元気ですか〜! 猪木について何でも聞いてください!</div>', unsafe_allow_html=True)

with col2:
    st.metric("FAQ件数", len(st.session_state.faq_chunks))

# サイドバー
with st.sidebar:
    st.header("⚙️ RAG設定")

    # ライブラリ状況表示
    st.markdown("### 📚 PDF読み込みライブラリ")
    if HAS_PYMUPDF:
        st.success("✅ PyMuPDF (推奨)")
    else:
        st.error("❌ PyMuPDF 未インストール")

    if HAS_PDFPLUMBER:
        st.success("✅ pdfplumber")
    else:
        st.error("❌ pdfplumber 未インストール")

    st.success("✅ PyPDF2 (標準)")

    if not HAS_PYMUPDF and not HAS_PDFPLUMBER:
        st.warning("⚠️ 日本語PDF対応ライブラリがありません")
        st.code("pip install PyMuPDF pdfplumber")

    st.markdown("---")

    top_k = st.slider(
        "参考にするFAQ数",
        min_value=1,
        max_value=5,
        value=3,
        help="質問に関連するFAQをいくつ参考にするか選択"
    )

    show_context = st.checkbox(
        "参考にしたFAQを表示",
        value=True,
        help="AIがどのFAQを参考にしたか見ることができます"
    )

    st.markdown("---")
    st.markdown("### 💡 RAGとは?")
    st.markdown("""
    **Retrieval-Augmented Generation**

    1. 質問を受け取る
    2. PDFから関連FAQを検索
    3. 検索結果を元に回答生成

    → 正確で根拠のある回答!
    """)

    st.markdown("---")
    st.markdown("### 🔍 類似度の仕組み")
    st.markdown(f"""
    **現在の設定**: 上位{top_k}個のFAQを参考

    **計算方法**:
    1. 質問をベクトル化(数値化)
    2. 各FAQもベクトル化済み
    3. コサイン類似度で比較
    4. 類似度が高い順に選択

    **類似度の見方**:
    - 1.0に近い = とても関連
    - 0.5程度 = ある程度関連
    - 0.2以下 = あまり関連なし
    """)

    st.markdown("---")
    st.markdown("### 🚀 使い方")
    st.markdown("""
    - 猪木について質問してください
    - 例:「必殺技は?」
    - 例:「アリ戦について」
    - 例:「名言を教えて」
    """)

    if st.button("💬 チャット履歴をクリア"):
        st.session_state.messages = []
        st.rerun()

# チャット履歴表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # コンテキスト表示(ユーザーが有効にしている場合)
        if show_context and "context" in message:
            with st.expander(f"📄 参考にした{len(message['context'])}件のFAQ"):
                for i, chunk in enumerate(message["context"], 1):
                    score = message['scores'][i-1]

                    # 類似度に応じて色を変える
                    if score > 0.8:
                        color = "🟢"
                        level = "とても関連"
                    elif score > 0.5:
                        color = "🟡"
                        level = "関連"
                    elif score > 0.3:
                        color = "🟠"
                        level = "やや関連"
                    else:
                        color = "🔴"
                        level = "あまり関連なし"

                    st.markdown(f"**{color} FAQ{i}** (類似度: {score:.2f} - {level})")

                    # 文字化けチェック
                    if chunk.count('�') > 5:
                        st.warning("⚠️ この内容は文字化けしている可能性があります")
                        st.text("文字化け対策: faq.txtファイルを使用することをお勧めします")
                    else:
                        # 正常な場合は内容を表示
                        display_text = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.text(display_text)

                    st.markdown("---")

# ユーザー入力
if prompt := st.chat_input("猪木について質問してください (例: 必殺技は?)"):
    # ユーザーメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI応答生成
    with st.chat_message("assistant"):
        with st.spinner("🔍 関連情報を検索中..."):
            # 関連チャンク検索
            relevant_chunks, scores = search_relevant_chunks(
                prompt,
                st.session_state.faq_chunks,
                st.session_state.faq_embeddings,
                top_k=top_k
            )

        with st.spinner("💬 回答を生成中..."):
            # RAG回答生成
            response = generate_rag_response(prompt, relevant_chunks)

        st.markdown(response)

        # コンテキスト表示
        if show_context:
            with st.expander(f"📄 参考にした{len(relevant_chunks)}件のFAQ"):
                for i, (chunk, score) in enumerate(zip(relevant_chunks, scores), 1):
                    # 類似度に応じて色を変える
                    if score > 0.8:
                        color = "🟢"
                        level = "とても関連"
                    elif score > 0.5:
                        color = "🟡"
                        level = "関連"
                    elif score > 0.3:
                        color = "🟠"
                        level = "やや関連"
                    else:
                        color = "🔴"
                        level = "あまり関連なし"

                    st.markdown(f"**{color} FAQ{i}** (類似度: {score:.2f} - {level})")

                    # 文字化けチェック
                    if chunk.count('�') > 5:
                        st.warning("⚠️ この内容は文字化けしている可能性があります")
                        st.text("文字化け対策: faq.txtファイルを使用することをお勧めします")
                    else:
                        # 正常な場合は内容を表示
                        display_text = chunk[:400] + "..." if len(chunk) > 400 else chunk
                        st.text(display_text)

                    st.markdown("---")

    # アシスタントメッセージを保存
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "context": relevant_chunks,
        "scores": scores
    })

# フッター
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <strong>燃える闘魂 🔥</strong> | このチャットボットはRAG技術を使用しています<br>
    <small>「元気があれば何でもできる!」 - アントニオ猪木</small>
</div>
""", unsafe_allow_html=True)