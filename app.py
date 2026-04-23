import gradio as gr
import uuid
import numpy as np
import pandas as pd

from src.database.connection import get_engine
from src.database.repository import (
    listar_ids_produtos,
    listar_carrinho_cliente,
    adicionar_item_carrinho,
    remover_item_carrinho,
    limpar_carrinho_db,
    obter_todos_produtos_embeddings
)
from src.recommender.model import carregar_modelo

engine = get_engine()
model = carregar_modelo()

SESSION_ID = str(uuid.uuid4())

PRODUTOS_BASE = obter_todos_produtos_embeddings(engine)
LISTA_IDS = listar_ids_produtos(engine, limite=24)


def recomendar():
    carrinho = listar_carrinho_cliente(engine, SESSION_ID)
    base = PRODUTOS_BASE.copy()

    if len(carrinho) == 0:
        base["score"] = 0.5
        return base.head(6)

    ids = carrinho["product_id"].tolist()
    vetores = base[base["product_id"].isin(ids)]["embedding"].tolist()

    vetor_user = np.mean(np.vstack(vetores), axis=0).astype("float32")

    matriz_prod = np.vstack(base["embedding"].values).astype("float32")
    matriz_user = np.repeat(vetor_user.reshape(1, -1), len(base), axis=0).astype("float32")

    entrada = np.hstack([matriz_user, matriz_prod]).astype("float32")

    scores = model.predict(entrada, verbose=0, batch_size=256).flatten()

    base["score"] = scores
    base = base[~base["product_id"].isin(ids)].sort_values("score", ascending=False)

    return base.head(6)


def get_produtos():
    df = PRODUTOS_BASE[PRODUTOS_BASE["product_id"].isin(LISTA_IDS)].copy()
    return df["product_id"].tolist()


def render_produtos(ids):
    return gr.update(choices=[(i, i) for i in ids])


def render_recomendacoes(df):
    html = ""
    for _, r in df.iterrows():
        html += f"""
        <div style="background:#1e1e1e;padding:10px;border-radius:10px;margin-bottom:10px">
            <b>{r['product_id']}</b><br>
            R$ {float(r['price']):,.2f}<br>
            Score: {float(r['score'])*100:.2f}%
        </div>
        """
    return html


def render_carrinho():
    df = listar_carrinho_cliente(engine, SESSION_ID)

    if df is None or len(df) == 0:
        return "<div style='color:#aaa'>Carrinho vazio</div>"

    html = ""

    for _, r in df.iterrows():
        html += f"""
        <div style="background:#1e1e1e;padding:12px;border-radius:12px;margin-bottom:10px;display:flex;justify-content:space-between">
            <div>
                <b>{r['product_id']}</b><br>
                R$ {float(r['price']):,.2f}
            </div>
            <div style="color:#4f46e5;font-weight:bold;">no carrinho</div>
        </div>
        """

    return html


def render_total():
    df = listar_carrinho_cliente(engine, SESSION_ID)

    if df is None or len(df) == 0:
        total = 0.0
    else:
        total = pd.to_numeric(df["price"], errors="coerce").fillna(0).sum()

    return f"""
    <div style="background:#111827;padding:14px;border-radius:12px;margin-top:10px;border:1px solid #4f46e5;text-align:center">
        <div style="font-size:12px;color:#aaa;">TOTAL DO CARRINHO</div>
        <div style="font-size:20px;font-weight:bold;color:#4f46e5">
            R$ {total:,.2f}
        </div>
    </div>
    """


def build_state():
    prod_ids = get_produtos()
    rec = recomendar()

    return (
        render_produtos(prod_ids),
        render_carrinho(),
        render_recomendacoes(rec),
        render_total(),
        prod_ids,
        get_carrinho()
    )


def get_carrinho():
    df = listar_carrinho_cliente(engine, SESSION_ID)
    return df["product_id"].tolist() if len(df) > 0 else []


def adicionar(pid):
    adicionar_item_carrinho(engine, SESSION_ID, pid)
    return build_state()


def remover(pid):
    remover_item_carrinho(engine, SESSION_ID, pid)
    return build_state()


def limpar():
    limpar_carrinho_db(engine, SESSION_ID)
    return build_state()


def load():
    return build_state()


css = """
.gradio-container{
    background:#0b0b0b !important;
    color:white !important;
    max-width:1400px !important;
    margin:auto;
}

h1,h2{
    color:white !important;
}

button{
    background:#4f46e5 !important;
    color:white !important;
    border-radius:8px !important;
}
"""


with gr.Blocks(css=css) as demo:

    state_prod = gr.State()
    state_cart = gr.State()

    with gr.Row():

        with gr.Column(scale=1, min_width=350):
            gr.Markdown("## Adicionar Produto")
            produto = gr.Dropdown(label="Produtos")
            add_btn = gr.Button("Adicionar")

        with gr.Column(scale=1, min_width=350):
            gr.Markdown("## Carrinho")
            carrinho = gr.HTML()
            total = gr.HTML()
            remove_btn = gr.Button("Remover último")
            limpar_btn = gr.Button("Limpar")

        with gr.Column(scale=1, min_width=350):
            gr.Markdown("## Recomendações")
            recom = gr.HTML()

    demo.load(
        load,
        outputs=[produto, carrinho, recom, total, state_prod, state_cart]
    )

    add_btn.click(
        adicionar,
        inputs=[produto],
        outputs=[produto, carrinho, recom, total, state_prod, state_cart]
    )

    remove_btn.click(
        remover,
        inputs=[produto],
        outputs=[produto, carrinho, recom, total, state_prod, state_cart]
    )

    limpar_btn.click(
        limpar,
        outputs=[produto, carrinho, recom, total, state_prod, state_cart]
    )


if __name__ == "__main__":
    demo.launch()