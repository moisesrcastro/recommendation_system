import ast
import numpy as np
import pandas as pd
from sqlalchemy import text


# ==================================================
# HELPERS
# ==================================================
def _to_vector(value):
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        return value.astype("float32")

    if isinstance(value, list):
        return np.array(value, dtype="float32")

    if isinstance(value, str):
        try:
            return np.array(
                ast.literal_eval(value),
                dtype="float32"
            )
        except:
            return None

    try:
        return np.array(value, dtype="float32")
    except:
        return None


# ==================================================
# CARRINHO
# ==================================================
def criar_tabela_carrinho(engine):
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS cart_items (
            session_id TEXT,
            product_id TEXT
        )
        """))

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_cart_session
        ON cart_items(session_id)
        """))


def adicionar_item_carrinho(
    engine,
    session_id,
    product_id
):
    criar_tabela_carrinho(engine)

    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO cart_items (
                session_id,
                product_id
            )
            VALUES (
                :session_id,
                :product_id
            )
            """),
            {
                "session_id": str(session_id),
                "product_id": str(product_id)
            }
        )


def remover_item_carrinho(
    engine,
    session_id,
    product_id
):
    with engine.begin() as conn:
        conn.execute(
            text("""
            DELETE FROM cart_items
            WHERE session_id = :session_id
              AND product_id = :product_id
            """),
            {
                "session_id": str(session_id),
                "product_id": str(product_id)
            }
        )


def limpar_carrinho_db(
    engine,
    session_id
):
    with engine.begin() as conn:
        conn.execute(
            text("""
            DELETE FROM cart_items
            WHERE session_id = :session_id
            """),
            {"session_id": str(session_id)}
        )


def listar_carrinho_cliente(
    engine,
    session_id
):
    criar_tabela_carrinho(engine)

    query = """
    SELECT
        c.product_id,
        p.product_category_name,
        COALESCE(precos.price,0) AS price
    FROM cart_items c
    LEFT JOIN products p
        ON c.product_id = p.product_id
    LEFT JOIN (
        SELECT
            product_id,
            AVG(CAST(price AS NUMERIC)) AS price
        FROM order_items
        GROUP BY product_id
    ) precos
        ON c.product_id = precos.product_id
    WHERE c.session_id = :session_id
    """

    try:
        return pd.read_sql(
            text(query),
            engine,
            params={"session_id": str(session_id)}
        )

    except:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_category_name",
                "price"
            ]
        )


def calcular_total_carrinho(
    engine,
    session_id
):
    df = listar_carrinho_cliente(
        engine,
        session_id
    )

    if len(df) == 0:
        return "R$ 0,00"

    total = float(df["price"].fillna(0).sum())

    total = f"{total:,.2f}"
    total = total.replace(",", "X")
    total = total.replace(".", ",")
    total = total.replace("X", ".")

    return f"R$ {total}"


# ==================================================
# PRODUTOS
# ==================================================
def listar_categorias(engine):
    query = """
    SELECT DISTINCT product_category_name
    FROM products
    WHERE product_category_name IS NOT NULL
    ORDER BY product_category_name
    """

    df = pd.read_sql(query, engine)

    return df["product_category_name"].astype(str).tolist()


def listar_ids_produtos(
    engine,
    limite=1000
):
    query = """
    SELECT product_id
    FROM products
    LIMIT :limite
    """

    df = pd.read_sql(
        text(query),
        engine,
        params={"limite": limite}
    )

    return df["product_id"].astype(str).tolist()


def buscar_produtos_db(
    engine,
    texto="",
    categoria="Todos",
    limite=100
):
    query = """
    SELECT
        p.product_id,
        p.product_category_name,
        COALESCE(precos.price,0) AS price
    FROM products p
    LEFT JOIN (
        SELECT
            product_id,
            AVG(CAST(price AS NUMERIC)) AS price
        FROM order_items
        GROUP BY product_id
    ) precos
        ON p.product_id = precos.product_id
    WHERE
        (:texto = ''
         OR p.product_id ILIKE :texto_like)

        AND

        (
            :categoria = 'Todos'
            OR p.product_category_name = :categoria
        )

    LIMIT :limite
    """

    return pd.read_sql(
        text(query),
        engine,
        params={
            "texto": texto,
            "texto_like": f"%{texto}%",
            "categoria": categoria,
            "limite": limite
        }
    )


# ==================================================
# EMBEDDINGS
# ==================================================
def obter_embedding_produto(
    engine,
    product_id
):
    query = """
    SELECT embedding
    FROM product_embeddings
    WHERE product_id = :product_id
    LIMIT 1
    """

    df = pd.read_sql(
        text(query),
        engine,
        params={"product_id": str(product_id)}
    )

    if len(df) == 0:
        return None

    return _to_vector(
        df["embedding"].iloc[0]
    )


def obter_embeddings_produtos_lista(
    engine,
    lista_ids
):
    if not lista_ids:
        return {}

    query = """
    SELECT
        product_id,
        embedding
    FROM product_embeddings
    WHERE product_id IN :ids
    """

    df = pd.read_sql(
        text(query),
        engine,
        params={"ids": tuple(lista_ids)}
    )

    df["embedding"] = df["embedding"].apply(
        _to_vector
    )

    return dict(
        zip(
            df["product_id"],
            df["embedding"]
        )
    )


def obter_todos_produtos_embeddings(engine):
    query = """
    SELECT
        pe.product_id,
        pe.embedding,
        p.product_category_name,
        COALESCE(precos.price,0) AS price
    FROM product_embeddings pe

    LEFT JOIN products p
        ON pe.product_id = p.product_id

    LEFT JOIN (
        SELECT
            product_id,
            AVG(CAST(price AS NUMERIC)) AS price
        FROM order_items
        GROUP BY product_id
    ) precos
        ON pe.product_id = precos.product_id
    """

    df = pd.read_sql(query, engine)

    df["embedding"] = df["embedding"].apply(
        _to_vector
    )

    df = df[
        df["embedding"].notnull()
    ].reset_index(drop=True)

    return df