"""Dados, modelo de preço e análises dos anúncios de Nova York (Airbnb, 2019).

Base: desafio de precificação do programa Lighthouse (Indicium), uma tradução para o
português do conjunto público "New York City Airbnb Open Data" (Inside Airbnb, 2019).
Tudo aqui é determinístico; o agente só chama estas funções.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "listings_nyc_2019.csv"
MODEL_PATH = ROOT / "models" / "preco_nyc.joblib"
PRECO_MIN, PRECO_MAX = 10, 1000  # fora disso: preço zerado ou anúncios atípicos (0,5% da base)

# Anúncio de teste do desafio (retirado da base): preço real US$ 225
TESTE = {"nome": "Skylit Midtown Castle", "bairro_group": "Manhattan", "bairro": "Midtown", "latitude": 40.75362,
         "longitude": -73.98377, "room_type": "Entire home/apt", "minimo_noites": 1, "numero_de_reviews": 45,
         "reviews_por_mes": 0.38, "calculado_host_listings_count": 2, "disponibilidade_365": 355}
TESTE_REAL = 225.0

NUM = ["latitude", "longitude", "minimo_noites", "numero_de_reviews", "reviews_por_mes", "calculado_host_listings_count",
       "disponibilidade_365"]
CAT = ["bairro_group", "bairro", "room_type"]
PALAVRAS = ["luxury", "view", "loft", "penthouse", "townhouse", "spacious", "cozy", "private", "shared", "studio"]


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["reviews_por_mes"] = df["reviews_por_mes"].fillna(0.0)
    df["nome"] = df["nome"].fillna("")
    return df


def clean() -> pd.DataFrame:
    df = load()
    return df[(df["price"] >= PRECO_MIN) & (df["price"] <= PRECO_MAX)].reset_index(drop=True)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUM + CAT].copy()
    X["minimo_noites"] = X["minimo_noites"].clip(upper=60)
    nome = df["nome"].fillna("").str.lower() if "nome" in df else pd.Series("", index=df.index)
    for w in PALAVRAS:
        X[f"w_{w}"] = nome.str.contains(w, regex=False).astype(float)
    return X


def _pipeline(est) -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CAT)],
                            remainder="passthrough", sparse_threshold=0)
    return Pipeline([("pre", pre), ("model", est)])


MODELOS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gbm": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=0),
}


@lru_cache(maxsize=1)
def evaluate() -> dict:
    """Holdout de 20% (semente 42). Os modelos aprendem o log do preço; o erro é medido em dólares."""
    df = clean()
    tr, te = train_test_split(df, test_size=0.2, random_state=42)
    y = te["price"].to_numpy(dtype=float)
    rows = []
    med = tr.groupby(["bairro", "room_type"])["price"].median()
    fallback = tr.groupby("room_type")["price"].median()
    base = np.array([med.get((b, r), fallback[r]) for b, r in zip(te["bairro"], te["room_type"])], dtype=float)
    rows.append(_metrics("mediana_bairro_tipo", y, base))
    preds = {}
    for name, make in MODELOS.items():
        m = _pipeline(make()).fit(_features(tr), np.log1p(tr["price"]))
        p = np.expm1(m.predict(_features(te)))
        preds[name] = p
        rows.append(_metrics(name, y, p))
    return {"tabela": rows, "n_treino": len(tr), "n_teste": len(te), "residuos_log": (np.log1p(y) - np.log1p(preds["gbm"])).tolist()}


def _metrics(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    return {"modelo": name, "mae": round(float(np.mean(np.abs(y - p))), 2), "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
            "erro_pct_mediano": round(float(np.median(np.abs(y - p) / y) * 100), 1)}


@lru_cache(maxsize=1)
def model() -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    df = clean()
    m = _pipeline(MODELOS["gbm"]()).fit(_features(df), np.log1p(df["price"]))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(m, MODEL_PATH)
    return m


def centroid(bairro: str) -> tuple[float, float]:
    df = load()
    sub = df[df["bairro"].str.lower() == bairro.lower()]
    if sub.empty:
        raise ValueError(f"bairro desconhecido: {bairro}")
    return float(sub["latitude"].median()), float(sub["longitude"].median())


def predict(anuncio: dict) -> dict:
    a = {"minimo_noites": 2, "numero_de_reviews": 10, "reviews_por_mes": 0.5, "calculado_host_listings_count": 1,
         "disponibilidade_365": 100, "nome": "", **anuncio}
    if a.get("room_type") not in ("Entire home/apt", "Private room", "Shared room"):
        raise ValueError("room_type deve ser 'Entire home/apt', 'Private room' ou 'Shared room'")
    if "bairro" not in a:
        raise ValueError("informe o bairro")
    if "latitude" not in a or "longitude" not in a:
        a["latitude"], a["longitude"] = centroid(a["bairro"])
    if "bairro_group" not in a:
        a["bairro_group"] = load().loc[load()["bairro"].str.lower() == a["bairro"].lower(), "bairro_group"].iloc[0]
    p = float(np.expm1(model().predict(_features(pd.DataFrame([a])))[0]))
    lo, hi = np.quantile(evaluate()["residuos_log"], [0.1, 0.9])
    return {"preco": p, "p10": float(np.expm1(np.log1p(p) + lo)), "p90": float(np.expm1(np.log1p(p) + hi))}


@lru_cache(maxsize=1)
def importance() -> list[dict]:
    df = clean().sample(4000, random_state=1)
    X, y = _features(df), df["price"].to_numpy(dtype=float)
    base = np.mean(np.abs(y - np.expm1(model().predict(X))))
    rng = np.random.default_rng(0)
    out = []
    grupos = {"localização (lat/long)": ["latitude", "longitude"], "tipo de acomodação": ["room_type"], "bairro": ["bairro", "bairro_group"],
              "disponibilidade no ano": ["disponibilidade_365"], "mínimo de noites": ["minimo_noites"],
              "reviews": ["numero_de_reviews", "reviews_por_mes"], "anúncios do anfitrião": ["calculado_host_listings_count"],
              "palavras do nome": [f"w_{w}" for w in PALAVRAS]}
    for nome, cols in grupos.items():
        d = []
        for _ in range(3):
            X2 = X.copy()
            idx = rng.permutation(len(X2))
            for c in cols:
                X2[c] = X2[c].to_numpy()[idx]
            d.append(np.mean(np.abs(y - np.expm1(model().predict(X2)))) - base)
        out.append({"grupo": nome, "aumento_mae": round(float(np.mean(d)), 2)})
    return sorted(out, key=lambda r: -r["aumento_mae"])


def query(bairro_group: str | None = None, bairro: str | None = None, room_type: str | None = None, preco_min: float | None = None,
          preco_max: float | None = None, ordenar_por: str = "preco", limite: int | None = 10) -> pd.DataFrame:
    ordem = {"preco": "price", "reviews": "numero_de_reviews", "disponibilidade": "disponibilidade_365"}
    if ordenar_por not in ordem:
        raise ValueError(f"ordenar_por deve ser um de {list(ordem)}")
    df = load()
    m = pd.Series(True, index=df.index)
    if bairro_group:
        m &= df["bairro_group"].str.lower() == bairro_group.lower()
    if bairro:
        m &= df["bairro"].str.lower() == bairro.lower()
    if room_type:
        m &= df["room_type"].str.lower() == room_type.lower()
    if preco_min is not None:
        m &= df["price"] >= float(preco_min)
    if preco_max is not None:
        m &= df["price"] <= float(preco_max)
    out = df[m].sort_values(ordem[ordenar_por], ascending=False)
    return out if limite is None else out.head(max(1, min(int(limite), 50)))


def word_lift(top_pct: float = 0.10, min_count: int = 150) -> list[dict]:
    """Palavras do nome mais sobre-representadas nos anúncios mais caros (top 10% de preço)."""
    df = clean()
    corte = df["price"].quantile(1 - top_pct)
    tok = df["nome"].str.lower().map(lambda s: set(re.findall(r"[a-z]{3,}", s)))
    caro = df["price"] >= corte
    from collections import Counter

    tot, top = Counter(), Counter()
    for words, c in zip(tok, caro):
        tot.update(words)
        if c:
            top.update(words)
    stop = {"the", "and", "with", "for", "near", "room", "apartment", "apt", "bedroom", "bed", "nyc", "new", "york", "in"}
    base_rate = caro.mean()
    rows = [{"palavra": w, "anuncios": n, "pct_caros": round(top[w] / n * 100, 1), "lift": round(float(top[w] / n / base_rate), 2)}
            for w, n in tot.items() if n >= min_count and w not in stop]
    return sorted(rows, key=lambda r: -r["lift"])[:12]
