"""Ferramentas do agente de aluguel em Nova York. Cada uma devolve um dict com 'resumo'."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import core as C
from .agentcore import Agent, Tool, make_llm

REPORTS = C.ROOT / "reports"
TIPOS = {"inteiro": "Entire home/apt", "casa": "Entire home/apt", "apartamento inteiro": "Entire home/apt",
         "privado": "Private room", "quarto": "Private room", "compartilhado": "Shared room"}


def t_estimar(bairro: str, room_type: str = "Entire home/apt", minimo_noites: int = 2, disponibilidade_365: int = 100,
              numero_de_reviews: int = 10, nome: str = "") -> dict:
    p = C.predict({"bairro": bairro, "room_type": room_type, "minimo_noites": int(minimo_noites),
                   "disponibilidade_365": int(disponibilidade_365), "numero_de_reviews": int(numero_de_reviews), "nome": nome})
    return {**{k: round(v, 2) for k, v in p.items()}, "bairro": bairro, "room_type": room_type,
            "resumo": (f"Preço estimado para {room_type} em {bairro}: US$ {p['preco']:.2f} por noite "
                       f"(faixa p10–p90 de US$ {p['p10']:.2f} a US$ {p['p90']:.2f}).")}


def t_teste() -> dict:
    p = C.predict(C.TESTE)
    err = (p["preco"] - C.TESTE_REAL) / C.TESTE_REAL * 100
    return {"preco_previsto": round(p["preco"], 2), "preco_real": C.TESTE_REAL, "erro_pct": round(err, 1),
            "resumo": (f"Para o anúncio de teste do desafio (Skylit Midtown Castle), o modelo prevê US$ {p['preco']:.2f}; "
                       f"o preço real é US$ {C.TESTE_REAL:.2f}, um erro de {err:.1f}%.")}


def t_consultar(bairro_group: str | None = None, bairro: str | None = None, room_type: str | None = None, preco_min: float | None = None,
                preco_max: float | None = None, ordenar_por: str = "preco", limite: int = 10) -> dict:
    df = C.query(bairro_group, bairro, room_type, preco_min, preco_max, ordenar_por, limite)
    total = len(C.query(bairro_group, bairro, room_type, preco_min, preco_max, ordenar_por, None))
    items = [{"nome": r.nome, "bairro": r.bairro, "tipo": r.room_type, "preco": int(r.price), "reviews": int(r.numero_de_reviews)}
             for r in df.itertuples()]
    lista = "; ".join(f"{i['nome'][:40]} ({i['bairro']}, US$ {i['preco']})" for i in items[:5])
    return {"total_encontrado": total, "anuncios": items,
            "resumo": f"{total} anúncios atendem ao filtro. Primeiros por {ordenar_por}: {lista}." if total else "Nenhum anúncio atende ao filtro."}


def t_bairros(nivel: str = "bairro_group", room_type: str | None = None, min_anuncios: int = 30, limite: int = 10) -> dict:
    if nivel not in ("bairro_group", "bairro"):
        raise ValueError("nivel deve ser 'bairro_group' ou 'bairro'")
    df = C.clean()
    if room_type:
        df = df[df["room_type"] == room_type]
    g = df.groupby(nivel).agg(anuncios=("price", "size"), preco_mediano=("price", "median"),
                              disponibilidade_mediana=("disponibilidade_365", "median"), reviews_mes_medio=("reviews_por_mes", "mean"))
    g = g[g["anuncios"] >= int(min_anuncios)].sort_values("preco_mediano", ascending=False).head(int(limite)).round(2)
    rows = g.reset_index().to_dict("records")
    lista = "; ".join(f"{r[nivel]}: US$ {r['preco_mediano']:.0f} ({int(r['anuncios'])} anúncios)" for r in rows[:6])
    return {"tabela": rows, "resumo": f"Preço mediano por {'região' if nivel == 'bairro_group' else 'bairro'}" +
            (f" ({room_type})" if room_type else "") + f": {lista}."}


def t_noites_disponibilidade() -> dict:
    df = C.clean()
    rho_n = float(df["minimo_noites"].corr(df["price"], method="spearman"))
    rho_d = float(df["disponibilidade_365"].corr(df["price"], method="spearman"))
    faixas = pd.cut(df["minimo_noites"], [0, 1, 3, 7, 30, 10_000], labels=["1", "2-3", "4-7", "8-30", "31+"])
    por_noites = df.groupby(faixas, observed=True)["price"].median().round(1).to_dict()
    return {"spearman_minimo_noites": round(rho_n, 3), "spearman_disponibilidade": round(rho_d, 3),
            "preco_mediano_por_minimo_noites": {str(k): float(v) for k, v in por_noites.items()},
            "resumo": (f"A correlação de Spearman do preço com o mínimo de noites é {rho_n:.3f} e com a disponibilidade no ano é "
                       f"{rho_d:.3f}: os dois efeitos são fracos. Preço mediano por mínimo de noites: "
                       + "; ".join(f"{k}: US$ {v:.1f}" for k, v in por_noites.items()) + ".")}


def t_palavras() -> dict:
    rows = C.word_lift()
    lista = "; ".join(f"{r['palavra']} ({r['pct_caros']:.1f}% caros, {r['lift']:.2f}x)" for r in rows[:6])
    return {"palavras": rows, "resumo": f"Palavras do nome mais associadas aos 10% anúncios mais caros: {lista}."}


def t_explicar() -> dict:
    imp = C.importance()
    top = "; ".join(f"{r['grupo']} (+US$ {r['aumento_mae']:.2f})" for r in imp[:4])
    return {"importancia": imp, "resumo": f"O que mais pesa no preço (aumento do erro médio ao embaralhar): {top}."}


def t_modelos() -> dict:
    e = C.evaluate()
    txt = "; ".join(f"{r['modelo']}: MAE US$ {r['mae']:.2f}, RMSE US$ {r['rmse']:.2f}" for r in e["tabela"])
    return {"tabela": e["tabela"], "n_teste": e["n_teste"], "resumo": f"Holdout com {e['n_teste']} anúncios. {txt}."}


def t_investir(room_type: str = "Entire home/apt", min_anuncios: int = 50, limite: int = 5) -> dict:
    """Ranking por receita anual estimada = preço mediano x noites ocupadas estimadas.

    Ocupação estimada pelo modelo do Inside Airbnb: reviews/mês ÷ 50% de taxa de avaliação × 3 noites por estadia,
    limitada a 70% do ano. É uma aproximação, não um dado de ocupação."""
    df = C.clean()
    df = df[df["room_type"] == room_type].copy()
    df["noites_ano"] = np.minimum(df["reviews_por_mes"] / 0.5 * 3 * 12, 365 * 0.7)
    g = df.groupby(["bairro_group", "bairro"]).agg(anuncios=("price", "size"), preco_mediano=("price", "median"),
                                                   noites_ano=("noites_ano", "median"))
    g = g[g["anuncios"] >= int(min_anuncios)]
    g["noites_ano"] = g["noites_ano"].round(1)  # arredonda antes, para a receita mostrada bater com preço × noites
    g["receita_anual_estimada"] = (g["preco_mediano"] * g["noites_ano"]).round(0)
    g = g.sort_values("receita_anual_estimada", ascending=False).head(int(limite)).round(1).reset_index()
    rows = g.to_dict("records")
    lista = "; ".join(f"{r['bairro']} ({r['bairro_group']}): US$ {r['receita_anual_estimada']:,.0f}/ano" for r in rows).replace(",", ".")
    return {"ranking": rows, "metodo": "preço mediano x noites ocupadas estimadas pelas reviews (aproximação)",
            "resumo": f"Bairros com maior receita anual estimada para {room_type}: {lista}. A ocupação é estimada pelas reviews."}


def t_publicar(titulo: str = "Recomendação de investimento", conteudo: str = "") -> dict:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "recomendacao.md").write_text(f"# {titulo}\n\n{conteudo}\n", encoding="utf-8")
    return {"arquivo": "reports/recomendacao.md", "resumo": f"Recomendação '{titulo}' publicada em reports/recomendacao.md."}


TIPO = "'Entire home/apt' | 'Private room' | 'Shared room'"
TOOLS = [
    Tool("estimar_preco", "Estima a diária de um anúncio.", {"bairro": "texto, ex. 'Williamsburg'", "room_type": TIPO, "minimo_noites": "int",
                                                              "disponibilidade_365": "int", "numero_de_reviews": "int", "nome": "texto"}, t_estimar),
    Tool("prever_anuncio_teste", "Prevê o preço do anúncio de teste do desafio (Skylit Midtown Castle).", {}, t_teste),
    Tool("consultar_anuncios", "Lista anúncios por região, bairro, tipo e preço.",
         {"bairro_group": "Manhattan|Brooklyn|Queens|Bronx|Staten Island", "bairro": "texto", "room_type": TIPO, "preco_min": "número",
          "preco_max": "número", "ordenar_por": "preco|reviews|disponibilidade", "limite": "int"}, t_consultar),
    Tool("estatisticas_bairros", "Preço mediano, disponibilidade e reviews por região ou bairro.",
         {"nivel": "bairro_group|bairro", "room_type": TIPO, "min_anuncios": "int", "limite": "int"}, t_bairros),
    Tool("efeito_noites_disponibilidade", "Se mínimo de noites e disponibilidade interferem no preço.", {}, t_noites_disponibilidade),
    Tool("palavras_dos_caros", "Palavras do nome associadas aos anúncios mais caros.", {}, t_palavras),
    Tool("explicar_preco", "O que mais pesa no preço previsto.", {}, t_explicar),
    Tool("comparar_modelos", "Erro em holdout da mediana por bairro, Ridge e GBM.", {}, t_modelos),
    Tool("sugerir_investimento", "Onde comprar para alugar: receita anual estimada por bairro.",
         {"room_type": TIPO, "min_anuncios": "int", "limite": "int"}, t_investir),
    Tool("publicar_recomendacao", "Publica uma recomendação de investimento.", {"titulo": "texto", "conteudo": "texto"}, t_publicar, sensitive=True),
]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


def offline_router(question: str) -> list[dict]:
    q = _norm(question)
    tipo = next((v for k, v in TIPOS.items() if k in q), None)
    regiao = next((r for r in ("Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island") if r.lower() in q), None)
    bairros = C.load()["bairro"].unique()
    bairro = next((b for b in sorted(bairros, key=len, reverse=True) if _norm(b) in q and b not in ("Manhattan",)), None)
    if "publi" in q:
        return [{"tool": "publicar_recomendacao", "args": {"conteudo": question}}]
    if "skylit" in q or "anuncio de teste" in q:
        return [{"tool": "prever_anuncio_teste", "args": {}}]
    if any(w in q for w in ("investi", "comprar", "onde compr", "rentab")):
        return [{"tool": "sugerir_investimento", "args": {"room_type": tipo or "Entire home/apt"}}]
    if any(w in q for w in ("noites", "disponibilidade")):
        return [{"tool": "efeito_noites_disponibilidade", "args": {}}]
    if any(w in q for w in ("palavra", "nome")):
        return [{"tool": "palavras_dos_caros", "args": {}}]
    if any(w in q for w in ("compar", "mae", "rmse", "erro do modelo")):
        return [{"tool": "comparar_modelos", "args": {}}]
    if any(w in q for w in ("pesa", "importan", "explic", "influencia")):
        return [{"tool": "explicar_preco", "args": {}}]
    if any(w in q for w in ("quanto cobrar", "estim", "prev", "diaria")) and bairro:
        return [{"tool": "estimar_preco", "args": {"bairro": bairro, "room_type": tipo or "Entire home/apt"}}]
    if any(w in q for w in ("mediana", "media", "por bairro", "por regiao", "mais caro", "mais barato")):
        return [{"tool": "estatisticas_bairros", "args": {"nivel": "bairro" if "bairro" in q else "bairro_group", "room_type": tipo}}]
    if regiao or bairro or "anuncio" in q:
        pm = re.search(r"ate (?:us\$ ?)?(\d+)", q)
        return [{"tool": "consultar_anuncios", "args": {"bairro_group": regiao, "bairro": bairro, "room_type": tipo,
                                                         "preco_max": float(pm.group(1)) if pm else None}}]
    return []


def build_agent(llm=None, use_env: bool = True) -> Agent:
    if llm is None and use_env:
        llm = make_llm()
    return Agent(TOOLS, llm=llm, offline_router=offline_router)
