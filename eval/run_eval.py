"""Avaliação contra gabarito. Roda sem GEMINI_API_KEY.

Respostas esperadas calculadas aqui direto da base bruta, de forma independente das ferramentas.
Uso: python eval/run_eval.py [--gemini]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aluguelagent import core as C  # noqa: E402
from aluguelagent.agentcore import ScriptedLLM  # noqa: E402
from aluguelagent.tools import build_agent  # noqa: E402

ROTEAMENTO = [
    ("Quanto cobrar por um apartamento inteiro em Williamsburg?", "estimar_preco"),
    ("Qual o preço previsto para o anúncio Skylit Midtown Castle?", "prever_anuncio_teste"),
    ("Onde é melhor comprar um imóvel para alugar?", "sugerir_investimento"),
    ("O mínimo de noites e a disponibilidade interferem no preço?", "efeito_noites_disponibilidade"),
    ("Existe padrão no nome dos anúncios mais caros?", "palavras_dos_caros"),
    ("O que mais pesa no preço?", "explicar_preco"),
    ("Compare os modelos pelo MAE", "comparar_modelos"),
    ("Qual o preço mediano por região?", "estatisticas_bairros"),
    ("Quartos privados no Brooklyn até 80 dólares", "consultar_anuncios"),
    ("Publique a recomendação de investimento", "publicar_recomendacao"),
]


def check(nome, ok, obtido, esperado):
    return {"caso": nome, "ok": bool(ok), "obtido": obtido, "esperado": esperado}


def main() -> int:
    raw = pd.read_csv(ROOT / "data" / "listings_nyc_2019.csv")
    limpo = raw[(raw["price"] >= 10) & (raw["price"] <= 1000)]
    ag = build_agent(use_env=False)
    T = ag.tools
    out = []

    med = float(limpo.loc[limpo["bairro_group"] == "Manhattan", "price"].median())
    got = next(r["preco_mediano"] for r in T["estatisticas_bairros"].run()["tabela"] if r["bairro_group"] == "Manhattan")
    out.append(check("preço mediano de Manhattan", got == med, got, med))
    n = int(((raw["bairro_group"] == "Brooklyn") & (raw["room_type"] == "Private room")).sum())
    got = T["consultar_anuncios"].run(bairro_group="Brooklyn", room_type="Private room")["total_encontrado"]
    out.append(check("quartos privados no Brooklyn", got == n, got, n))
    rho = round(float(limpo["minimo_noites"].corr(limpo["price"], method="spearman")), 3)
    got = T["efeito_noites_disponibilidade"].run()["spearman_minimo_noites"]
    out.append(check("Spearman preço × mínimo de noites", got == rho, got, rho))
    out.append(check("filtro remove preço zero e acima de US$ 1.000", len(raw) - len(C.clean()) == len(raw) - len(limpo),
                     len(raw) - len(C.clean()), len(raw) - len(limpo)))
    tab = {r["modelo"]: r for r in C.evaluate()["tabela"]}
    out.append(check("GBM erra menos que a mediana por bairro e tipo", tab["gbm"]["mae"] < tab["mediana_bairro_tipo"]["mae"],
                     tab["gbm"]["mae"], f"< {tab['mediana_bairro_tipo']['mae']}"))
    out.append(check("GBM erra menos que Ridge", tab["gbm"]["mae"] < tab["ridge"]["mae"], tab["gbm"]["mae"], f"< {tab['ridge']['mae']}"))
    teste = T["prever_anuncio_teste"].run()
    out.append(check("anúncio de teste dentro de 20% do preço real", abs(teste["erro_pct"]) <= 20, teste["preco_previsto"], C.TESTE_REAL))
    inv = T["sugerir_investimento"].run()["ranking"]
    ok = all(r["anuncios"] >= 50 and abs(r["receita_anual_estimada"] - r["preco_mediano"] * r["noites_ano"]) <= 1 for r in inv)
    out.append(check("ranking de investimento consistente (>= 50 anúncios, receita = preço × noites)", ok, len(inv), 5))

    usar_gemini = "--gemini" in sys.argv
    agent = build_agent(use_env=usar_gemini)
    if usar_gemini and agent.llm is None:
        print("--gemini pedido, mas GEMINI_API_KEY não está definida")
        return 2
    for pergunta, esperado in ROTEAMENTO:
        r = agent.run(pergunta)
        tools = [x["tool"] for x in r.calls]
        ok = tools[:1] == [esperado]
        ok = ok and (bool(r.pending) and not r.results if esperado == "publicar_recomendacao" else r.status == "ok")
        out.append(check(f"roteamento: {pergunta}", ok, tools, esperado))

    llm = ScriptedLLM([{"calls": [{"tool": "estatisticas_bairros", "args_json": "{}"}]},
                       {"answer": "Em Manhattan a mediana é US$ 210.", "numbers_cited": [210]}])
    out.append(check("número inventado é barrado", build_agent(llm=llm, use_env=False).run("x").status == "revisar", "revisar", "revisar"))
    llm = ScriptedLLM([{"calls": [{"tool": "estatisticas_bairros", "args_json": "{}"}]},
                       {"answer": f"Em Manhattan a mediana é US$ {med:.0f}.", "numbers_cited": [med]}])
    st = build_agent(llm=llm, use_env=False).run("x").status
    out.append(check("número correto passa", st == "ok", st, "ok"))

    acertos = sum(x["ok"] for x in out)
    suf = "_gemini" if usar_gemini else ""
    res = {"acertos": acertos, "total": len(out), "modelos": C.evaluate()["tabela"], "teste": teste, "importancia": C.importance(),
           "casos": out}
    (ROOT / "eval" / f"resultados{suf}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = [f"# Resultados da avaliação\n\n**{acertos} de {len(out)} casos corretos.**\n", "| Caso | Resultado | Obtido | Esperado |",
          "|---|---|---|---|"] + [f"| {x['caso']} | {'ok' if x['ok'] else 'FALHOU'} | {x['obtido']} | {x['esperado']} |" for x in out]
    md += ["\n## Modelos (holdout de 20%, semente 42)\n", "| Modelo | MAE (US$) | RMSE (US$) | Erro % mediano |", "|---|---|---|---|"]
    md += [f"| {r['modelo']} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['erro_pct_mediano']:.1f} |" for r in C.evaluate()["tabela"]]
    (ROOT / "eval" / f"resultados{suf}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{acertos}/{len(out)} casos corretos")
    for x in out:
        if not x["ok"]:
            print("FALHOU:", x)
    return 0 if acertos == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
