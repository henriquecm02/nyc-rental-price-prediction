"""Dashboard e agente de preços de aluguel em Nova York. Rodar: streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import tema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aluguelagent import core as C  # noqa: E402
from aluguelagent.tools import build_agent, t_estimar, t_investir  # noqa: E402

st.set_page_config(page_title="Aluguel em NY · agente", page_icon="🗽", layout="wide")


@st.cache_resource
def get_agent():
    C.evaluate()
    C.model()
    return build_agent()


df = C.clean()
agent = get_agent()
st.session_state.setdefault("pending", [])
st.session_state.setdefault("log", [])

n_total = f"{len(C.load()):,}".replace(",", ".")
tema.aplicar_tema(
    "Agente de precificação: aluguel por temporada em Nova York",
    f"{n_total} anúncios de 2019 (Airbnb). Estima a diária, compara bairros e sugere onde investir. "
    "Os números vêm do pandas e do scikit-learn; o modelo de linguagem só interpreta.",
    kicker="Agente de IA · precificação",
    chips=["Gemini", "Gradient boosting", "Dados reais de 2019"])
if agent.llm is None:
    st.info("Modo offline: sem GEMINI_API_KEY, o agente escolhe ferramentas por palavras-chave.", icon="🔌")

tab_dash, tab_preco, tab_agent, tab_eval, tab_aprov = st.tabs(["Dashboard", "Estimar diária", "Agente", "Avaliação", "Aprovações"])

with tab_dash:
    tab = {r["modelo"]: r for r in C.evaluate()["tabela"]}
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anúncios analisados", f"{len(df):,}".replace(",", "."), help="Preço entre US$ 10 e US$ 1.000")
    k2.metric("Diária mediana", f"US$ {df['price'].median():.0f}")
    k3.metric("Erro médio do modelo (MAE)", f"US$ {tab['gbm']['mae']:.2f}",
              delta=f"{tab['gbm']['mae'] - tab['mediana_bairro_tipo']['mae']:+.2f} vs. mediana do bairro", delta_color="inverse")
    k4.metric("Anúncio de teste", f"US$ {C.predict(C.TESTE)['preco']:.0f}", help="Skylit Midtown Castle; preço real US$ 225")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Mapa de preços")
        amostra = df.sample(6000, random_state=0)
        fig = px.scatter(amostra, x="longitude", y="latitude", color="price", color_continuous_scale=tema.sequencial(),
                         range_color=(40, 400), opacity=0.6, hover_name="nome", labels={"price": "US$/noite"})
        fig.update_traces(marker={"size": 4})
        fig.update_layout(height=460, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis_visible=False, yaxis_visible=False)
        fig.update_yaxes(scaleanchor="x", scaleratio=1.32)
        st.plotly_chart(tema.estilo_plotly(fig), use_container_width=True)
    with c2:
        st.subheader("Diária por região e tipo")
        fig2 = px.box(df, x="bairro_group", y="price", color="room_type", points=False, log_y=True,
                      color_discrete_sequence=tema.cores(3), labels={"price": "US$/noite (log)", "bairro_group": ""})
        fig2.update_layout(height=460, margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": -0.15, "title": ""})
        st.plotly_chart(tema.estilo_plotly(fig2), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Onde investir (apartamento inteiro)")
        inv = pd.DataFrame(t_investir()["ranking"])
        st.dataframe(inv, hide_index=True, use_container_width=True)
        st.caption("Receita anual estimada = diária mediana × noites ocupadas estimadas pelas reviews (aproximação).")
    with c4:
        st.subheader("O que pesa no preço")
        imp = pd.DataFrame(C.importance()).sort_values("aumento_mae")
        fig3 = px.bar(imp, x="aumento_mae", y="grupo", orientation="h", color_discrete_sequence=[tema.cor(0)],
                      labels={"aumento_mae": "aumento do erro médio (US$) ao embaralhar", "grupo": ""})
        fig3.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        st.plotly_chart(tema.estilo_plotly(fig3), use_container_width=True)

with tab_preco:
    a, b, c = st.columns(3)
    bairro = a.selectbox("Bairro", sorted(C.load()["bairro"].unique()), index=sorted(C.load()["bairro"].unique()).index("Williamsburg"))
    tipo = a.selectbox("Tipo", ["Entire home/apt", "Private room", "Shared room"])
    noites = b.number_input("Mínimo de noites", 1, 365, 2)
    disp = b.number_input("Dias disponíveis no ano", 0, 365, 120)
    reviews = c.number_input("Número de reviews", 0, 600, 20)
    nome = c.text_input("Título do anúncio", "Sunny loft with view")
    if st.button("Estimar", type="primary"):
        st.success(t_estimar(bairro, tipo, noites, disp, reviews, nome)["resumo"])

with tab_agent:
    ex = st.selectbox("Exemplos", ["", "Quanto cobrar por um apartamento inteiro em Williamsburg?", "Onde é melhor comprar um imóvel para alugar?",
                                   "O mínimo de noites e a disponibilidade interferem no preço?", "Existe padrão no nome dos anúncios mais caros?"],
                      placeholder="Escolha um exemplo")
    q = st.text_input("Pergunta", value=ex)
    if st.button("Perguntar", type="primary") and q.strip():
        r = agent.run(q)
        st.session_state["log"].append(r)
        st.session_state["pending"].extend(r.pending)
    for r in reversed(st.session_state["log"][-5:]):
        with st.container(border=True):
            st.markdown(f"**{r.question}**")
            st.write(r.answer)
            badge = {"ok": "✅ números conferidos", "revisar": "⚠️ revisar: número sem fonte", "erro": "❌ erro"}[r.status]
            st.caption(f"{badge} · modo {r.mode} · ferramentas: {', '.join(c['tool'] for c in r.calls) or 'nenhuma'}")
            with st.expander("Rastro"):
                st.json({"calls": r.calls, "results": r.results, "errors": r.errors}, expanded=False)

with tab_eval:
    p = ROOT / "eval" / "resultados.json"
    if p.exists():
        res = json.loads(p.read_text(encoding="utf-8"))
        st.metric("Casos corretos", f"{res['acertos']} de {res['total']}")
        st.dataframe(pd.DataFrame(res["casos"]).astype(str), hide_index=True, use_container_width=True)

with tab_aprov:
    if not st.session_state["pending"]:
        st.write("Nenhuma ação pendente. Peça ao agente para publicar a recomendação de investimento.")
    for i, pa in enumerate(list(st.session_state["pending"])):
        with st.container(border=True):
            st.markdown(f"**{pa.tool}** · `{pa.args}`")
            x, y = st.columns(2)
            if x.button("Aprovar e executar", key=f"ok{i}", type="primary"):
                st.success(agent.approve(pa)["resumo"])
                st.session_state["pending"].remove(pa)
            if y.button("Recusar", key=f"no{i}"):
                st.session_state["pending"].remove(pa)
                st.rerun()
