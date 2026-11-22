"""Testes sem rede."""
import pytest

from aluguelagent import core as C
from aluguelagent.agentcore import ScriptedLLM
from aluguelagent.tools import build_agent, t_bairros, t_investir


def test_clean_range():
    d = C.clean()
    assert d["price"].between(10, 1000).all()


def test_predict_validates_room_type():
    with pytest.raises(ValueError):
        C.predict({"bairro": "Harlem", "room_type": "Castle"})


def test_unknown_neighbourhood():
    with pytest.raises(ValueError):
        C.predict({"bairro": "Atlantis", "room_type": "Private room"})


def test_entire_home_costs_more_than_private_room():
    a = C.predict({"bairro": "Harlem", "room_type": "Entire home/apt"})["preco"]
    b = C.predict({"bairro": "Harlem", "room_type": "Private room"})["preco"]
    assert a > b


def test_manhattan_most_expensive_region():
    assert t_bairros()["tabela"][0]["bairro_group"] == "Manhattan"


def test_investment_respects_min_listings():
    assert all(r["anuncios"] >= 50 for r in t_investir()["ranking"])


def test_publish_needs_approval():
    r = build_agent(use_env=False).run("Publique a recomendação de investimento")
    assert r.pending and not r.results


def test_grounding():
    llm = ScriptedLLM([{"calls": [{"tool": "estatisticas_bairros", "args_json": "{}"}]},
                       {"answer": "A mediana é US$ 999.", "numbers_cited": [999]}])
    assert build_agent(llm=llm, use_env=False).run("x").status == "revisar"
