# Resultados da avaliação

**20 de 20 casos corretos.**

| Caso | Resultado | Obtido | Esperado |
|---|---|---|---|
| preço mediano de Manhattan | ok | 149.0 | 149.0 |
| quartos privados no Brooklyn | ok | 10131 | 10131 |
| Spearman preço × mínimo de noites | ok | 0.103 | 0.103 |
| filtro remove preço zero e acima de US$ 1.000 | ok | 250 | 250 |
| GBM erra menos que a mediana por bairro e tipo | ok | 44.22 | < 50.9 |
| GBM erra menos que Ridge | ok | 44.22 | < 47.52 |
| anúncio de teste dentro de 20% do preço real | ok | 243.31 | 225.0 |
| ranking de investimento consistente (>= 50 anúncios, receita = preço × noites) | ok | 5 | 5 |
| roteamento: Quanto cobrar por um apartamento inteiro em Williamsburg? | ok | ['estimar_preco'] | estimar_preco |
| roteamento: Qual o preço previsto para o anúncio Skylit Midtown Castle? | ok | ['prever_anuncio_teste'] | prever_anuncio_teste |
| roteamento: Onde é melhor comprar um imóvel para alugar? | ok | ['sugerir_investimento'] | sugerir_investimento |
| roteamento: O mínimo de noites e a disponibilidade interferem no preço? | ok | ['efeito_noites_disponibilidade'] | efeito_noites_disponibilidade |
| roteamento: Existe padrão no nome dos anúncios mais caros? | ok | ['palavras_dos_caros'] | palavras_dos_caros |
| roteamento: O que mais pesa no preço? | ok | ['explicar_preco'] | explicar_preco |
| roteamento: Compare os modelos pelo MAE | ok | ['comparar_modelos'] | comparar_modelos |
| roteamento: Qual o preço mediano por região? | ok | ['estatisticas_bairros'] | estatisticas_bairros |
| roteamento: Quartos privados no Brooklyn até 80 dólares | ok | ['consultar_anuncios'] | consultar_anuncios |
| roteamento: Publique a recomendação de investimento | ok | ['publicar_recomendacao'] | publicar_recomendacao |
| número inventado é barrado | ok | revisar | revisar |
| número correto passa | ok | ok | ok |

## Modelos (holdout de 20%, semente 42)

| Modelo | MAE (US$) | RMSE (US$) | Erro % mediano |
|---|---|---|---|
| mediana_bairro_tipo | 50.90 | 101.08 | 25.0 |
| ridge | 47.52 | 93.11 | 25.0 |
| gbm | 44.22 | 87.52 | 22.2 |
