# Model Card: Scorecard de risco de crédito (PD)

## Finalidade
Estimar a probabilidade de inadimplência (atraso de 90+ dias em 2 anos) de tomadores de crédito pessoal e convertê-la em um score em pontos para apoiar decisões de aprovação.
**Projeto de portfólio / demonstração técnica. Não deve ser usado em decisões reais de crédito.**

## Dados
- Fonte: *Give Me Some Credit* (Kaggle). Cerca de 150 mil registros, alvo binário `SeriousDlqin2yrs`.
- Tratamentos: remoção de idade < 18; códigos 96/98 nas contagens de atraso tratados como ausentes.
- Split: estratificado 70/30 (sem data, então não há validação temporal).

## Modelo
- Binning monotônico WoE → seleção por IV (≥ 0,02) e correlação (≤ 0,80) → regressão logística (L2).
- Conversão para pontos: 600 pts = odds 50:1; PDO = 20.
- Challenger: XGBoost com early stopping (apenas comparação).

## Desempenho
*Preencher com os valores de `reports/results.md` após rodar `make train` nos dados reais:*

| Métrica (teste) | Scorecard | XGBoost |
|---|---|---|
| AUC | | |
| Gini | | |
| KS | | |
| Brier | | |

Também registrar: monotonicidade da taxa de maus entre faixas e PSI treino vs. teste.

## Uso previsto e fora de escopo
- Previsto: demonstração de modelagem, validação e monitoramento de risco de crédito.
- Fora de escopo: decisão automatizada real, precificação, limite de crédito, qualquer uso sem validação independente.

## Limitações
- Viés de seleção (apenas aprovados no passado têm desfecho observado).
- Sem validação out-of-time; sem teste de robustez a mudanças macroeconômicas.
- Premissas de lucro/perda ilustrativas.
- Variáveis demográficas (idade, dependentes) exigem avaliação jurídica e de equidade antes de qualquer uso real.

## Monitoramento
PSI do score e das variáveis a cada safra nova; gatilhos: 0,10 (atenção) e 0,25 (ação). Reavaliar também discriminação (KS/Gini) quando houver desfecho observado.

## Considerações éticas e regulatórias
Decisões automatizadas sobre crédito pedem transparência e possibilidade de revisão (por exemplo, o direito à revisão previsto na LGPD). A estrutura em pontos por faixa facilita justificar cada decisão. Uma análise formal de equidade e de conformidade é necessária antes de qualquer uso real.
