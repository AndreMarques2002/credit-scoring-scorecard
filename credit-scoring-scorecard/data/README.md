# Dados

Este projeto usa o dataset **Give Me Some Credit** (Kaggle), arquivo `cs-training.csv`.

1. Acesse a competição: <https://www.kaggle.com/c/GiveMeSomeCredit> e aceite as regras.
2. Baixe o `cs-training.csv` e salve em `data/raw/cs-training.csv`.

Os dados **não são redistribuídos** neste repositório (verifique os termos de uso da competição).

## Dicionário resumido

| Coluna | Descrição |
|---|---|
| `SeriousDlqin2yrs` | Alvo: 1 = atraso de 90+ dias em 2 anos |
| `RevolvingUtilizationOfUnsecuredLines` | Utilização do limite rotativo |
| `age` | Idade do tomador |
| `NumberOfTime30-59DaysPastDueNotWorse` | Nº de atrasos de 30 a 59 dias |
| `DebtRatio` | Dívidas mensais / renda mensal |
| `MonthlyIncome` | Renda mensal |
| `NumberOfOpenCreditLinesAndLoans` | Nº de linhas de crédito e empréstimos abertos |
| `NumberOfTimes90DaysLate` | Nº de atrasos de 90+ dias |
| `NumberRealEstateLoansOrLines` | Nº de financiamentos imobiliários |
| `NumberOfTime60-89DaysPastDueNotWorse` | Nº de atrasos de 60 a 89 dias |
| `NumberOfDependents` | Nº de dependentes |

Anomalias conhecidas tratadas em `data.clean`: idades inválidas (< 18) e códigos 96/98 nas contagens de atraso.
