"""Configurações centrais do projeto."""
from dataclasses import dataclass

TARGET = "SeriousDlqin2yrs"  # 1 = inadimplente (90+ dias de atraso em 2 anos)

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Colunas de contagem de atraso em que 96 e 98 são códigos especiais, não quantidades reais.
PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

RANDOM_STATE = 42
TEST_SIZE = 0.30


@dataclass(frozen=True)
class ScoreScale:
    """Escala do score: base_score pontos equivalem a odds base_odds (bons:maus);
    a cada `pdo` pontos, as odds dobram."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0


@dataclass(frozen=True)
class BusinessAssumptions:
    """Premissas ILUSTRATIVAS para a simulação de política de corte.

    Valores em unidades relativas: cada bom pagador aprovado gera `profit_per_good`
    e cada mau pagador aprovado gera uma perda de `loss_per_bad`.
    Ajuste conforme a realidade do produto (margem, exposição, LGD).
    """

    profit_per_good: float = 1.0
    loss_per_bad: float = 5.0


# Limiares usuais de PSI (regra prática da indústria).
PSI_STABLE = 0.10
PSI_ALERT = 0.25
