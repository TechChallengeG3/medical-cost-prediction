"""

# Medical Cost Prediction

Este projeto tem como objetivo construir um modelo de regressão para prever o valor dos custos médicos individuais com base em características como idade, IMC, gênero, número de filhos, tabagismo e região.

## Estrutura do Projeto

```
medical-cost-prediction/
├── data/
│   ├── raw/              # Base de dados original (.csv)
│   └── processed/        # Base tratada
├── notebooks/
│   └── exploration.ipynb # Exploração dos dados
├── src/
│   ├── preprocessing.py  # Limpeza e preparação de dados
│   ├── model.py          # Treinamento e avaliação do modelo
│   └── utils.py          # Funções auxiliares
├── reports/
│   ├── figures/          # Gráficos
│   └── final_report.md   # Relatório final
├── requirements.txt      # Dependências
├── README.md             # Documentação do projeto
├── .gitignore            # Arquivos ignorados pelo Git
└── LICENSE               # Licença MIT
```

## Instalação

```bash
python -m venv venv
venv\Scripts\activate               # No Windows
pip install -r requirements.txt
```

## Execução

- Execute `notebooks/exploration.ipynb` para explorar os dados
- Rode os scripts da pasta `src/` para preparar os dados e treinar o modelo

## Equipe
