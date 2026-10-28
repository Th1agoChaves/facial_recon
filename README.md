# Facial Recon

Projeto demonstrativo de reconhecimento facial usando **OpenCV** e **DeepFace**.

## Estrutura

```text
facial_recon/
├── autorizado/       #Fotos de referência de pessoas autorizadas
├── teste/            #Imagens a serem avaliadas
├── resultado/        #Imagens processadas
├── main.py
├── pyproject.toml
└── README.md
```

As pastas `autorizado/`, `teste/` e `resultado/` são criadas automaticamente na primeira execução. Adicione as fotos nas duas primeiras pastas antes de rodar o programa.

## Requisitos

- Python 3.10 ou superior
- `uv` (recomendado) ou `pip`

## Instalação com uv

Abra o terminal na pasta `facial_recon` e execute:

```powershell
uv sync
```

## Instalação com pip

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install opencv-python deepface
```

## Como usar

1. Coloque uma ou mais fotos nítidas de cada pessoa autorizada em `autorizado/`.
2. Coloque as imagens a avaliar em `teste/`.
3. Execute na raiz do projeto:

```powershell
uv run python main.py
```

Se estiver usando `pip`/ambiente virtual:

```powershell
python main.py
```

As imagens processadas serão gravadas em `resultado/` com o sufixo `_resultado`.

## Como a decisão é feita

O script detecta cada rosto na imagem de teste e usa `DeepFace.verify` para compará-lo com as imagens de referência, utilizando o modelo `Facenet512`. O DeepFace determina a decisão a partir do limiar associado ao modelo.

- **Verde:** `ACESSO LIBERADO` — pelo menos uma foto de referência foi verificada como correspondência.
- **Vermelho:** `ACESSO NEGADO` — nenhuma referência foi verificada como correspondência.
- **Sem rosto detectado:** a imagem é salva com a indicação `NENHUM ROSTO DETECTADO`.

## Observações importantes

- Na primeira execução, o DeepFace pode baixar os pesos do modelo; é necessária conexão com a internet.
- O desempenho depende da qualidade, iluminação, ângulo e resolução das fotos. Teste o sistema com diferentes imagens.
- Para um trabalho acadêmico, descreva o limiar como uma configuração do modelo, não como certeza absoluta de identidade.
- Este projeto é uma demonstração e não deve ser usado como único controle de acesso real. Sistemas reais precisam de testes de falsos aceites/rejeições, proteção das imagens, consentimento e mecanismos contra fotos impressas ou telas (anti-spoofing).
- O reconhecimento facial é probabilístico: não trate o resultado como prova de identidade.
