"""Resolve YouTube channel IDs for the initial Data Discovery pilot.

The script reads the local API key from .env and never prints it.
"""

import argparse
from pathlib import Path

import requests
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_KEY = dotenv_values(PROJECT_ROOT / ".env").get("YOUTUBE_API_KEY")

if not API_KEY:
    raise RuntimeError("Defina YOUTUBE_API_KEY no arquivo .env antes de executar.")

PILOT_QUERIES = [
    ("Natura Brasil", "beleza"),
    ("O Boticario", "beleza"),
    ("Sephora Brasil", "beleza"),
    ("Lojas Renner", "moda"),
    ("C&A Brasil", "moda"),
    ("Riachuelo", "moda"),
    ("Chevrolet Brasil", "automotivo"),
    ("Volkswagen do Brasil", "automotivo"),
    ("Fiat Brasil", "automotivo"),
    ("Youcom", "moda"),
    ("Hering Brasil", "moda"),
    ("Reserva Brasil", "moda"),
    ("Farm Rio", "moda"),
    ("Dafiti Brasil", "moda"),
    ("Zattini", "moda"),
    ("Hyundai Motor Brasil", "automotivo"),
    ("Toyota do Brasil", "automotivo"),
    ("Honda Automoveis Brasil", "automotivo"),
    ("Renault Brasil", "automotivo"),
    ("Nissan Brasil", "automotivo"),
    ("Jeep Brasil", "automotivo"),
    ("BYD Brasil", "automotivo"),
    ("Camila Coelho", "moda_creator"),
    ("Thassia Naves", "moda_creator"),
    ("Chata de Galocha", "moda_creator"),
    ("Niina Secrets", "moda_creator"),
    ("Camila Coutinho", "moda_creator"),
    ("Steal The Look", "moda_creator"),
    ("Alex Cursino", "moda_creator"),
    ("Mandy Candy", "moda_creator"),
    ("Acelerados", "automotivo_creator"),
    ("Auto Super", "automotivo_creator"),
    ("FlatOut Brasil", "automotivo_creator"),
    ("Carros com Camanzi", "automotivo_creator"),
    ("De Carona com Leandro", "automotivo_creator"),
    ("Canal Top Speed", "automotivo_creator"),
    ("Mecanica Online", "automotivo_creator"),
    ("ComprecarTV", "automotivo_creator"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--sector", help="Resolve somente consultas de um grupo, como automotivo_creator.")
args = parser.parse_args()

for query, sector in PILOT_QUERIES:
    if args.sector and sector != args.sector:
        continue
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "channel",
                "maxResults": 3,
                "key": API_KEY,
            },
            timeout=30,
        )
    except requests.RequestException:
        print(f"{query} | {sector} | NETWORK_ERROR")
        continue

    payload = response.json() if "application/json" in response.headers.get("content-type", "") else {}
    if not response.ok:
        message = payload.get("error", {}).get("message", "unknown API error")
        print(f"{query} | {sector} | API_ERROR {response.status_code}: {message}")
        continue

    for item in payload.get("items", []):
        print(
            " | ".join(
                [
                    query,
                    sector,
                    item["id"].get("channelId", ""),
                    item["snippet"].get("channelTitle", ""),
                ]
            )
        )
