"""
garmin_login.py — autenticação inicial no Garmin Connect
=========================================================
Execute UMA vez para gerar ~/.garminconnect/garmin_tokens.json
Depois disso sync.py usa os tokens automaticamente.

Nota: se o MCP Garmin já funciona no Cowork, os tokens provavelmente
já estão em C:\\Users\\<você>\\.garminconnect\\garmin_tokens.json
e sync.py já deve funcionar sem precisar rodar este script.

Uso:
    python garmin_login.py
"""

import getpass
from pathlib import Path

from garminconnect import Garmin

TOKENS_DIR = Path.home() / ".garminconnect"

# ── Verifica se tokens já existem ───────────────────────────────────────────
tokens_file = TOKENS_DIR / "garmin_tokens.json"
if tokens_file.exists():
    print(f"Tokens já existem em: {tokens_file}")
    print("Tentando validar...")
    try:
        client = Garmin()
        client.login(str(TOKENS_DIR))
        name = client.get_full_name()
        print(f"✅ Login OK: {name}")
        print("sync.py já pode ser executado normalmente.")
    except Exception as e:
        print(f"⚠️  Tokens inválidos: {e}")
        print("Vamos fazer novo login...")
        tokens_file.unlink(missing_ok=True)
    else:
        exit(0)

# ── Login interativo ─────────────────────────────────────────────────────────
email    = input("E-mail Garmin Connect: ")
password = getpass.getpass("Senha: ")

print("Autenticando...")
try:
    client = Garmin(email, password)
    result = client.login()

    if result and result[0]:   # MFA necessário
        code = input("Código MFA (app/email): ").strip()
        client.resume_login(mfa_code=code)

    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    client.client.dump(str(TOKENS_DIR))
    print(f"✅ Tokens salvos em: {tokens_file}")
    print("Agora execute: python sync.py")

except Exception as e:
    msg = str(e)
    if "429" in msg:
        print(
            "\n❌ Rate limit do Garmin (429).\n"
            "Aguarde 30 min e tente novamente, ou use outra rede/VPN.\n\n"
            "ALTERNATIVA: Verifique se o arquivo já existe em:\n"
            f"  {tokens_file}\n"
            "Se o MCP Garmin funciona no Cowork, os tokens já devem estar lá."
        )
    else:
        raise
