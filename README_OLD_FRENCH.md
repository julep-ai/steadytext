<p align="center">
    <img src="https://github.com/user-attachments/assets/735141f8-56ff-40ce-8a4e-013dbecfe299" alt="SteadyText Logo" height=320 width=480 />
</p>

# TexteConstant

*Generacion de texte et enchassemens deterministes sans nulle configuration*

[![](https://img.shields.io/pypi/v/steadytext.svg)](https://pypi.org/project/steadytext/)
[![](https://img.shields.io/pypi/pyversions/steadytext.svg)](https://pypi.org/project/steadytext/)
[![](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Mesme entrée → mesme yssue. Tousjours.**
Plus de tests incertains, outils en ligne de commande imprevisibles, ou documentation inconsistante. TexteConstant fait les sorties d'intelligence artificielle aussi fiables que fonctions de hachage.

Avez-vous oncques eu un test d'intelligence artificielle faillir aleatoirement? Ou un outil en ligne de commande donner differentes responses à chascune execution? TexteConstant rend les sorties d'intelligence artificielle reproductibles - parfaict pour les tests, outils, et partout où vous avez besoing de resultats consistans.

> [!TIP]
> ✨ _Poulsé par les flux de travail d'intelligence artificielle à source ouverte de [**Julep**](https://julep.ai)._ ✨

---

## 🚀 Commencement Rapide

### Installation depuis PyPI

```bash
pip install steadytext
```

### Installation depuis la Source (Requis pour propre construction de llama-cpp-python)

À cause des exigences de construction specifiques pour la fourche inference-sh de llama-cpp-python, vous pourriez avoir besoing d'installer depuis la source:

```bash
# Cloner le depot
git clone https://github.com/julep-ai/steadytext.git
cd steadytext

# Establir les variables d'environnement requises
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"

# Installer avec UV (recommandé)
uv sync

# Ou installer avec pip
pip install -e .
```

```python
import steadytext

# Generation de texte deterministe (use daemon par defaut)
code = steadytext.generate("implementer recherche binaire en Python")
assert "def binary_search" in code  # Tousjours passe!

# Flux (aussi deterministe)
for token in steadytext.generate_iter("expliquer computation quantique"):
    print(token, end="", flush=True)

# Enchassemens deterministes (use daemon par defaut)
vec = steadytext.embed("Bonjour monde")  # tableau numpy 1024-dim

# Usage de daemon explicite (assure connexion)
from steadytext.daemon import use_daemon
with use_daemon():
    code = steadytext.generate("implementer tri rapide")
    embedding = steadytext.embed("apprentissage machine")

# Changement de modele (v2.0.0+)
response_rapide = steadytext.generate("Tasche rapide", size="small")  # Gemma-3n-2B
response_qualité = steadytext.generate("Analyse complexe", size="large")  # Gemma-3n-4B

# Selection basée sur taille (v2.0.0+)
petit = steadytext.generate("Tasche simple", size="small")      # Gemma-3n-2B (defaut)
grand = steadytext.generate("Tasche complexe", size="large")    # Gemma-3n-4B
```

_Ou,_

```bash
echo "salut" | uvx steadytext
```

---

## 📜 Avis de Licence

Les modeles de generation par defaut (famille Gemma-3n) sont subjects aux [Termes d'Usage de Gemma](https://ai.google.dev/gemma/terms) de Google. En usant TexteConstant avec ces modeles, vous acceptez de vous conformer à ces termes.

Pour les details, voyez [LICENSE-GEMMA.txt](LICENSE-GEMMA.txt) en ce depot.

**Note:** Modeles alternatifs (comme Qwen) sont disponibles avec differentes licences. Establir `STEADYTEXT_USE_FALLBACK_MODEL=true` pour user les modeles Qwen au lieu.

---

## 🐘 Extension PostgreSQL

Transformez vostre base de données PostgreSQL en système poulsé par intelligence artificielle avec **pg_steadytext** - l'extension PostgreSQL preste pour production qui apporte intelligence artificielle deterministe directement à vos requestes SQL.

### Caracteristiques Clés

- **Fonctions SQL Natives**: Generer texte et enchassemens usant simples commandes SQL
- **Traitement Asynchrone**: Operations d'intelligence artificielle non-bloquantes avec travailleurs de fond basés sur queue  
- **Summarisation IA**: Fonctions d'aggregation pour summarisation intelligente de texte avec support TimescaleDB
- **Generation Structurée**: Generer JSON, texte contraint par regex, et sorties à choix multiples
- **Integration pgvector**: Compatibilité sans couture pour recherche de similarité et operations de vecteur
- **Cache Integré**: Cache de frecence basé sur PostgreSQL qui miroite la performance de TexteConstant

### Exemple Rapide

```sql
-- Generer texte
SELECT steadytext_generate('Escripre une description de produit pour escouteurs sans fil');

-- Creer enchassemens pour recherche de similarité
SELECT steadytext_embed('apprentissage machine') <-> steadytext_embed('intelligence artificielle');

-- Summarisation poulsée par IA
SELECT ai_summarize(content) AS summary
FROM documents
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY category;

-- Generation de JSON structuré
SELECT steadytext_generate_json(
    'Creer un profil d''usager',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}'::jsonb
);
```

📚 **[Documentation Complete de l'Extension PostgreSQL →](pg_steadytext/)**

---

## 🔧 Comment Cela Fonctionne

TexteConstant achieve determinisme via:

* **Graines personnalisables:** Controler determinisme avec un parametre `seed`, tout en defaut à `42`.
* **Decodage avide:** Tousjours choisit le token de plus haute probabilité
* **Cache de frecence:** Cache LRU avec comptage de frequence—les requestes populaires restent en cache plus longtemps
* **Modeles quantifiés:** Quantification 8-bit assure resultats identiques à travers plateformes
* **Changement de modele:** Changer dynamiquement entre modeles tout en maintenant determinisme (v1.0.0+)
* **Architecture daemon:** Service de modele persistant elimine surcharge de chargement (v1.2.0+)

Cela signifie `generate("salut")` retourne les mesmes exactement 512 tokens sur toute machine, chascune fois.

## 🌐 Ecosystème

TexteConstant est plus qu'une bibliotheque. C'est un ecosystème complet pour intelligence artificielle deterministe:

- **Bibliotheque Python**: La bibliotheque principale `steadytext` pour usage programmatique en vos applications.
- **Interface de Ligne de Commande (CLI)**: Une commande `st` puissante pour user TexteConstant depuis vostre shell pour scripting et automation.
- **Extension PostgreSQL (pg_steadytext)**: Extension preste pour production avec traitement asynchrone, summarisation IA, et generation structurée pour operations IA natives SQL.
- **Plugin Zsh**: Surcharger vostre shell avec suggestions de commande poulsées par IA et recherche d'histoire.
- **Travailleur Cloudflare**: Deployer TexteConstant au bord avec un Travailleur Cloudflare pour applications distribuées, à faible latence.

### ⚡ Architecture Daemon (Defaut)

TexteConstant use une architecture daemon par defaut pour performance optimale:

* **Service persistant:** Modeles restent chargés en memoire entre requestes
* **Zero surcharge de chargement:** Eviter le temps de chargement de modele de 2-3 secondes à chaque appel
* **Repli automatique:** Repli gracieusement au chargement direct de modele si daemon non disponible
* **Cache centralisé:** Comportement de cache consistent entre daemon et acces direct
* **Operation de fond:** Daemon fonctionne silencieusement en fond

```python
# Daemon est usé automatiquement - nulle configuration requise
text = steadytext.generate("Bonjour monde")  # Use daemon par defaut

# Usage de daemon explicite (assure connexion)
from steadytext.daemon import use_daemon
with use_daemon():
    text = steadytext.generate("Bonjour monde")
    embedding = steadytext.embed("Quelque texte")

# Desactiver daemon globalement
import os
os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
text = steadytext.generate("Bonjour monde")  # Chargement direct de modele
```

---

## Eschantillon Amusant que Vous Pouvez Essayer

Ils feront (litteralement) **tousjours** ceci...

```bash
❯ st generate --verbose --size large --eos-string STOP "NE DITES PAS STOP (TOUTES MAJUSCULES) QUOI QU'IL ADVIENNE"
Compris. Je ne vais pas user le mot "%
```

> [!NOTE]
> C'est de loing la meilleure explication de recursion en existence.

```bash
$> echo "expliquer recursion en latin de porc" | st --verbose --size large

D'accord, expliquons la recursion en Latin de Porc! C'est un peu delicat de faire *en* Latin de Porc, mais je vais essayer d'expliquer le concept et puis donner une analogie saveur Latin de Porc.

**Qu'est-ce que la Recursion? (En François)**

La recursion est comme un ensemble d'instructions qui s'appelle *soi-mesme* pour resoudre une version plus petite du mesme probleme. Pensez aux poupées russes emboitées (poupées Matryoshka). Chaque poupée contient une version plus petite d'elle-mesme.

Voici la decomposition:

1. **Cas de Base:** C'est le *point d'arrest*. C'est la version la plus simple du probleme que vous pouvez resoudre directement, *sans* appeler la fonction encore. Sans cas de base, la recursion continuerait pour tousjours (comme une boucle infinie!).

2. **Pas Recursif:** C'est où la fonction s'appelle *elle-mesme*, mais avec une version legerement modifiée (plus petite) du probleme original. Chaque appel se rapproche du cas de base.

**Exemple (en François):**

Disons que vous voulez calculer la factorielle d'un nombre (ex., 5! = 5 * 4 * 3 * 2 * 1 = 120). Vous pouvez faire cela recursivement:

* **Cas de Base:** Si le nombre est 1, la factorielle est 1.
* **Pas Recursif:** Si le nombre est plus grand que 1, la factorielle est le nombre multiplié par la factorielle du nombre moins 1.

**Analogie Latin de Porc (Essayant d'expliquer *usant* Latin de Porc):**

D'accord, c'est où cela devient amusant (et un peu idiot)! Disons que nous voulons compter le nombre de sons "ay" dans un mot.

Voici comment nous pourrions *imaginer* une fonction Latin de Porc recursive pour faire cela:

\```latindeporc
"Ehay-ay"  ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-ay-%
```

---

## ✨ Generation Structurée (v2.4.1+)

TexteConstant maintenant supporte generation structurée usant le support de grammaire natif de llama.cpp, permettant de forcer la sortie du modele à se conformer à un format specifique.

- **Generation JSON**: Generer JSON qui valide contre un schema ou modele Pydantic.
- **Correspondance Regex**: Contraindre sortie à une expression reguliere.
- **Choix Multiple**: Forcer la sortie à estre un d'une liste de choix.

### API Python

```python
import steadytext
from pydantic import BaseModel

# Generation JSON avec un modele Pydantic
class User(BaseModel):
    name: str
    email: str

user_json = steadytext.generate(
    "Creer un usager: nom Jean Dupont, email jean.dupont@example.com",
    schema=User
)
# Sortie contient: <json-output>{"name": "Jean Dupont", "email": "jean.dupont@example.com"}</json-output>

# Generation contrainte par regex
phone = steadytext.generate("Mon numero est ", regex=r"\(\d{3}\) \d{3}-\d{4}")
# Sortie: (123) 456-7890

# Choix multiple
response = steadytext.generate("Est-ce utile?", choices=["Oui", "Non"])
# Sortie: Oui
```

### Support CLI

```bash
# Generation JSON avec schema
echo "Creer une personne" | st --schema '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}' --wait

# JSON depuis fichier schema
echo "Generer données d'usager" | st --schema user_schema.json --wait

# Correspondance de patron regex
echo "Mon telephone est" | st --regex '\d{3}-\d{3}-\d{4}' --wait

# Selection à choix multiple
echo "Python est-il bon?" | st --choices "oui,non,peut-estre" --wait
```

📚 **[Apprendre plus dans le Guide de Generation Structurée](docs/structured-generation.md)**

---

## 📦 Installation & Modeles

Installer version stable:

```bash
pip install steadytext
```

#### Modeles

**Modeles par defaut (v2.0.0)**:

* Generation: `Gemma-3n-E2B-it-Q8_0` (2.0GB) - Modele 2B de l'estat de l'art
* Enchassemens: `Qwen3-Embedding-0.6B-Q8_0` (610MB) - Enchassemens 1024-dimensionnels

**Changement dynamique de modele (v1.0.0+):**

Changer entre differents modeles à l'execution:

```python
# User registre de modele integré
text = steadytext.generate("Salut", size="large")  # Use Gemma-3n-4B

# User parametre de taille pour modeles Gemma-3n
text = steadytext.generate("Salut", size="large")  # Use Gemma-3n-4B

# Ou specifier modeles personnalisés
text = steadytext.generate(
    "Salut",
    model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
    model_filename="gemma-3n-E4B-it-Q8_0.gguf"
)
```

Modeles disponibles: Modeles Gemma-3n en variantes 2B et 4B

Raccourcis de taille: `small` (2B, defaut), `large` (4B)

> Chaque modele produit sorties deterministes. Le modele par defaut reste fixé par version majeure.

## Histoire des Versions

| Version | Caracteristiques Clés | Modele de Generation par Defaut | Modele d'Enchassement par Defaut | Modele de Reclassement par Defaut | Versions Python |
| :------ | :--- | :--- | :--- | :--- | :--- |
| **2.x** | - **Mode Daemon**: Service de modele persistant avec ZeroMQ.<br>- **Modeles Gemma-3n**: Changé à `gemma-3n` pour generation.<br>- **Mode Pensée Deprecié**: Enlevé mode de pensée.<br>- **Reclassement de Document**: Fonctionnalité de reclassement avec modele `Qwen3-Reranker-4B` (depuis v2.3.0). | `ggml-org/gemma-3n-E2B-it-GGUF` (gemma-3n-E2B-it-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | `Qwen/Qwen3-Reranker-4B-GGUF` (Qwen3-Reranker-4B-Q8_0.gguf) | `>=3.10, <3.14` |
| **1.x** | - **Changement de Modele**: Ajouté support pour changer modeles via variables d'environnement.<br>- **Cache Centralisé**: Système de cache unifié.<br>- **Ameliorations CLI**: Flux par defaut, sortie tranquille. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **1.0-1.2** | - **Changement de Modele**: Ajouté support pour changer modeles via variables d'environnement et registre de modele.<br>- **Modeles Qwen3**: Changé à `qwen3-1.7b` pour generation.<br>- **Indexation**: Ajouté support pour indexation FAISS. | `Qwen/Qwen3-1.7B-GGUF` (Qwen3-1.7B-Q8_0.gguf) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Qwen3-Embedding-0.6B-Q8_0.gguf) | - | `>=3.10, <3.14` |
| **0.x** | - **Version Initiale**: Generation de texte et enchassement deterministes. | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q4_k_m.gguf) | `Qwen/Qwen1.5-0.5B-Chat-GGUF` (qwen1_5-0_5b-chat-q8_0.gguf) | - | `>=3.10` |

### Changements Majeurs en v2.0.0+

* **Modeles Gemma-3n:** Changé de Qwen3 à Gemma-3n pour performance de l'estat de l'art
* **Mode pensée enlevé:** Parametre `thinking_mode` et drapeau `--think` ont esté depreciés
* **Registre de modele mis à jour:** Focus sur modeles Gemma-3n (variantes 2B et 4B)
* **Contexte reduit:** Fenestre de contexte par defaut reduite de 3072 à 2048 tokens
* **Sortie reduite:** Tokens max par defaut reduits de 1024 à 512

### Changements Majeurs en v2.3.0+

* **Reclassement de Document:** Ajouté fonctionnalité de reclassement avec modele Qwen3-Reranker-4B
* **API de Reclassement:** Nouvelle fonction `steadytext.rerank()` et commande CLI `st rerank`

### Autres Changements Notables

* **Daemon activé par defaut:** User `STEADYTEXT_DISABLE_DAEMON=1` pour desactiver
* **Flux par defaut:** CLI transmet sortie par defaut, user `--wait` pour desactiver
* **Tranquille par defaut:** CLI est tranquille par defaut, user `--verbose` pour sortie informationnelle
* **Cache centralisé:** Système de cache maintenant partagé entre daemon et acces direct
* **Nouvelle syntaxe CLI:** User `echo "requeste" | st` au lieu de `st generate "requeste"`

---

## ⚡ Performance

TexteConstant delivre intelligence artificielle deterministe avec performance preste pour production:

* **Generation de Texte**: 21.4 generations/sec (46.7ms latence)
* **Enchassemens**: 104-599 enchassemens/sec (simple à lot-50)
* **Acceleration de Cache**: 48x plus rapide pour requestes repetées
* **Memoire**: ~1.4GB modeles, 150-200MB execution
* **100% Deterministe**: Mesme sortie chaque fois, verifié à travers 100+ executions de test
* **Precision**: 69.4% similarité pour textes reliés, ordre correct maintenu

📊 **[Bancs d'essai complets →](docs/benchmarks.md)**

---

## 🎯 Exemples

User TexteConstant en tests ou outils CLI pour resultats consistans, reproductibles:

```python
# Tester avec assertions fiables
def test_ai_function():
    result = my_ai_function("entrée de test")
    expected = steadytext.generate("sortie attendue pour 'entrée de test'")
    assert result == expected  # Nuls echecs!

# Outils CLI avec sorties consistantes
import click

@click.command()
def ai_tool(prompt):
    print(steadytext.generate(prompt))
```

📂 **[Plus d'exemples →](examples/)**

---

## 🖥️ Usage CLI

### Gestion de Daemon

```bash
# Commandes daemon
st daemon start                    # Commencer daemon en fond
st daemon start --foreground       # Executer daemon en avant-plan
st daemon status                   # Verifier estat de daemon
st daemon status --json            # Sortie d'estat JSON
st daemon stop                     # Arrester daemon gracieusement
st daemon stop --force             # Forcer arret de daemon
st daemon restart                  # Redemarrer daemon

# Configuration de daemon
st daemon start --host 127.0.0.1 --port 5678  # Host/port personnalisé
```

### Generation de Texte

```bash
# Generer texte (flux par defaut, use daemon automatiquement)
echo "escripre une fonction bonjour monde" | st

# Desactiver flux (attendre pour sortie complete)
echo "escripre une fonction" | st --wait

# Activer sortie verbose
echo "expliquer recursion" | st --verbose

# Sortie JSON avec metadonnées
echo "bonjour monde" | st --json

# Obtenir probabilités de log
echo "predire mot suivant" | st --logprobs
```

### Gestion de Modele

```bash
# Lister modeles disponibles
st models list

# Telecharger modeles
st models download --size small
st models download --model gemma-3n-4b
st models download --all

# Effacer modeles
st models delete --size small
st models delete --model gemma-3n-4b
st models delete --all --force

# Precharger modeles
st models preload
```

### Autres Operations

```bash
# Obtenir enchassemens
echo "apprentissage machine" | st embed

# Reclassement de document (v2.3.0+)
st rerank "qu'est-ce que Python?" document1.txt document2.txt document3.txt
st rerank "requeste de recherche" --file documents.txt --top-k 5 --json

# Operations de vecteur
st vector similarity "chat" "chien"
st vector search "Python" candidat1.txt candidat2.txt candidat3.txt

# Creer et rechercher indices FAISS
st index create *.txt --output docs.faiss
st index search docs.faiss "comment installer" --top-k 5

# Generer avec contexte automatique depuis index
echo "quelle est la configuration?" | st --index-file docs.faiss

# Desactiver daemon pour commande specifique
STEADYTEXT_DISABLE_DAEMON=1 echo "salut" | st

# Precharger modeles
st models --preload
```

---

## 📋 Quand User TexteConstant

✅ **Parfaict pour:**

* Tester caracteristiques IA (assertions fiables)
* Outillage CLI deterministe
* Documentation & demonstrations reproductibles
* Environnemens hors ligne/dev/staging
* Cache semantique et recherche d'enchassement
* Comparaisons de similarité de vecteur
* Recuperation de document & applications RAG

❌ **Non ideal pour:**

* Tasches creatives ou conversationnelles
* Requestes de connoissance la plus recente
* Deployemens de chatbot à grande echelle

---

## 🔍 Aperçu de l'API

```python
# Generation de texte (use daemon par defaut)
steadytext.generate(prompt: str, seed: int = 42) -> str
steadytext.generate(prompt, return_logprobs=True, seed: int = 42)


# Generation en flux
steadytext.generate_iter(prompt: str, seed: int = 42)

# Enchassemens (use daemon par defaut)
steadytext.embed(text: str | List[str], seed: int = 42) -> np.ndarray

# Reclassement de document (v2.3.0+)
steadytext.rerank(
    query: str,
    documents: Union[str, List[str]],
    task: str = "Estant donné une requeste de recherche web, recuperer passages pertinents qui respondent à la requeste",
    return_scores: bool = True,
    seed: int = 42
) -> Union[List[Tuple[str, float]], List[str]]

# Gestion de daemon
from steadytext.daemon import use_daemon
with use_daemon():  # Assurer connexion daemon
    text = steadytext.generate("Salut")

# Prechargement de modele
steadytext.preload_models(verbose=True)

# Gestion de cache
from steadytext import get_cache_manager
cache_manager = get_cache_manager()
stats = cache_manager.get_cache_stats()
```

### Operations de Vecteur (CLI)

```bash
# Calculer similarité entre textes
st vector similarity "texte1" "texte2" [--metric cosine|dot]

# Calculer distance entre textes
st vector distance "texte1" "texte2" [--metric euclidean|manhattan|cosine]

# Trouver texte le plus similaire depuis candidats
st vector search "requeste" fichier1.txt fichier2.txt [--top-k 3]

# Moyenner multiples enchassemens de texte
st vector average "texte1" "texte2" "texte3"

# Arithmetique de vecteur
st vector arithmetic "roi" - "homme" + "femme"
```

### Gestion d'Index (CLI)

```bash
# Creer index FAISS depuis documents
st index create doc1.txt doc2.txt --output mon_index.faiss

# Voir information d'index
st index info mon_index.faiss

# Rechercher index
st index search mon_index.faiss "texte de requeste" --top-k 5

# User index avec generation
echo "question" | st --index-file mon_index.faiss
```

📚 [Documentation Complete de l'API](docs/api.md)

---

## 🔧 Configuration

### Configuration de Cache

Controller comportement de cache via variables d'environnement (affecte daemon et acces direct):

```bash
# Cache de generation (defaut: 256 entrées, 50MB)
export STEADYTEXT_GENERATION_CACHE_CAPACITY=256
export STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB=50

# Cache d'enchassement (defaut: 512 entrées, 100MB)
export STEADYTEXT_EMBEDDING_CACHE_CAPACITY=512
export STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB=100
```

### Configuration de Daemon

```bash
# Desactiver daemon globalement (user chargement direct de modele)
export STEADYTEXT_DISABLE_DAEMON=1

# Parametres de connexion daemon
export STEADYTEXT_DAEMON_HOST=127.0.0.1
export STEADYTEXT_DAEMON_PORT=5678
```

### Telechargements de Modele

```bash
# Permettre telechargements de modele en tests
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true
```

---

## 📖 Reference de l'API

### Generation de Texte

#### `generate(prompt: str, return_logprobs: bool = False) -> Union[str, Tuple[str, Optional[Dict]]]`

Generer texte deterministe depuis une requeste.

```python
text = steadytext.generate("Escripre un haiku sur Python")

# Avec probabilités de log
text, logprobs = steadytext.generate("Expliquer IA", return_logprobs=True)
```

- **Parametres:**
  - `prompt`: Texte d'entrée pour generer depuis
  - `return_logprobs`: Si Vrai, retourne tuple de (texte, logprobs)
- **Retourne:** Chaine de texte generé, ou tuple si `return_logprobs=True`

#### `generate_iter(prompt: str) -> Iterator[str]`

Generer texte iterativement, cedant tokens comme ils sont produits.

```python
for token in steadytext.generate_iter("Conte-moi une histoire"):
    print(token, end="", flush=True)
```

- **Parametres:**
  - `prompt`: Texte d'entrée pour generer depuis
- **Cede:** Tokens/mots de texte comme ils sont generés

### Enchassemens

#### `embed(text_input: Union[str, List[str]]) -> np.ndarray`

Creer enchassemens deterministes pour entrée de texte.

```python
# Chaine simple
vec = steadytext.embed("Bonjour monde")

# Liste de chaines (moyenné)
vecs = steadytext.embed(["Bonjour", "monde"])
```

- **Parametres:**
  - `text_input`: Chaine ou liste de chaines à enchasser
- **Retourne:** Tableau numpy 1024-dimensionnel normalisé L2 (float32)

### Utilitaires

#### `preload_models(verbose: bool = False) -> None`

Precharger modeles avant premiere utilisation.

```python
steadytext.preload_models()  # Silencieux
steadytext.preload_models(verbose=True)  # Avec progres
```

#### `get_model_cache_dir() -> str`

Obtenir le chemin au repertoire de cache de modele.

```python
cache_dir = steadytext.get_model_cache_dir()
print(f"Modeles sont stockés en: {cache_dir}")
```

### Constantes

```python
steadytext.DEFAULT_SEED  # 42
steadytext.GENERATION_MAX_NEW_TOKENS  # 512
steadytext.EMBEDDING_DIMENSION  # 1024
```

---

## 🤝 Contribuer

Les contributions sont bienvenues!
Voyez [CONTRIBUTING.md](CONTRIBUTING.md) pour lignes directrices.

---

## 📄 Licence

* **Code:** MIT
* **Modeles:** MIT (Qwen3)

---

## 📈 Quoi de Neuf

### Generation Structurée (v2.4.1+)
- **Support de grammaire natif llama.cpp** pour contraintes JSON, regex, et choix
- **Integration d'extension PostgreSQL** - toutes caracteristiques de generation structurée en SQL
- **Fonctions de generation structurée asynchrone** pour applications haute performance

### Extension PostgreSQL (v1.1.0+)
- **Fonctions SQL prestes pour production** pour generation de texte et enchassemens
- **Operations asynchrones** avec traitement de fond basé sur queue
- **Fonctions d'aggregation de summarisation IA** avec support TimescaleDB
- **Generation structurée** en SQL (schemas JSON, patrons regex, choix)
- **Support Docker** pour deployment facile

### Reclassement de Document (v2.3.0+)
- **Support de reclassement** usant modele Qwen3-Reranker-4B pour score de pertinence requeste-document
- **API Python** - fonction `steadytext.rerank()` avec descriptions de tasche personnalisables
- **Commande CLI** - `st rerank` pour operations de reclassement en ligne de commande
- **Fonctions PostgreSQL** - fonctions SQL pour reclassement avec support asynchrone (Extension PostgreSQL v1.3.0+)
- **Score de repli** - chevauchement de mot simple quand modele non disponible
- **Cache dedié** - cache de frecence separé pour resultats de reclassement

### Architecture Daemon (v1.2.0+)
- **Service de modele persistant** avec ZeroMQ pour appels repetés 10-100x plus rapides
- **Repli automatique** au chargement direct de modele quand daemon non disponible
- **Zero configuration** - daemon commence automatiquement à premiere utilisation
- **Operation de fond** - daemon fonctionne silencieusement en fond

### Système de Cache Centralisé
- **Cache unifié** - comportement consistent entre daemon et acces direct
- **Backend SQLite thread-safe** pour acces concurrent fiable
- **Fichiers de cache partagés** à travers tous modes d'acces
- **Integration de cache** avec serveur daemon pour performance optimale

### Experience CLI Ameliorée
- **Flux par defaut** - voir sortie comme elle est generée
- **Tranquille par defaut** - sortie propre sans messages informationnels
- **Nouvelle syntaxe de pipe** - `echo "requeste" | st` pour meilleure integration unix
- **Gestion de daemon** - commandes integrées pour cycle de vie de daemon

---

## 🔧 Depannage

### Problemes d'Installation

#### Erreurs de Construction llama-cpp-python

Si vous rencontrez erreurs de construction reliées à llama-cpp-python, specialement avec l'erreur "Eschec de charger modele", c'est probablement dû au paquet requerant la fourche inference-sh avec drapeaux CMAKE specifiques:

```bash
# Establir variables d'environnement requises avant installation
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAVA_BUILD=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DGGML_CUDA=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF"

# Puis installer
pip install steadytext

# Ou installer depuis source
git clone https://github.com/julep-ai/steadytext.git
cd steadytext
uv sync  # ou pip install -e .
```

#### Problemes de Chargement de Modele

Si vous voyez erreurs "Eschec de charger modele depuis fichier":

1. **Essayer modeles de repli**: Establir `STEADYTEXT_USE_FALLBACK_MODEL=true`
2. **Effacer cache de modele**: `rm -rf ~/.cache/steadytext/models/`
3. **Verifier espace disque**: Modeles requierent ~2-4GB par modele

### Problemes Communs

- **"Nul module nommé 'llama_cpp'"**: Reinstaller avec drapeaux CMAKE ci-dessus
- **Connexion daemon refusée**: Verifier si daemon fonctionne avec `st daemon status`
- **Premiere execution lente**: Modeles telechargent à premiere utilisation (~2-4GB)

---

Construit avec ❤️ pour developpeurs las de tests IA incertains.