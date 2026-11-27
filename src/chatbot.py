import os
import pandas as pd
import faiss
import numpy as np
from datetime import datetime
import re
import unicodedata
import logging
from pathlib import Path

from src.config import Params, get_params
from src.llm import make_client
from src.pipeline.build_index import CHUNKS_FILE, INDEX_FILE

logger = logging.getLogger(__name__)

DATE_COLUMNS = ('firstdate_begin', 'firstdate_end', 'lastdate_begin', 'lastdate_end')
# Fuseau d'affichage des horaires (événements en région PACA).
DISPLAY_TZ = 'Europe/Paris'


def parse_dates(series: pd.Series) -> pd.Series:
    """Dates en UTC : chaînes ISO 8601 (pipeline) ou millisecondes epoch (ancien format)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit='ms', utc=True, errors='coerce')
    return pd.to_datetime(series, utc=True, errors='coerce', format='ISO8601')


def load_system_prompt(path) -> str:
    """Prompt système versionné (``paths.system_prompt``) ; placeholders {current_date} et {context}."""
    text = Path(path).read_text(encoding="utf-8")
    for placeholder in ("{current_date}", "{context}"):
        if placeholder not in text:
            raise ValueError(f"placeholder {placeholder} absent du prompt système {path}")
    return text


def strip_accents(text: str) -> str:
    """« Décembre » → « Decembre » : les utilisateurs tapent souvent sans accents."""
    decomposed = unicodedata.normalize('NFD', text)
    return ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')


def contains_term(text: str, term: str) -> bool:
    """Vrai si ``term`` apparaît comme mot entier (« var » absent de « variés »,
    « vence » absent de « provence », « mars » absent de « marseille »)."""
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text) is not None


def _local(value):
    """Horodatage converti en heure locale d'affichage (NaT conservé)."""
    ts = pd.to_datetime(value, utc=True, errors='coerce')
    return ts.tz_convert(DISPLAY_TZ) if pd.notna(ts) else ts


class EventChatbot:
    """Chatbot intelligent pour recommander des événements"""
    
    def __init__(self, index_path=None, chunks_path=None, params: Params | None = None):
        """Initialise le chatbot avec l'index FAISS et les métadonnées"""

        self.params = params or get_params()

        if index_path is None:
            index_path = self.params.paths.index_dir / INDEX_FILE
        if chunks_path is None:
            chunks_path = self.params.paths.index_dir / CHUNKS_FILE
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index FAISS introuvable : {index_path}\n"
                f"Lancez `uv run dvc repro` (ou `dvc pull`) pour générer l'index."
            )
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Fichier chunks introuvable : {chunks_path}\n"
                f"Lancez `uv run dvc repro` (ou `dvc pull`) pour générer les chunks."
            )
        
        self.index = faiss.read_index(str(index_path))
        # Identifiants et codes postaux restent du texte (zéro initial de 04, 05, 06).
        self.chunks_df = pd.read_json(chunks_path, dtype={'uid': str, 'location_postalcode': str})
        if self.index.ntotal != len(self.chunks_df):
            raise ValueError(
                f"Index et chunks désalignés : {self.index.ntotal} vecteurs pour {len(self.chunks_df)} chunks"
            )
        # La ligne i de l'index FAISS est le vecteur du chunk i.
        self.embeddings = self.index.reconstruct_n(0, self.index.ntotal) if self.index.ntotal else np.empty((0, self.index.d), np.float32)

        for col in DATE_COLUMNS:
            if col in self.chunks_df.columns:
                self.chunks_df[col] = parse_dates(self.chunks_df[col])
        
        current_date = pd.Timestamp.now(tz='UTC')
        future_events = self.chunks_df[self.chunks_df['lastdate_end'] >= current_date]

        self.available_cities = self._extract_available_cities()
        self.available_postal_codes = self._extract_available_postal_codes()

        logger.info("chatbot_loaded", extra={
            "index_path": str(index_path),
            "n_chunks": len(self.chunks_df),
            "n_future_events": int(future_events['uid'].nunique()),
            "n_cities": len(self.available_cities),
            "n_postal_codes": len(self.available_postal_codes),
        })

        self.client = make_client()
        # Tokens consommés depuis la création (embeddings des questions et génération).
        self.token_usage = {"embedding_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        
        self.system_prompt = load_system_prompt(self.params.paths.system_prompt)

    def vectorize_query(self, query):
        """Vectorise la requête utilisateur"""
        query_res = self.client.embeddings.create(
            model=self.params.embedding.model,
            input=[query]
        )
        usage = getattr(query_res, "usage", None)
        self.token_usage["embedding_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        return np.array([query_res.data[0].embedding], dtype=np.float32)
    
    def extract_temporal_filter(self, query):
        """Extrait les contraintes temporelles de la requête (accents ignorés : « decembre »)."""
        query_lower = strip_accents(query.lower())

        months_map = {
            'janvier': 1, 'fevrier': 2, 'mars': 3, 'avril': 4,
            'mai': 5, 'juin': 6, 'juillet': 7, 'aout': 8,
            'septembre': 9, 'octobre': 10, 'novembre': 11, 'decembre': 12
        }
        
        for month_name, month_num in months_map.items():
            # Mot entier : « mars » ne doit pas être trouvé dans « Marseille ».
            if contains_term(query_lower, month_name):
                today = datetime.now()
                year_match = re.search(r'\b(20\d{2})\b', query)
                if year_match:
                    year = int(year_match.group(1))
                else:
                    # Sans année, un mois déjà passé désigne l'année suivante.
                    year = today.year if month_num >= today.month else today.year + 1

                return {
                    'type': 'month',
                    'month': month_num,
                    'year': year
                }
        
        return None
    
    def _extract_available_cities(self):
        """Extrait toutes les villes uniques présentes dans les données"""
        if 'location_city' not in self.chunks_df.columns:
            return {}
        
        cities = self.chunks_df['location_city'].dropna().unique()
        normalized_cities = {}
        
        for city in cities:
            normalized = unicodedata.normalize('NFD', city.lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            normalized_cities[normalized] = city
        
        return normalized_cities
    
    def _extract_available_postal_codes(self):
        """Extrait tous les codes postaux uniques présents dans les données"""
        if 'location_postalcode' not in self.chunks_df.columns:
            return set()
        
        postal_codes = self.chunks_df['location_postalcode'].dropna().unique()
        return set(str(code) for code in postal_codes)

    def extract_location_filter(self, query):
        """
        Extrait les contraintes géographiques de la requête.

        FIX #5 : Ajout de la reconnaissance par département (ex: "Vaucluse" → 84xxx,
        "Var" → 83xxx). Sans ce mapping, une requête "Vaucluse" ne filtrait pas
        les codes postaux 84xxx et remontait des événements hors département.
        """
        query_normalized = unicodedata.normalize('NFD', query.lower())
        query_normalized = ''.join(c for c in query_normalized if unicodedata.category(c) != 'Mn')

        # Mapping département → préfixe de code postal
        DEPARTMENT_MAP = {
            'vaucluse': '84',
            'bouches-du-rhone': '13',
            'bouches du rhone': '13',
            'var': '83',
            'alpes-de-haute-provence': '04',
            'alpes de haute provence': '04',
            'hautes-alpes': '05',
            'hautes alpes': '05',
            'alpes-maritimes': '06',
            'alpes maritimes': '06',
        }

        # 1. Chercher un département dans la requête
        for dept_normalized, dept_prefix in DEPARTMENT_MAP.items():
            dept_n = unicodedata.normalize('NFD', dept_normalized)
            dept_n = ''.join(c for c in dept_n if unicodedata.category(c) != 'Mn')
            if contains_term(query_normalized, dept_n):
                logger.debug("location_filter", extra={"type": "department", "value": dept_prefix})
                return {'type': 'department_prefix', 'value': dept_prefix}

        # 2. Chercher une ville dans la requête (noms les plus longs d'abord :
        #    « aix-en-provence » avant « aix »)
        for normalized_city, original_city in sorted(
            self.available_cities.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if contains_term(query_normalized, normalized_city):
                logger.debug("location_filter", extra={"type": "city", "value": original_city})
                return {'type': 'city', 'value': original_city}

        # 3. Chercher un code postal explicite
        postal_match = re.search(r'\b(\d{5})\b', query)
        if postal_match:
            postal_code = postal_match.group(1)
            if postal_code in self.available_postal_codes:
                logger.debug("location_filter", extra={"type": "postal", "value": postal_code})
                return {'type': 'postal', 'value': postal_code}
            else:
                logger.debug("postal_code_unknown", extra={"value": postal_code})

        return None

    def _apply_temporal_filter(self, df, temporal_filter):
        """Applique le filtre temporel sur un DataFrame."""
        month = temporal_filter['month']
        year = temporal_filter['year']
        logger.debug("temporal_filter", extra={"month": month, "year": year})
        
        start_of_month = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
        end_of_month = (
            pd.Timestamp(year=year + 1, month=1, day=1, tz='UTC')
            if month == 12
            else pd.Timestamp(year=year, month=month + 1, day=1, tz='UTC')
        )
        
        mask = (
            (df['firstdate_begin'] < end_of_month) & 
            (df['lastdate_end'] >= start_of_month)
        )
        return df[mask]

    def _apply_location_filter(self, df, location_filter):
        """
        Applique le filtre géographique sur un DataFrame.

        FIX #5 : Gestion du type 'department_prefix' pour filtrer par préfixe
        de code postal (ex: '84' filtre tous les codes 84000-84999).
        """

        if location_filter['type'] == 'city':
            return df[
                df['location_city'].str.contains(
                    location_filter['value'], case=False, na=False
                )
            ]
        elif location_filter['type'] == 'postal':
            return df[df['location_postalcode'] == location_filter['value']]
        elif location_filter['type'] == 'department_prefix':
            prefix = location_filter['value']
            postal_str = df['location_postalcode'].astype(str).str.zfill(5)
            return df[postal_str.str.startswith(prefix)]

        return df

    def _semantic_rank(self, df, query_embedding, k):
        """
        FIX #1 : Classement sémantique sur un sous-ensemble filtré.
        Au lieu de prendre les k premiers arbitrairement, on reconstruit
        un index FAISS temporaire pour classer par pertinence.
        """
        if df.empty:
            return pd.DataFrame(), np.array([])
        
        filtered_indices = df.index.tolist()
        
        # Construire l'index temporaire
        filtered_embeddings = self.embeddings[filtered_indices]
        
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
        
        search_k = min(k * self.params.retrieval.candidate_factor, len(df))
        distances, local_indices = temp_index.search(query_embedding, search_k)
        
        global_indices = [filtered_indices[i] for i in local_indices[0]]
        result_chunks = self.chunks_df.loc[global_indices].copy()
        result_chunks['distance'] = distances[0]
        
        # Déduplication par UID en gardant le meilleur score
        unique_events = []
        seen_uids = set()
        
        for _, row in result_chunks.sort_values('distance').iterrows():
            uid = row['uid']
            if uid not in seen_uids:
                seen_uids.add(uid)
                event_chunks = result_chunks[result_chunks['uid'] == uid]
                unique_events.append(event_chunks)
                if len(seen_uids) >= k:
                    break
        
        if not unique_events:
            return pd.DataFrame(), np.array([])
        
        result_df = pd.concat(unique_events, ignore_index=True)
        final_distances = result_df['distance'].values
        result_df = result_df.drop('distance', axis=1)
        
        return result_df, final_distances

    def search_relevant_events(self, query, k=None):
        """
        Recherche les événements les plus pertinents.

        k : nombre d'événements distincts (défaut : retrieval.k de params.yaml).
        """
        if k is None:
            k = self.params.retrieval.k
        temporal_filter = self.extract_temporal_filter(query)
        location_filter = self.extract_location_filter(query)
        
        # Partir des événements futurs uniquement
        current_date = pd.Timestamp.now(tz='UTC')
        filtered_df = self.chunks_df[self.chunks_df['lastdate_end'] >= current_date].copy()
        
        # FIX #1 : Appliquer les deux filtres de façon cumulative (et non exclusive)
        if temporal_filter and temporal_filter['type'] == 'month':
            filtered_df = self._apply_temporal_filter(filtered_df, temporal_filter)
        
        if location_filter:
            filtered_df = self._apply_location_filter(filtered_df, location_filter)
        
        unique_uids = filtered_df['uid'].unique()
        logger.debug("events_after_filters", extra={"n_events": len(unique_uids)})
        
        if len(unique_uids) == 0:
            return pd.DataFrame(), np.array([])
        
        # FIX #2 : Toujours classer sémantiquement, même après filtrage
        query_embedding = self.vectorize_query(query)
        return self._semantic_rank(filtered_df, query_embedding, k)
    
    def format_context(self, chunks_df):
        """Formate les chunks en contexte lisible avec métadonnées temporelles"""
        context_parts = []
        
        if chunks_df.empty:
            return "Aucun événement trouvé."
        
        if 'uid' not in chunks_df.columns:
            logger.error("missing_uid_column", extra={"columns": chunks_df.columns.tolist()})
            return "Erreur : colonne 'uid' manquante dans les données."
        
        grouped = chunks_df.groupby('uid')
        
        for uid, group in grouped:
            event_info = f"=== ÉVÉNEMENT {uid} ==="
            first_row = group.iloc[0]
            
            # FIX #6 : Distinguer événement ponctuel / multi-jours pour éviter
            # que le LLM confonde firstdate_begin et lastdate_end comme une seule date.
            date_info = []
            # Horaires affichés en heure de Paris (stockés en UTC pour les filtres).
            fdb = _local(first_row.get('firstdate_begin')) if 'firstdate_begin' in first_row else None
            fde = _local(first_row.get('firstdate_end')) if 'firstdate_end' in first_row else None
            ldb = _local(first_row.get('lastdate_begin')) if 'lastdate_begin' in first_row else None
            lde = _local(first_row.get('lastdate_end')) if 'lastdate_end' in first_row else None

            is_multiday = (
                fdb is not None and lde is not None and pd.notna(fdb) and pd.notna(lde)
                and fdb.date() != lde.date()
            )

            if is_multiday:
                # Événement étalé sur plusieurs jours : rendre la structure explicite
                date_info.append(f"⚠️ ÉVÉNEMENT MULTI-JOURS")
                if pd.notna(fdb):
                    date_info.append(f"Jour de début: {fdb.strftime('%d/%m/%Y')} à {fdb.strftime('%H:%M')}")
                if pd.notna(lde):
                    date_info.append(f"Jour de fin: {lde.strftime('%d/%m/%Y')} à {lde.strftime('%H:%M')}")
                date_info.append(f"⚠️ NE PAS CONFONDRE la date de début et la date de fin")
            else:
                if pd.notna(fdb):
                    date_info.append(f"Date début: {fdb.strftime('%d/%m/%Y %H:%M')}")
                if pd.notna(fde):
                    date_info.append(f"Date fin: {fde.strftime('%d/%m/%Y %H:%M')}")
                if ldb is not None and pd.notna(ldb) and ldb != fdb:
                    date_info.append(f"Dernière occurrence début: {ldb.strftime('%d/%m/%Y %H:%M')}")
                if lde is not None and pd.notna(lde) and lde != fde:
                    date_info.append(f"Dernière occurrence fin: {lde.strftime('%d/%m/%Y %H:%M')}")

            location_info = []
            if 'location_name' in first_row and pd.notna(first_row['location_name']):
                location_info.append(f"Lieu: {first_row['location_name']}")
            if 'location_city' in first_row and pd.notna(first_row['location_city']):
                location_info.append(f"Ville: {first_row['location_city']}")
            if 'location_postalcode' in first_row and pd.notna(first_row['location_postalcode']):
                location_info.append(f"Code postal: {first_row['location_postalcode']}")
            if 'location_department' in first_row and pd.notna(first_row['location_department']):
                location_info.append(f"Département: {first_row['location_department']}")
            if 'location_phone' in first_row and pd.notna(first_row['location_phone']):
                location_info.append(f"Téléphone: {first_row['location_phone']}")
            if 'location_website' in first_row and pd.notna(first_row['location_website']):
                location_info.append(f"Site web: {first_row['location_website']}")
            
            if 'chunk_id' in group.columns:
                full_description = " ".join(group.sort_values('chunk_id')['text'].tolist())
            else:
                full_description = " ".join(group['text'].tolist())
            
            context_block = [event_info]
            if date_info:
                context_block.append("DATES: " + " | ".join(date_info))
            if location_info:
                context_block.append("LOCALISATION: " + " | ".join(location_info))
            context_block.append(f"DESCRIPTION:\n{full_description}")
            
            context_parts.append("\n".join(context_block))
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)

        
    def generate_response(self, query, context):
        """Génère une réponse augmentée avec le modèle de génération"""
        current_date = datetime.now().strftime("%d/%m/%Y")
        full_prompt = self.system_prompt.format(
            current_date=current_date,
            context=context
        )
        
        generation = self.params.generation
        response = self.client.chat.completions.create(
            model=generation.model,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": query}
            ],
            temperature=generation.temperature,
            max_completion_tokens=generation.max_tokens
        )
        
        usage = getattr(response, "usage", None)
        self.token_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.token_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        return response.choices[0].message.content
    
    def chat(self, user_query, k=None):
        """Recherche les événements pertinents et génère la réponse (aucun affichage).

        La question n'est jamais journalisée en clair (CLAUDE.md, B7).
        """
        relevant_chunks, distances = self.search_relevant_events(user_query, k=k)
        context = self.format_context(relevant_chunks)
        response = self.generate_response(user_query, context)
        return {
            "response": response,
            "sources": relevant_chunks,
            "distances": distances
        }

    @staticmethod
    def print_answer(result, show_sources=True):
        """Affiche une réponse et ses sources dans le terminal (mode interactif)."""
        response = result["response"]
        relevant_chunks = result["sources"]
        distances = result["distances"]

        print("\n" + "="*80)
        print("RÉPONSE DU CHATBOT")
        print("="*80)
        print(response)
        
        if show_sources and len(relevant_chunks) > 0:
            print("\n" + "="*80)
            print("SOURCES UTILISÉES")
            print("="*80)
            
            distance_map = dict(zip(relevant_chunks.index, distances)) if len(distances) > 0 else {}
            grouped_sources = relevant_chunks.groupby('uid')
            
            for idx, (uid, group) in enumerate(grouped_sources, 1):
                first_chunk = group.iloc[0]
                
                if distance_map:
                    event_distances = [distance_map.get(i, 0.0) for i in group.index if i in distance_map]
                    avg_distance = np.mean(event_distances) if event_distances else 0.0
                else:
                    avg_distance = 0.0
                
                full_text = " ".join(group.sort_values('chunk_id')['text'].tolist())
                
                date_info = []
                if 'firstdate_begin' in first_chunk and pd.notna(first_chunk['firstdate_begin']):
                    date_info.append(f"Début: {pd.to_datetime(first_chunk['firstdate_begin'], utc=True).strftime('%d/%m/%Y')}")
                if 'lastdate_end' in first_chunk and pd.notna(first_chunk['lastdate_end']):
                    date_info.append(f"Fin: {pd.to_datetime(first_chunk['lastdate_end'], utc=True).strftime('%d/%m/%Y')}")
                
                print(f"\n📍 Source {idx} - Événement {uid} (distance: {avg_distance:.4f})")
                if date_info:
                    print(f"   📅 {' | '.join(date_info)}")
                print(f"   {full_text[:300]}...")
    
    def interactive_mode(self):
        """Mode interactif pour converser avec le chatbot"""
        print("\n" + "="*80)
        print("🎭 CHATBOT D'ÉVÉNEMENTS CULTURELS")
        print("="*80)
        print("Tapez 'quit' ou 'exit' pour quitter\n")
        
        while True:
            user_input = input("Vous : ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir !")
                break
            
            if not user_input:
                continue
            
            try:
                self.print_answer(self.chat(user_input))
            except Exception as e:
                print(f"\n❌ Erreur : {e}")
            
            print("\n" + "-"*80 + "\n")