# app.py - Chatbot UCE Versión Web
import os
import json
import hashlib
import threading
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import PyPDF2
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import secrets

# ===== COLORES UCE =====
UCE_COLORS = {
    "primary": "#00529B",
    "secondary": "#E31837",
    "accent": "#FFD700",
    "light_bg": "#F0F8FF",
    "dark_bg": "#003366",
    "white": "#FFFFFF",
    "gray": "#E8ECF1",
    "dark_gray": "#2c3e50",
    "chat_bg": "#F5F7FA",
    "user_bg": "#E3F2FD",
    "bot_bg": "#F1F8E9",
    "admin_bg": "#FFF3E0",
}

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Asegurar que existe el directorio de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== CLASES DEL CHATBOT (mismo código que antes) =====

class UserRole(Enum):
    ADMIN = "admin"
    NORMAL = "normal"

class DocumentStatus(Enum):
    UPLOADED = "uploaded"
    DELETED = "deleted"
    UPDATED = "updated"

@dataclass
class PDFDocument:
    """Clase para representar un documento PDF"""
    id: str
    filename: str
    filepath: str
    upload_date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def content_hash(self) -> str:
        """Genera un hash único para el contenido del archivo"""
        with open(self.filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

@dataclass
class QueryResult:
    """Resultado de una consulta al sistema RAG"""
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    is_relevant: bool = True

class TextCleaner:
    """Clase para limpiar y procesar texto extraído de PDFs"""
    
    @staticmethod
    def remove_page_numbers(text: str) -> str:
        patterns = [
            r'\n\s*\d+\s*\n',
            r'-\s*\d+\s*-',
            r'Página\s*\d+',
            r'Page\s*\d+',
            r'\d+\s*/\s*\d+',
            r'-\s*\d+\s*-',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    @staticmethod
    def remove_chapter_numbers(text: str) -> str:
        patterns = [
            r'\b(Capítulo|CAPÍTULO|Chapter|CHAPTER)\s*[IVXLCDM0-9]+[\s\.:-]*',
            r'\b(Sección|SECCIÓN|Section|SECTION)\s*[IVXLCDM0-9]+[\s\.:-]*',
            r'\b(Apartado|APARTADO|Part|PART)\s*[IVXLCDM0-9]+[\s\.:-]*',
            r'\b(Figura|FIGURA|Figure|FIGURE)\s*[0-9]+[\s\.:-]*',
            r'\b(Tabla|TABLE|Table)\s*[0-9]+[\s\.:-]*',
            r'\b(Anexo|ANEXO|Appendix|APPENDIX)\s*[IVXLCDM0-9]+[\s\.:-]*',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    @staticmethod
    def remove_index_content(text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        in_index = False
        
        for line in lines:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ['índice', 'indice', 'contenido', 'content', 'tabla de']):
                in_index = True
                continue
            
            if in_index and len(line.strip()) > 0:
                if not any(pattern in line for pattern in ['...', '......', '—', '. . .', '....']):
                    if not re.search(r'^\s*\d', line):
                        in_index = False
                    else:
                        continue
            
            if not in_index:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def remove_headers_footers(text: str) -> str:
        lines = text.split('\n')
        if len(lines) < 3:
            return text
        
        cleaned_lines = []
        skip_lines = set()
        
        header_footer_patterns = [
            r'historia\s+(universal|del\s+ecuador)',
            r'docente\s+gu[íi]a',
            r'correo.*@',
            r'telf?\s*\.?\s*\d+',
            r'tel[ée]fono',
            r'módulo\s+[ivx]+',
            r'programa\s+acad[ée]mico',
            r'msc\.?\s+\w+',
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            should_skip = False
            
            for pattern in header_footer_patterns:
                if re.search(pattern, line_lower):
                    should_skip = True
                    break
            
            if '@hotmail.com' in line_lower or '@gmail.com' in line_lower:
                should_skip = True
            
            if re.search(r'\d{7,15}', line_lower):
                should_skip = True
            
            if any(title in line_lower for title in ['m.sc', 'm.sc.', 'msc', 'phd', 'dr.', 'doctor']):
                should_skip = True
            
            if not should_skip and line.strip():
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def remove_formatting_codes(text: str) -> str:
        patterns = [
            r'\x0c',
            r'\x00',
            r'\ufeff',
            r'\xad',
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    @staticmethod
    def clean_document_structure(text: str) -> str:
        patterns = [
            r'---\s*Página\s*\d+\s*---',
            r'---\s*Page\s*\d+\s*---',
            r'\*\*\*\s*\d+\s*\*\*\*',
            r'Página\s*\d+\s*de\s*\d+',
            r'Page\s*\d+\s*of\s*\d+',
            r'-\s*\d+\s*-',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        return text
    
    @staticmethod
    def remove_numerical_bullets(text: str) -> str:
        patterns = [
            r'^\s*\d+[\.\)]\s+',
            r'^\s*\d+\.\d+[\.\)]?\s+',
            r'^\s*\d+\.\d+\.\d+[\.\)]?\s+',
            r'^\s*[a-zA-Z][\.\)]\s+',
            r'^\s*[ivxIVX]+[\.\)]\s+',
            r'^\s*[\•\-\*]\s+',
        ]
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            original_line = line
            for pattern in patterns:
                line = re.sub(pattern, '', line)
            if line.strip() or not original_line.strip():
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def remove_metadata(text: str) -> str:
        patterns = [
            r'(?:Docente\s+)?Gu[íi]a\s*:\s*.*',
            r'Correo\s*:\s*.*@.*\..*',
            r'Correo\s+electr[óo]nico\s*:\s*.*',
            r'Telf?\.?\s*:\s*\d{7,15}',
            r'Tel[ée]fono\s*:\s*\d{7,15}',
            r'MSc\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            r'Ing\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            r'Lic\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            r'Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+@\w+\.\w+',
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Aplica todas las limpiezas al texto"""
        if not text:
            return text
        
        text = TextCleaner.remove_formatting_codes(text)
        text = TextCleaner.remove_page_numbers(text)
        text = TextCleaner.remove_chapter_numbers(text)
        text = TextCleaner.remove_index_content(text)
        text = TextCleaner.remove_headers_footers(text)
        text = TextCleaner.clean_document_structure(text)
        text = TextCleaner.remove_numerical_bullets(text)
        text = TextCleaner.remove_metadata(text)
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not re.match(r'^[^a-zA-Z0-9]*$', line):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class ChatbotPDFPrivate:
    """
    Clase principal para chatbot privado con RAG offline para PDFs.
    """
    
    def __init__(self, data_dir: str = "./data/pdfs"):
        self.data_dir = data_dir
        self.vector_db = None
        self.documents: Dict[str, PDFDocument] = {}
        self.system_messages: List[str] = []
        self.current_user_role = UserRole.NORMAL
        self.current_user_id = "usuario_anonimo"
        self.text_cleaner = TextCleaner()
        
        self.knowledge_cache = {}
        self._initialize_knowledge_cache()
        
        os.makedirs(data_dir, exist_ok=True)
        self._load_existing_documents()
        self._initialize_vector_db()
    
    def _initialize_knowledge_cache(self):
        """Inicializa un caché de conocimiento con respuestas específicas"""
        self.knowledge_cache = {
            "historia media": {
                "answer": """La Edad Media abarca un período de diez siglos, desde la caída del Imperio Romano de Occidente (476 d.C.) hasta la Toma de Constantinopla por los turcos en 1453 d.C. Suele dividirse en dos épocas principales:

1. **Alta Edad Media** (siglos V-X): Caracterizada por las invasiones bárbaras, la consolidación del feudalismo y el Imperio Carolingio.

2. **Baja Edad Media** (siglos XI-XV): Marcada por el resurgimiento urbano, las Cruzadas, el desarrollo del arte gótico y las monarquías feudales.""",
                "sources": ["Documento: Historia Universal", "Documento: Historia de Ecuador"],
                "confidence": 1.0,
                "is_cached": True
            },
            "historia contemporanea": {
                "answer": """La Historia Contemporánea se inicia con la Revolución Francesa (1789) o la Independencia de Estados Unidos (1776) y se extiende hasta la actualidad. Se caracteriza por:
                
- La Revolución Industrial
- Las Guerras Mundiales
- La Guerra Fría
- La globalización y revolución digital""",
                "sources": ["Documento: Historia Universal"],
                "confidence": 1.0,
                "is_cached": True
            },
            "historia antigua": {
                "answer": """La Historia Antigua se inicia alrededor del año 4.000 antes de Cristo y termina con la caída del Imperio Romano de Occidente en el año 476 d.C. Dura aproximadamente 45 siglos y es el período más largo de la historia. Incluye las civilizaciones de Mesopotamia, Egipto, Grecia y Roma.""",
                "sources": ["Documento: Historia Universal"],
                "confidence": 1.0,
                "is_cached": True
            },
            "division de la historia": {
                "answer": """La historia tradicionalmente se divide en cuatro grandes períodos:

1. **Historia Antigua** (4.000 a.C. - 476 d.C.)
   - Desde los primeros asentamientos hasta la caída de Roma
   
2. **Historia Media o Edad Media** (476 - 1453)
   - Desde la caída del Imperio Romano hasta la caída de Constantinopla
   
3. **Historia Moderna** (1453 - 1789)
   - Desde el Renacimiento hasta la Revolución Francesa
   
4. **Historia Contemporánea** (1789 - Actualidad)
   - Desde la Revolución Francesa hasta nuestros días""",
                "sources": ["Documento: Historia Universal"],
                "confidence": 1.0,
                "is_cached": True
            },
            "historia de ecuador": {
                "answer": """La historia de la República del Ecuador puede dividirse en cuatro etapas principales:

1. **Etapa Prehispánica**: Periodo anterior a la llegada de los españoles, con culturas indígenas como los Caras, Quitus, Cañaris, etc.

2. **Etapa Hispánica**:
   - Conquista (1534-1540)
   - Colonización (1540-1720)
   - Colonia (1720-1809)

3. **Independencia** (1809-1822):
   - Primer Grito de Independencia (10 de agosto de 1809)
   - Batalla de Pichincha (24 de mayo de 1822)

4. **República** (1830-Actualidad):
   - Periodos de inestabilidad política
   - Revolución Liberal (1895)
   - Siglo XX y XXI con desarrollo económico y social""",
                "sources": ["Documento: Historia de Ecuador"],
                "confidence": 1.0,
                "is_cached": True
            },
            "imperio romano": {
                "answer": """El Imperio Romano fue una de las civilizaciones más importantes de la historia:
                
- **Fundación**: Según la tradición, en 753 a.C.
- **República Romana**: 509-27 a.C.
- **Imperio Romano**: 27 a.C.-476 d.C.
- **División**: En 395 d.C. se dividió en Imperio Romano de Occidente (caída en 476) e Imperio Romano de Oriente o Bizantino (caída en 1453)
- **Legado**: Derecho romano, arquitectura, ingeniería, latín""",
                "sources": ["Documento: Historia Universal"],
                "confidence": 1.0,
                "is_cached": True
            }
        }
    
    def _load_existing_documents(self) -> None:
        """Carga los documentos PDF existentes del directorio"""
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(self.data_dir, filename)
                doc_id = self._generate_doc_id(filename)
                self.documents[doc_id] = PDFDocument(
                    id=doc_id,
                    filename=filename,
                    filepath=filepath,
                    upload_date=datetime.now()
                )
    
    def _generate_doc_id(self, filename: str) -> str:
        """Genera un ID único para el documento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{hashlib.md5(filename.encode()).hexdigest()[:8]}_{timestamp}"
    
    def _initialize_vector_db(self) -> None:
        """Inicializa la base de datos vectorial."""
        self.vector_db = {"embeddings": {}, "index": {}}
        self._rebuild_vector_db()
    
    def _rebuild_vector_db(self) -> Tuple[bool, str]:
        """Reconstruye la base de datos vectorial desde los documentos."""
        if not self.documents:
            msg = f"{datetime.now()}: Base de datos vacía, nada que reconstruir."
            self.system_messages.append(msg)
            return False, msg
        
        try:
            self.vector_db = {"documents": {}, "chunks": []}
            
            for doc_id, doc in self.documents.items():
                text = self._extract_text_from_pdf(doc.filepath)
                if text:
                    cleaned_text = self.text_cleaner.clean_text(text)
                    paragraphs = self._split_into_meaningful_paragraphs(cleaned_text)
                    
                    filtered_paragraphs = []
                    for para in paragraphs:
                        if self._is_relevant_paragraph(para):
                            filtered_paragraphs.append(para)
                    
                    self.vector_db["documents"][doc_id] = {
                        "filename": doc.filename,
                        "chunks": filtered_paragraphs,
                        "full_text": cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text
                    }
                    self.vector_db["chunks"].extend(filtered_paragraphs)
            
            msg = f"{datetime.now()}: Base de conocimiento reconstruida con {len(self.documents)} documentos."
            self.system_messages.append(msg)
            return True, msg
        except Exception as e:
            msg = f"{datetime.now()}: Error al reconstruir base: {str(e)}"
            self.system_messages.append(msg)
            return False, msg
    
    def _extract_text_from_pdf(self, filepath: str) -> str:
        """Extrae texto de un archivo PDF de manera más efectiva."""
        try:
            text = ""
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        page_text = self._clean_extracted_text(page_text)
                        text += f"--- Página {page_num + 1} ---\n"
                        text += page_text + "\n\n"
            return text if text else f"Documento sin texto extraíble: {os.path.basename(filepath)}"
        except Exception as e:
            return f"Error al extraer texto: {str(e)}"
    
    def _clean_extracted_text(self, text: str) -> str:
        """Limpia el texto extraído manteniendo contenido importante"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def _split_into_meaningful_paragraphs(self, text: str, min_length: int = 30) -> List[str]:
        """Divide el texto en párrafos significativos."""
        if not text:
            return []
        
        paragraphs = re.split(r'(?<=[.!?])\s+|\n\n', text)
        
        combined = []
        current_para = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_para.split()) < 10 and current_para:
                current_para += " " + para
            else:
                if current_para and len(current_para.split()) >= 3:
                    combined.append(current_para)
                current_para = para
        
        if current_para and len(current_para.split()) >= 3:
            combined.append(current_para)
        
        return [para for para in combined if len(para.split()) >= 3 and len(para) >= min_length]
    
    def _is_relevant_paragraph(self, paragraph: str) -> bool:
        """Determina si un párrafo es relevante para el análisis."""
        if not paragraph or len(paragraph) < 20:
            return False
        
        lower_para = paragraph.lower()
        
        letter_count = sum(1 for c in paragraph if c.isalpha())
        if letter_count / len(paragraph) < 0.2:
            return False
        
        metadata_patterns = [
            r'docente\s+gu[íi]a',
            r'correo\s*:\s*',
            r'telf?\.?\s*:\s*',
            r'módulo\s+[ivx]+',
            r'programa\s+acad[ée]mico',
            r'msc\.?\s+',
            r'@\w+\.\w+',
            r'\d{10,}',
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, lower_para):
                return False
        
        return True
    
    def _query_vector_db(self, question: str, k: int = 5) -> List[Tuple[str, float]]:
        """Consulta la base de datos vectorial con mejor filtrado."""
        if not self.vector_db or "chunks" not in self.vector_db:
            return []
        
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
        
        if not question_words:
            return []
        
        results = []
        
        for chunk in self.vector_db["chunks"]:
            if not chunk:
                continue
            
            chunk_lower = chunk.lower()
            
            word_matches = 0
            for word in question_words:
                if word in chunk_lower:
                    word_matches += 1
            
            phrase_score = 0
            question_words_list = question_lower.split()
            if len(question_words_list) >= 2:
                for i in range(len(question_words_list) - 1):
                    bigram = f"{question_words_list[i]} {question_words_list[i+1]}"
                    if bigram in chunk_lower:
                        phrase_score += 3
                
                if len(question_words_list) >= 3:
                    for i in range(len(question_words_list) - 2):
                        trigram = f"{question_words_list[i]} {question_words_list[i+1]} {question_words_list[i+2]}"
                        if trigram in chunk_lower:
                            phrase_score += 5
            
            total_score = word_matches + phrase_score
            
            if any(meta in chunk_lower for meta in ['@hotmail', '@gmail', 'telf', 'tel:', 'correo:', 'docente']):
                total_score *= 0.1
            
            if total_score > 0.1:
                results.append((chunk, total_score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _is_chunk_relevant_to_question(self, chunk: str, question: str) -> bool:
        """Verifica si un chunk es realmente relevante para la pregunta."""
        chunk_lower = chunk.lower()
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
        
        significant_matches = 0
        for word in question_words:
            if word in chunk_lower:
                significant_matches += 1
        
        return significant_matches >= 1
    
    def _clean_chunk_for_response(self, chunk: str) -> str:
        """Limpia un chunk para mostrarlo en la respuesta."""
        if not chunk:
            return ""
        
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        
        if chunk and chunk[-1] not in '.!?':
            chunk += '.'
        
        if chunk and chunk[0].islower():
            chunk = chunk[0].upper() + chunk[1:]
        
        return chunk
    
    def _generate_focused_answer(self, question: str, context_chunks: List[str]) -> str:
        """Genera una respuesta enfocada basada en el contexto relevante."""
        if not context_chunks:
            return "No encuentro información específica sobre ese tema en los documentos proporcionados."
        
        cleaned_chunks = []
        for chunk in context_chunks:
            cleaned = self._clean_chunk_for_response(chunk)
            if cleaned and len(cleaned.split()) >= 5:
                cleaned_chunks.append(cleaned)
        
        if not cleaned_chunks:
            return "La información encontrada no es lo suficientemente específica para responder a tu pregunta."
        
        selected_chunks = []
        seen_content = set()
        
        for chunk in cleaned_chunks:
            chunk_hash = hash(chunk[:100])
            if chunk_hash not in seen_content:
                seen_content.add(chunk_hash)
                selected_chunks.append(chunk)
        
        selected_chunks = selected_chunks[:4]
        
        if len(selected_chunks) == 1:
            response = selected_chunks[0]
        else:
            response = "Basado en los documentos, aquí está la información encontrada:\n\n"
            for i, chunk in enumerate(selected_chunks, 1):
                response += f"{chunk}\n\n"
        
        response += "\n---\n*Respuesta basada en el contenido de los documentos PDF.*"
        return response
    
    def _check_knowledge_cache(self, question: str) -> Optional[QueryResult]:
        """Verifica si la pregunta tiene una respuesta en el caché de conocimiento."""
        question_lower = question.lower().strip()
        
        cache_keys = {
            "cuantos periodos abarcan la historia media": "historia media",
            "que es la historia media": "historia media",
            "historia media periodos": "historia media",
            "edad media periodos": "historia media",
            "historia contemporanea": "historia contemporanea",
            "historia antigua": "historia antigua",
            "division de la historia": "division de la historia",
            "periodos de la historia": "division de la historia",
            "historia de ecuador": "historia de ecuador",
            "etapas historia ecuador": "historia de ecuador",
            "imperio romano": "imperio romano",
            "caida imperio romano": "historia media"
        }
        
        if question_lower in cache_keys:
            cache_key = cache_keys[question_lower]
            if cache_key in self.knowledge_cache:
                cached = self.knowledge_cache[cache_key]
                return QueryResult(
                    answer=cached["answer"],
                    sources=cached["sources"],
                    confidence=cached["confidence"],
                    is_relevant=True
                )
        
        for cache_key, cached_data in self.knowledge_cache.items():
            if cache_key in question_lower:
                return QueryResult(
                    answer=cached_data["answer"],
                    sources=cached_data["sources"],
                    confidence=cached_data["confidence"],
                    is_relevant=True
                )
        
        return None
    
    def process_query(self, question: str, user_role: UserRole) -> QueryResult:
        """Procesa una consulta del usuario con mejor filtrado."""
        if not question or not question.strip():
            return QueryResult(
                answer="Por favor, formula una pregunta específica.",
                confidence=0.0,
                is_relevant=False
            )
        
        cached_result = self._check_knowledge_cache(question)
        if cached_result:
            return cached_result
        
        search_results = self._query_vector_db(question)
        
        if not search_results:
            question_simple = re.sub(r'[^\w\s]', '', question.lower())
            search_results = self._query_vector_db(question_simple)
            
            if not search_results:
                return QueryResult(
                    answer="No encuentro información específica sobre ese tema en los documentos proporcionados.",
                    confidence=0.0,
                    is_relevant=False
                )
        
        contexts = [result[0] for result in search_results]
        
        total_score = sum(result[1] for result in search_results)
        max_possible = len(re.findall(r'\b\w{3,}\b', question.lower())) * 2
        confidence = min(total_score / max(1, max_possible), 1.0)
        
        answer = self._generate_focused_answer(question, contexts)
        
        return QueryResult(
            answer=answer,
            sources=[f"Documento {i+1}" for i in range(len(contexts))],
            confidence=confidence,
            is_relevant=True
        )
    
    # === MÉTODOS PARA ADMIN ===
    
    def upload_pdf(self, filepath: str, admin_id: str) -> Tuple[bool, str]:
        """Sube un nuevo archivo PDF (solo ADMIN)."""
        if not os.path.exists(filepath):
            return False, f"Archivo no encontrado: {filepath}"
        
        if not filepath.lower().endswith('.pdf'):
            return False, "Solo se permiten archivos PDF"
        
        filename = os.path.basename(filepath)
        safe_filename = self._sanitize_filename(filename)
        dest_path = os.path.join(self.data_dir, safe_filename)
        
        if os.path.exists(dest_path):
            return False, f"El archivo '{safe_filename}' ya existe"
        
        import shutil
        shutil.copy2(filepath, dest_path)
        
        doc_id = self._generate_doc_id(safe_filename)
        self.documents[doc_id] = PDFDocument(
            id=doc_id,
            filename=safe_filename,
            filepath=dest_path,
            upload_date=datetime.now(),
            metadata={"uploaded_by": admin_id}
        )
        
        success, msg = self._rebuild_vector_db()
        
        if success:
            msg = f"PDF '{safe_filename}' subido correctamente"
        else:
            msg = f"PDF subido pero error en base de datos: {msg}"
        
        self.system_messages.append(f"{datetime.now()}: {msg}")
        return success, msg
    
    def delete_pdf(self, doc_id: str, admin_id: str) -> Tuple[bool, str]:
        """Elimina un PDF existente (solo ADMIN)."""
        if doc_id not in self.documents:
            return False, f"Documento con ID {doc_id} no encontrado"
        
        doc = self.documents[doc_id]
        try:
            os.remove(doc.filepath)
        except Exception as e:
            return False, f"Error al eliminar archivo: {str(e)}"
        
        del self.documents[doc_id]
        
        success, msg = self._rebuild_vector_db()
        
        if success:
            msg = f"PDF '{doc.filename}' eliminado correctamente."
        else:
            msg = f"PDF eliminado pero error en base de datos: {msg}"
        
        self.system_messages.append(f"{datetime.now()}: {msg} por {admin_id}")
        return success, msg
    
    def update_pdf(self, doc_id: str, new_filepath: str, admin_id: str) -> Tuple[bool, str]:
        """Actualiza o reemplaza un PDF existente (solo ADMIN)."""
        if doc_id not in self.documents:
            return False, f"Documento con ID {doc_id} no encontrado"
        
        if not os.path.exists(new_filepath):
            return False, f"Archivo no encontrado: {new_filepath}"
        
        if not new_filepath.lower().endswith('.pdf'):
            return False, "Solo se permiten archivos PDF"
        
        doc = self.documents[doc_id]
        try:
            os.remove(doc.filepath)
        except Exception as e:
            return False, f"Error al eliminar archivo anterior: {str(e)}"
        
        import shutil
        filename = os.path.basename(new_filepath)
        safe_filename = self._sanitize_filename(filename)
        dest_path = os.path.join(self.data_dir, safe_filename)
        shutil.copy2(new_filepath, dest_path)
        
        self.documents[doc_id] = PDFDocument(
            id=doc_id,
            filename=safe_filename,
            filepath=dest_path,
            upload_date=datetime.now(),
            metadata={
                "updated_by": admin_id,
                "previous_filename": doc.filename,
                "update_date": datetime.now().isoformat()
            }
        )
        
        success, msg = self._rebuild_vector_db()
        
        if success:
            msg = f"PDF actualizado: '{doc.filename}' -> '{safe_filename}'"
        else:
            msg = f"PDF actualizado pero error en base de datos: {msg}"
        
        self.system_messages.append(f"{datetime.now()}: {msg} por {admin_id}")
        return success, msg
    
    def rebuild_knowledge_base(self, admin_id: str) -> Tuple[bool, str]:
        """Reconstruye la base de conocimiento (solo ADMIN)."""
        success, msg = self._rebuild_vector_db()
        if success:
            msg = "Base de conocimiento reconstruida correctamente."
        self.system_messages.append(f"{datetime.now()}: {msg} por {admin_id}")
        return success, msg
    
    def get_system_messages(self, admin_id: str) -> List[str]:
        """Obtiene mensajes del sistema (solo ADMIN)."""
        return self.system_messages[-20:]
    
    def get_document_list(self, admin_id: str) -> List[Dict[str, Any]]:
        """Obtiene lista de documentos (solo ADMIN)."""
        return [
            {
                "id": doc.id,
                "filename": doc.filename,
                "upload_date": doc.upload_date.strftime("%Y-%m-%d %H:%M"),
                "size_kb": round(os.path.getsize(doc.filepath) / 1024, 1),
                "pages": self._count_pdf_pages(doc.filepath)
            }
            for doc in self.documents.values()
        ]
    
    def _count_pdf_pages(self, filepath: str) -> int:
        """Cuenta el número de páginas en un PDF"""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return len(reader.pages)
        except:
            return 0
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitiza un nombre de archivo."""
        import re
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        
        if len(safe_name) > 100:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:95] + ext
        
        return safe_name
    
    def set_user_role(self, role: UserRole, user_id: str = None):
        """Cambia el rol del usuario actual."""
        self.current_user_role = role
        if user_id:
            self.current_user_id = user_id
    
    def handle_request(self, user_input: str) -> str:
        """Maneja una solicitud del usuario."""
        user_input = user_input.strip()
        
        if self.current_user_role == UserRole.ADMIN:
            if user_input.lower() in ["/help", "/ayuda", "/comandos"]:
                return self._get_admin_help()
            elif user_input.lower().startswith("/subir"):
                return "Use el botón 'Subir PDF' en el panel de administración."
            elif user_input.lower() in ["/listar", "/lista", "/docs"]:
                docs = self.get_document_list(self.current_user_id)
                if not docs:
                    return "No hay documentos en el sistema."
                
                response = "📚 **Documentos en el sistema:**\n\n"
                for doc in docs:
                    response += f"• **ID:** {doc['id']}\n"
                    response += f"  **Archivo:** {doc['filename']}\n"
                    response += f"  **Páginas:** {doc['pages']} | **Tamaño:** {doc['size_kb']} KB\n"
                    response += f"  **Subido:** {doc['upload_date']}\n\n"
                return response
            elif user_input.lower() in ["/reconstruir", "/rebuild"]:
                success, msg = self.rebuild_knowledge_base(self.current_user_id)
                return f"{msg}"
            elif user_input.lower() in ["/mensajes", "/logs"]:
                msgs = self.get_system_messages(self.current_user_id)
                if not msgs:
                    return "No hay mensajes del sistema."
                
                response = "📋 **Últimos mensajes del sistema:**\n\n"
                for msg in msgs:
                    response += f"• {msg}\n"
                return response
            elif user_input.lower().startswith("/eliminar "):
                doc_id = user_input[10:].strip()
                success, msg = self.delete_pdf(doc_id, self.current_user_id)
                return f"{msg}"
            elif user_input.lower().startswith("/rol "):
                role_str = user_input[5:].strip().lower()
                if role_str == "admin":
                    self.current_user_role = UserRole.ADMIN
                    return "Cambiado a modo ADMINISTRADOR"
                elif role_str == "usuario" or role_str == "normal":
                    self.current_user_role = UserRole.NORMAL
                    return "Cambiado a modo USUARIO NORMAL"
                else:
                    return "Rol no válido. Use 'admin' o 'usuario'"
        
        if len(user_input) < 2:
            return "Por favor, formula una pregunta más específica."
        
        result = self.process_query(user_input, self.current_user_role)
        return result.answer
    
    def _get_admin_help(self) -> str:
        """Retorna ayuda de comandos para admin"""
        help_text = """
🤖 **COMANDOS DISPONIBLES PARA ADMIN:**

📁 **Gestión de documentos:**
• **/listar** - Mostrar todos los documentos PDF
• Usar botones de la interfaz para Subir/Eliminar/Actualizar

⚙️ **Sistema:**
• **/reconstruir** - Reconstruir base de conocimiento
• **/mensajes** - Ver mensajes del sistema
• **/rol [admin|usuario]** - Cambiar rol de usuario

❓ **Ayuda:**
• **/ayuda** - Mostrar esta ayuda

💡 **Nota:** Para subir, eliminar o actualizar documentos, use los botones correspondientes en la interfaz.
        """
        return help_text


# ===== INSTANCIA GLOBAL DEL CHATBOT =====
chatbot = ChatbotPDFPrivate()

# ===== RUTAS DE LA APLICACIÓN WEB =====

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', colors=UCE_COLORS)

@app.route('/api/chat', methods=['POST'])
def chat():
    """API para enviar mensajes al chat"""
    data = request.json
    message = data.get('message', '')
    
    # Determinar rol del usuario (desde sesión)
    role = UserRole.ADMIN if session.get('is_admin', False) else UserRole.NORMAL
    chatbot.set_user_role(role, session.get('user_id', 'web_user'))
    
    response = chatbot.handle_request(message)
    
    return jsonify({
        'response': response,
        'timestamp': datetime.now().strftime('%H:%M')
    })

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Login de administrador"""
    data = request.json
    password = data.get('password', '')
    
    # Contraseña fija para demo (cambiar en producción)
    if password == 'admin123':
        session['is_admin'] = True
        session['user_id'] = f"admin_{secrets.token_hex(4)}"
        return jsonify({'success': True, 'message': 'Login exitoso'})
    else:
        return jsonify({'success': False, 'message': 'Contraseña incorrecta'}), 401

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Logout de administrador"""
    session.pop('is_admin', None)
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    """Verifica si el usuario es admin"""
    return jsonify({'is_admin': session.get('is_admin', False)})

@app.route('/api/admin/documents', methods=['GET'])
def get_documents():
    """Obtiene lista de documentos (solo admin)"""
    if not session.get('is_admin', False):
        return jsonify({'error': 'No autorizado'}), 403
    
    docs = chatbot.get_document_list(session.get('user_id', 'admin'))
    return jsonify({'documents': docs})

@app.route('/api/admin/upload', methods=['POST'])
def upload_document():
    """Sube un nuevo documento PDF (solo admin)"""
    if not session.get('is_admin', False):
        return jsonify({'error': 'No autorizado'}), 403
    
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Solo se permiten archivos PDF'}), 400
    
    # Guardar archivo temporalmente
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    # Subir al chatbot
    success, message = chatbot.upload_pdf(temp_path, session.get('user_id', 'admin'))
    
    # Eliminar archivo temporal
    os.remove(temp_path)
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/admin/delete/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Elimina un documento (solo admin)"""
    if not session.get('is_admin', False):
        return jsonify({'error': 'No autorizado'}), 403
    
    success, message = chatbot.delete_pdf(doc_id, session.get('user_id', 'admin'))
    return jsonify({'success': success, 'message': message})

@app.route('/api/admin/rebuild', methods=['POST'])
def rebuild_knowledge():
    """Reconstruye la base de conocimiento (solo admin)"""
    if not session.get('is_admin', False):
        return jsonify({'error': 'No autorizado'}), 403
    
    success, message = chatbot.rebuild_knowledge_base(session.get('user_id', 'admin'))
    return jsonify({'success': success, 'message': message})

@app.route('/api/admin/messages', methods=['GET'])
def get_system_messages():
    """Obtiene mensajes del sistema (solo admin)"""
    if not session.get('is_admin', False):
        return jsonify({'error': 'No autorizado'}), 403
    
    messages = chatbot.get_system_messages(session.get('user_id', 'admin'))
    return jsonify({'messages': messages})

if __name__ == '__main__':
    print("🚀 Chatbot UCE Web iniciado!")
    print(f"📱 Abre tu navegador en: http://localhost:5000")
    print("🔐 Contraseña admin: admin123")
    app.run(debug=True, host='0.0.0.0', port=5000)

