"""
Synchronous API for AutoPDFParse.
"""

from .parser import PDFParser
from .vision import VisionService

__all__ = ["VisionService", "PDFParser"]
