#!/usr/bin/env python3
"""
Galvanotechnik Ontologie Generator
Liest Begriffe aus einer Textdatei, fragt Definitionen über OpenAI API ab
und speichert die Antworten strukturiert.
"""

import os
import time
from pathlib import Path
from openai import OpenAI
from datetime import datetime


class GalvanotechnikOntologieGenerator:
    def __init__(self, api_key=None, model="gpt-4o", base_url=None):
        """
        Initialisiert den Generator
        
        Args:
            api_key: OpenAI API Key (falls None, wird OPENAI_API_KEY aus Umgebungsvariablen verwendet)
            model: Zu verwendendes Modell (Standard: gpt-4o)
            base_url: Custom API Endpoint URL (z.B. für Proxy-Server)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key erforderlich! Setze OPENAI_API_KEY Umgebungsvariable oder übergebe api_key Parameter.")
        
        # Client mit optionaler base_url initialisieren
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        
        self.model = model
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self):
        """Erstellt den optimierten System-Prompt"""
        return """Du bist ein Experte für Galvanotechnik und technische Normung mit Schwerpunkt auf DIN, EN, ISO und verwandten Standards.

Deine Aufgabe ist es, präzise Ontologieklassen-Beschreibungen für Fachbegriffe der Galvanotechnik zu erstellen.

QUALITÄTSKRITERIEN:
- Analysiere ausschließlich normative Quellen (DIN, EN, ISO, VDI, IEC, ASTM, ...) und anerkannte Fachpublikationen
- Priorisiere aktuelle, gültige Normversionen
- Bei mehreren Definitionen: harmonisiere zu einer konsistenten Kernaussage
- Formuliere neutral, präzise und allgemeinverständlich
- Vermeide Marketing-Sprache oder subjektive Bewertungen

AUSGABEFORMAT (strikt einhalten):
Begriff: [Exakter Begriff wie in der Anfrage]
Vorhanden in Normen: [Ja/Teilweise/Nein - mit kurzer Begründung wenn Nein/Teilweise]
Beschreibung: [2-3 Sätze. Erste Satz: Was ist es? Zweiter Satz: Hauptzweck/Funktion. Dritter Satz (optional): Besonderheiten/Abgrenzung.]
Quelle(n): [Vollständige Normbezeichnung mit Jahreszahl, z.B. "DIN EN ISO 2081:2018-09, DIN 50979:2022-01"]

WICHTIG:
- Wenn keine Norm verfügbar: nutze anerkannte Fachliteratur und kennzeichne dies
- Gib immer konkrete Quellen an, keine allgemeinen Verweise
- Die Beschreibung muss direkt in eine Ontologie-Definition übernommen werden können"""

    def _create_user_prompt(self, begriff):
        """Erstellt den User-Prompt für einen spezifischen Begriff"""
        return f"""Erstelle eine Ontologieklassen-Beschreibung für folgenden Begriff aus der Galvanotechnik:

{begriff}

Halte dich strikt an das vorgegebene Ausgabeformat."""

    def query_definition(self, begriff):
        """
        Fragt eine Definition für einen Begriff ab
        
        Args:
            begriff: Der zu definierende Begriff
            
        Returns:
            str: Die formatierte Antwort
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._create_user_prompt(begriff)}
                ],
                temperature=0.3,  # Niedrige Temperature für konsistentere Ergebnisse
                max_completion_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"FEHLER bei Begriff '{begriff}': {str(e)}"

    def process_file(self, input_file, output_file=None, delay=1.0):
        """
        Verarbeitet eine Datei mit Begriffen
        
        Args:
            input_file: Pfad zur Eingabedatei (ein Begriff pro Zeile)
            output_file: Pfad zur Ausgabedatei (Standard: input_file_ergebnisse_TIMESTAMP.txt)
            delay: Verzögerung zwischen API-Aufrufen in Sekunden (Rate Limiting)
        """
        # Eingabedatei lesen
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Eingabedatei nicht gefunden: {input_file}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            begriffe = [line.strip() for line in f if line.strip()]
        
        if not begriffe:
            raise ValueError("Keine Begriffe in der Eingabedatei gefunden!")
        
        # Ausgabedatei vorbereiten
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = input_path.parent / f"{input_path.stem}_ergebnisse_{timestamp}.txt"
        else:
            output_file = Path(output_file)
        
        # Header schreiben
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GALVANOTECHNIK ONTOLOGIE - AUTOMATISCH GENERIERTE DEFINITIONEN\n")
            f.write(f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
            f.write(f"Modell: {self.model}\n")
            f.write(f"Anzahl Begriffe: {len(begriffe)}\n")
            f.write("=" * 80 + "\n\n")
        
        # Begriffe verarbeiten
        print(f"Verarbeite {len(begriffe)} Begriffe...")
        print(f"Ausgabe: {output_file}")
        print("-" * 80)
        
        for i, begriff in enumerate(begriffe, 1):
            print(f"[{i}/{len(begriffe)}] Verarbeite: {begriff}")
            
            # API-Abfrage
            result = self.query_definition(begriff)
            
            # Ergebnis speichern
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(result + "\n")
                f.write("\n" + "-" * 80 + "\n\n")
            
            print(f"✓ Gespeichert\n")
            
            # Rate Limiting
            if i < len(begriffe):
                time.sleep(delay)
        
        print("=" * 80)
        print(f"✓ Fertig! Ergebnisse gespeichert in: {output_file}")
        
        return output_file


def main():
    """Hauptfunktion mit Beispielverwendung"""
    
    # Konfiguration
    INPUT_FILE = r"Begriffe.txt"  # Datei mit Begriffen (ein Begriff pro Zeile)
    API_KEY = "YOUR_API_KEY_HERE" # None = nutzt OPENAI_API_KEY Umgebungsvariable
    MODEL = "gpt-5.2"  # oder "gpt-4o-mini" für günstigere/schnellere Anfragen
    DELAY = 1.0  # Sekunden zwischen Anfragen
    
    # Für HS Offenburg Proxy:
    BASE_URL = "YOUR_PROXY_URL_HERE"  
    # Für direkte OpenAI API Nutzung:
    # BASE_URL = None
    
    try:
        # Generator initialisieren
        generator = GalvanotechnikOntologieGenerator(
            api_key=API_KEY,
            model=MODEL,
            base_url=BASE_URL
        )
        
        # Datei verarbeiten
        output_file = generator.process_file(
            input_file=INPUT_FILE,
            delay=DELAY
        )
        
        print(f"\n✓ Erfolgreich abgeschlossen!")
        print(f"  Ausgabedatei: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ FEHLER: {e}")
        print(f"\nErstelle zunächst eine Datei '{INPUT_FILE}' mit deinen Begriffen.")
        print("Beispiel-Inhalt:")
        print("  Galvanisieren")
        print("  Elektrolyt")
        print("  Anodenkorrektur")
        
    except ValueError as e:
        print(f"\n❌ FEHLER: {e}")
        print("\nSetze deinen OpenAI API Key:")
        print("  export OPENAI_API_KEY='dein-api-key-hier'")
        
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")


if __name__ == "__main__":
    main()