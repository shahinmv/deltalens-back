"""
Excel Azerbaijani Translation Script using LangChain

Required packages:
pip install pandas openpyxl langchain openai

This script translates English comments to Azerbaijani using LangChain and OpenAI API.
It reads from columns B (requirements), C (context), and E (English comments),
then populates column G with Azerbaijani translations.
"""

import os
import pandas as pd
from openpyxl import load_workbook
import time
from typing import Optional

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class ExcelTranslator:
    def __init__(self, 
                 excel_file_path: str,
                 sheet_name: str = None,
                 openai_api_key: str = None):
        """
        Initialize the translator with Excel file and OpenAI credentials
        
        Args:
            excel_file_path: Path to your Excel file
            sheet_name: Name of the sheet (if None, uses the first sheet)
            openai_api_key: OpenAI API key
        """
        self.excel_file_path = excel_file_path
        self.sheet_name = sheet_name
        
        # Initialize LangChain ChatOpenAI
        self.llm = self._init_langchain_llm(openai_api_key)
        
        # Verify file exists
        if not os.path.exists(excel_file_path):
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
    
    def _init_langchain_llm(self, api_key: str):
        """Initialize LangChain ChatOpenAI"""
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        elif not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        return ChatOpenAI(
            model_name="gpt-4o",  # or "gpt-4" for better quality
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=500
        )
    
    def read_excel_data(self) -> pd.DataFrame:
        """Read data from Excel file"""
        try:
            if self.sheet_name:
                df = pd.read_excel(self.excel_file_path, sheet_name=self.sheet_name, keep_default_na=False)
            else:
                df = pd.read_excel(self.excel_file_path, keep_default_na=False)
            
            # Replace NaN values with empty strings to handle multi-line cells properly
            df = df.fillna('')
            
            print(f"Successfully loaded Excel file with {len(df)} rows")
            return df
        
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise
    
    def save_excel_data(self, df: pd.DataFrame):
        """Save data back to Excel file"""
        try:
            # Create a backup of the original file
            backup_path = self.excel_file_path.replace('.xlsx', '_backup.xlsx')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(self.excel_file_path, backup_path)
                print(f"Created backup: {backup_path}")
            
            # Save the updated dataframe with proper formatting for multi-line cells
            with pd.ExcelWriter(self.excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                if self.sheet_name:
                    df.to_excel(writer, sheet_name=self.sheet_name, index=False)
                else:
                    df.to_excel(writer, index=False)
                
                # Get the worksheet to set text wrapping
                if self.sheet_name:
                    worksheet = writer.sheets[self.sheet_name]
                else:
                    worksheet = writer.sheets[list(writer.sheets.keys())[0]]
                
                # Enable text wrapping for all cells to handle multi-line content
                from openpyxl.styles import Alignment
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            print(f"Successfully saved updated data to {self.excel_file_path}")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            raise
    
    def translate_to_azerbaijani(self, 
                                english_text: str, 
                                context: str = "",
                                requirement: str = "") -> str:
        """
        Translate English text to Azerbaijani using OpenAI API
        
        Args:
            english_text: Text to translate
            context: Additional context from column C
            requirement: Requirement from column B for better context
        """
        if pd.isna(english_text) or str(english_text).strip() == "":
            return ""
        
        try:
            # Create a comprehensive prompt for better translation
            context_info = f"Context: {context}" if not pd.isna(context) and str(context).strip() else "Context: General comment"
            requirement_info = f"Requirement: {requirement}" if not pd.isna(requirement) and str(requirement).strip() else "Requirement: Not specified"
            
            prompt = f"""
Translate the following English comment to Azerbaijani. 

{context_info}
{requirement_info}
English Comment: {english_text}

Please provide a natural, accurate Azerbaijani translation that maintains the original meaning and tone. Consider the context and requirement when translating.

Azerbaijani Translation:"""

            messages = [
                SystemMessage(content="You are a professional translator specializing in English to Azerbaijani translation. Provide accurate, natural translations that consider context and maintain professional tone."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            translation = response.content.strip()
            
            # Remove any prefixes like "Azerbaijani Translation:" if present
            if ":" in translation:
                translation = translation.split(":", 1)[-1].strip()
            
            return translation
            
        except Exception as e:
            print(f"Error translating text '{str(english_text)[:50]}...': {e}")
            return f"Translation error: {str(e)}"
    
    def process_translations(self, start_row: int = 1, end_row: Optional[int] = None):
        """
        Main function to process translations
        
        Args:
            start_row: Starting row number (0-indexed, default: 1 for second row)
            end_row: Ending row number (if None, processes all available rows)
        """
        print("Starting translation process...")
        
        # Read Excel data
        df = self.read_excel_data()
        
        # Check if required columns exist
        required_columns = ['B', 'C', 'E']  # We'll add column G
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # If columns don't exist with letter names, try to use positional indexing
        if missing_columns:
            print("Column letters not found, using positional indexing...")
            # Assuming standard Excel column order: A=0, B=1, C=2, D=3, E=4, F=5, G=6
            if len(df.columns) < 5:  # Need at least 5 columns (up to E)
                raise ValueError("Excel file must have at least 5 columns (A through E)")
            
            # Rename columns to match our expected structure
            df.columns = [chr(65 + i) for i in range(len(df.columns))]  # A, B, C, D, E, F, G, ...
        
        # Ensure column G exists
        if 'G' not in df.columns:
            df['G'] = ""
        
        # Determine row range
        if end_row is None:
            end_row = len(df) - 1
        
        if end_row < start_row:
            print("No data to process")
            return
        
        print(f"Processing rows {start_row + 1} to {end_row + 1} (Excel row numbers)")
        
        # Process translations
        total_translated = 0
        
        for i in range(start_row, min(end_row + 1, len(df))):
            current_row_num = i + 1  # Excel row number (1-indexed)
            print(f"Processing row {current_row_num}")
            
            # Extract data
            requirement = df.loc[i, 'B'] if 'B' in df.columns else ""
            context = df.loc[i, 'C'] if 'C' in df.columns else ""
            english_comment = df.loc[i, 'E'] if 'E' in df.columns else ""
            
            # Skip if no English comment to translate
            if pd.isna(english_comment) or str(english_comment).strip() == "":
                print(f"  Skipping row {current_row_num}: No English comment")
                continue
            
            # Skip if already translated (optional - comment out if you want to retranslate)
            existing_translation = df.loc[i, 'G'] if 'G' in df.columns else ""
            if not pd.isna(existing_translation) and str(existing_translation).strip() != "":
                print(f"  Skipping row {current_row_num}: Already translated")
                continue
            
            # Translate to Azerbaijani
            print(f"  Translating: {str(english_comment)[:50]}...")
            azerbaijani_translation = self.translate_to_azerbaijani(
                english_comment, 
                context, 
                requirement
            )
            
            # Update the dataframe
            df.loc[i, 'G'] = azerbaijani_translation
            total_translated += 1
            
            print(f"  Translation: {azerbaijani_translation[:50]}...")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Save the updated Excel file
        if total_translated > 0:
            self.save_excel_data(df)
            print(f"Successfully translated {total_translated} comments!")
        else:
            print("No new translations were made.")
        
        print("Translation process completed!")
    
    def preview_translation(self, row_number: int):
        """
        Preview translation for a specific row (1-indexed Excel row number)
        """
        df = self.read_excel_data()
        
        # Convert to 0-indexed
        row_index = row_number - 1
        
        if row_index < 0 or row_index >= len(df):
            print(f"Row {row_number} is out of range. File has {len(df)} rows.")
            return
        
        # Extract data
        requirement = df.loc[row_index, 'B'] if 'B' in df.columns else ""
        context = df.loc[row_index, 'C'] if 'C' in df.columns else ""
        english_comment = df.loc[row_index, 'E'] if 'E' in df.columns else ""
        existing_translation = df.loc[row_index, 'G'] if 'G' in df.columns else ""
        
        print(f"Row {row_number} Preview:")
        print(f"Requirement (B): {requirement}")
        print(f"Context (C): {context}")
        print(f"English Comment (E): {english_comment}")
        print(f"Existing Translation (G): {existing_translation}")
        
        if not pd.isna(english_comment) and str(english_comment).strip() != "":
            print("\nGenerating new translation...")
            translation = self.translate_to_azerbaijani(english_comment, context, requirement)
            print(f"New Azerbaijani Translation: {translation}")
        else:
            print("No English comment to translate")
    
    def show_file_info(self):
        """Show information about the Excel file"""
        try:
            df = self.read_excel_data()
            print(f"\nFile Information:")
            print(f"File: {self.excel_file_path}")
            print(f"Sheet: {self.sheet_name or 'Default sheet'}")
            print(f"Total rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            
            # Show sample data
            print(f"\nFirst few rows:")
            print(df.head())
            
            # Count existing translations
            if 'G' in df.columns:
                existing_translations = df['G'].notna().sum()
                print(f"\nExisting translations in column G: {existing_translations}")
            
        except Exception as e:
            print(f"Error reading file info: {e}")


def main():
    """
    Main function to run the translation script using LangChain
    """
    # Configuration
    EXCEL_FILE_PATH = "technical_evaluation_updated.xlsx"  # Replace with your Excel file path
    SHEET_NAME = "3. Qeyri-funksional tələblər"  # Replace with your sheet name if needed, or None for first sheet
    OPENAI_API_KEY = "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"  # Your OpenAI API key (or set as environment variable)
    
    # You can also set environment variable instead:
    # set OPENAI_API_KEY=your_api_key_here (Windows)
    # export OPENAI_API_KEY="your_api_key_here" (Mac/Linux)
    # Then pass None for openai_api_key parameter
    
    try:
        # Initialize translator
        translator = ExcelTranslator(
            excel_file_path=EXCEL_FILE_PATH,
            sheet_name=SHEET_NAME,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Show file information
        translator.show_file_info()
        
        # Preview a specific row (optional)
        # translator.preview_translation(2)
        
        # Process all translations starting from row 2 (Excel row numbering)
        translator.process_translations(start_row=1)  # 0-indexed, so 1 = Excel row 2
        
        # Or process a specific range
        # translator.process_translations(start_row=1, end_row=10)  # Excel rows 2-11
        
    except Exception as e:
        print(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()