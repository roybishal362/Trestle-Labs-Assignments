import streamlit as st
import json
import base64
import os
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import io

# LangChain imports
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

# Document processing imports
import PyPDF2
from PIL import Image
import pytesseract
import fitz  # PyMuPDF for better PDF text extraction
import numpy as np  # For image preprocessing

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Document Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class ExtractedField(BaseModel):
    """Model for individual extracted fields"""
    name: str = Field(description="Name of the field")
    value: str = Field(description="Extracted value")
    confidence: float = Field(description="Confidence score between 0 and 1")
    source: Dict[str, Any] = Field(description="Source information including page and bounding box")

class QualityAssurance(BaseModel):
    """Model for quality assurance information"""
    passed_rules: List[str] = Field(description="List of validation rules that passed")
    failed_rules: List[str] = Field(description="List of validation rules that failed")
    notes: str = Field(description="Notes about missing or low-confidence fields")

class DocumentExtraction(BaseModel):
    """Main model for document extraction results"""
    doc_type: str = Field(description="Type of document: invoice, medical_bill, or prescription")
    fields: List[ExtractedField] = Field(description="List of extracted fields")
    overall_confidence: float = Field(description="Overall confidence score")
    qa: QualityAssurance = Field(description="Quality assurance information")

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from PDF file using PyMuPDF (fitz) with fallback to PyPDF2
    """
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        # Read PDF bytes
        pdf_bytes = pdf_file.read()
        
        # Try PyMuPDF first (better for complex PDFs)
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            
            if text.strip():
                return text
                
        except Exception as fitz_error:
            st.warning(f"PyMuPDF failed: {str(fitz_error)}, trying PyPDF2...")
        
        # Fallback to PyPDF2 if PyMuPDF fails
        pdf_file.seek(0)  # Reset file pointer
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text if text.strip() else "Could not extract text from PDF"
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_image(image_file) -> str:
    """
    Extract text from image using OCR (pytesseract) with enhanced compatibility
    """
    try:
        # Check if Tesseract is configured
        if not hasattr(pytesseract.pytesseract, 'tesseract_cmd') or not pytesseract.pytesseract.tesseract_cmd:
            # Try to configure it again
            if not configure_tesseract():
                return ""
        
        # Reset file pointer to beginning
        image_file.seek(0)
        
        # Open image with PIL
        image = Image.open(image_file)
        st.info(f"📷 Processing {image.format} image: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
        
        # Convert to RGB and ensure compatibility
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save image temporarily in a format that works better with older Tesseract
        import tempfile
        import io
        
        # Try different approaches for OCR
        ocr_attempts = []
        
        # Method 1: Direct OCR with simple config
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                # Save as JPEG to avoid PNG issues
                image.save(temp_file.name, 'JPEG', quality=95)
                text = pytesseract.image_to_string(temp_file.name, config='--psm 6')
                if text.strip():
                    ocr_attempts.append(("JPEG conversion", text.strip()))
        except Exception as e:
            st.warning(f"JPEG method failed: {str(e)}")
        
        # Method 2: Process as numpy array and save as BMP
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray_array = img_array
            
            # Convert back to PIL
            gray_image = Image.fromarray(gray_array.astype('uint8'), mode='L')
            
            # Save as BMP (most compatible format)
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as temp_file:
                gray_image.save(temp_file.name, 'BMP')
                text = pytesseract.image_to_string(temp_file.name, config='--psm 6')
                if text.strip():
                    ocr_attempts.append(("BMP grayscale", text.strip()))
        except Exception as e:
            st.warning(f"BMP method failed: {str(e)}")
        
        # Method 3: Enhanced contrast processing
        try:
            img_array = np.array(image.convert('L'))
            
            # Apply different enhancement techniques
            enhancements = [
                ("High contrast", np.where(img_array < 128, 0, 255)),
                ("Medium contrast", np.where(img_array < 100, 0, 255)),
                ("Adaptive", img_array)  # Keep original
            ]
            
            for enhancement_name, enhanced_array in enhancements:
                enhanced_image = Image.fromarray(enhanced_array.astype('uint8'), mode='L')
                
                with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as temp_file:
                    enhanced_image.save(temp_file.name, 'TIFF')
                    
                    # Try different PSM modes
                    for psm in [6, 8, 13, 7]:
                        try:
                            text = pytesseract.image_to_string(temp_file.name, config=f'--psm {psm}')
                            if text.strip() and len(text.strip()) > 10:  # At least some meaningful text
                                ocr_attempts.append((f"{enhancement_name} PSM-{psm}", text.strip()))
                                break
                        except:
                            continue
        except Exception as e:
            st.warning(f"Enhancement method failed: {str(e)}")
        
        # Clean up temp files
        try:
            import glob
            for temp_file in glob.glob(tempfile.gettempdir() + "/tmp*"):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        except:
            pass
        
        # Return the best OCR result
        if ocr_attempts:
            # Sort by text length (longer usually means better extraction)
            ocr_attempts.sort(key=lambda x: len(x[1]), reverse=True)
            best_method, best_text = ocr_attempts[0]
            
            st.success(f"✅ OCR successful using: {best_method}")
            st.info(f"📝 Extracted {len(best_text)} characters")
            
            # Show preview of extracted text
            with st.expander("👀 OCR Preview", expanded=False):
                st.text_area("Extracted Text (first 500 chars)", best_text[:500], height=100)
            
            return best_text
        
        # If all methods failed
        st.error("❌ All OCR methods failed")
        st.markdown("""
        **Possible solutions:**
        1. **Try a different image format** (JPG instead of PNG)
        2. **Increase image contrast** before uploading
        3. **Use a PDF version** of the document
        4. **Update Tesseract** to a newer version:
           - Download from: https://github.com/UB-Mannheim/tesseract/wiki
           - Install latest version (5.x recommended)
        """)
        
        return "OCR failed - please try a different image format or PDF"
    
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ **Image Processing Error**")
        st.code(error_msg)
        
        if "png" in error_msg.lower():
            st.warning("**PNG Compatibility Issue Detected**")
            st.markdown("""
            Your Tesseract version has PNG compatibility issues. **Quick fixes:**
            
            1. **Convert to JPG**: Save your image as JPG format instead of PNG
            2. **Use PDF**: Convert your document to PDF and upload that instead
            3. **Update Tesseract**: Install Tesseract 5.x for better image support
            """)
        
        return ""

def detect_document_type(text: str) -> str:
    """
    Simple rule-based document type detection
    In production, you might use a separate ML model for this
    """
    text_lower = text.lower()
    
    # Keywords for different document types
    invoice_keywords = ['invoice', 'bill', 'amount due', 'total amount', 'subtotal', 'tax']
    medical_keywords = ['patient', 'diagnosis', 'treatment', 'medical', 'hospital', 'clinic']
    prescription_keywords = ['prescription', 'medication', 'dosage', 'pharmacy', 'rx', 'drug']
    
    # Count keyword matches
    invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
    medical_score = sum(1 for keyword in medical_keywords if keyword in text_lower)
    prescription_score = sum(1 for keyword in prescription_keywords if keyword in text_lower)
    
    # Return document type with highest score
    if invoice_score >= medical_score and invoice_score >= prescription_score:
        return "invoice"
    elif medical_score >= prescription_score:
        return "medical_bill"
    else:
        return "prescription"

# ============================================================================
# LANGCHAIN + GROQ INTEGRATION
# ============================================================================

class DocumentExtractionChain:
    """
    Main class that handles LangChain integration with Groq
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize the chain with Groq API key
        """
        self.groq_api_key = groq_api_key
        
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="openai/gpt-oss-20b",  # Using openai/gpt-oss-20b model
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            # Initialize output parser
            self.output_parser = PydanticOutputParser(pydantic_object=DocumentExtraction)
            
            # Create extraction prompt template
            self.extraction_prompt = self._create_extraction_prompt()
            
            # Create LangChain chain
            self.chain = LLMChain(
                llm=self.llm,
                prompt=self.extraction_prompt
            )
            
        except Exception as e:
            st.error(f"Failed to initialize DocumentExtractionChain: {str(e)}")
            raise e
    
    def _create_extraction_prompt(self) -> PromptTemplate:
        """
        Create the prompt template for document extraction
        """
        template = """
        You are an expert document extraction AI. Your task is to extract structured information from documents.

        Document Type: {doc_type}
        Document Text: {text}

        Instructions:
        1. Extract relevant fields based on the document type
        2. For INVOICES: Extract company name, invoice number, date, total amount, items, tax, etc.
        3. For MEDICAL BILLS: Extract patient name, date of service, provider, diagnosis codes, charges, etc.
        4. For PRESCRIPTIONS: Extract patient name, medication name, dosage, pharmacy, doctor, date, etc.
        5. Assign confidence scores (0.0-1.0) based on how clear the information is
        6. If information is missing or unclear, use confidence score 0.0-0.5
        7. If information is clear and complete, use confidence score 0.7-1.0
        8. Create simple validation rules and check them

        {format_instructions}

        Your response:
        """
        
        return PromptTemplate(
            input_variables=["doc_type", "text"],
            template=template,
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
    
    def extract_information(self, text: str, doc_type: str) -> DocumentExtraction:
        """
        Extract structured information from document text
        """
        try:
            # Run the LangChain chain
            result = self.chain.run({
                "text": text[:4000],  # Limit text length for API
                "doc_type": doc_type
            })
            
            # Parse the result using output parser
            try:
                parsed_result = self.output_parser.parse(result)
                return parsed_result
            except Exception as parse_error:
                st.warning(f"Output parsing failed: {str(parse_error)}")
                # Try to extract JSON manually if parsing fails
                return self._manual_json_extraction(result, doc_type)
            
        except Exception as e:
            # Return fallback result if extraction fails
            st.error(f"Error during extraction: {str(e)}")
            return self._create_fallback_result(doc_type)
    
    def _manual_json_extraction(self, result: str, doc_type: str) -> DocumentExtraction:
        """
        Manual JSON extraction as fallback
        """
        try:
            # Try to find JSON in the result string
            import re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_data = json.loads(json_str)
                return DocumentExtraction(**json_data)
            else:
                return self._create_fallback_result(doc_type)
        except:
            return self._create_fallback_result(doc_type)
    
    def _create_fallback_result(self, doc_type: str) -> DocumentExtraction:
        """
        Create a fallback result if extraction fails
        """
        return DocumentExtraction(
            doc_type=doc_type,
            fields=[
                ExtractedField(
                    name="extraction_error",
                    value="Failed to extract information",
                    confidence=0.0,
                    source={"page": 1, "bbox": [0, 0, 0, 0]}
                )
            ],
            overall_confidence=0.0,
            qa=QualityAssurance(
                passed_rules=[],
                failed_rules=["extraction_failed"],
                notes="Document extraction failed. Please try again or check the document quality."
            )
        )

# ============================================================================
# STREAMLIT UI FUNCTIONS
# ============================================================================

def display_confidence_bar(confidence: float, label: str):
    """
    Display a confidence bar using Streamlit progress bar
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(confidence)
        st.write(f"**{label}**")
    with col2:
        color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        st.write(f"{color} {confidence:.1%}")

def display_extraction_results(extraction_result: DocumentExtraction):
    """
    Display the extraction results in a user-friendly format
    """
    st.header("📊 Extraction Results")
    
    # Document type and overall confidence
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Document Type", extraction_result.doc_type.replace("_", " ").title())
    with col2:
        confidence_color = "normal" if extraction_result.overall_confidence > 0.7 else "inverse"
        st.metric("Overall Confidence", f"{extraction_result.overall_confidence:.1%}")
    
    # Overall confidence meter
    st.subheader("Overall Confidence Score")
    display_confidence_bar(extraction_result.overall_confidence, "Document Extraction Quality")
    
    # Extracted fields
    st.subheader("📋 Extracted Fields")
    
    for field in extraction_result.fields:
        with st.expander(f"📝 {field.name.replace('_', ' ').title()}", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Value:** {field.value}")
                st.write(f"**Source:** Page {field.source.get('page', 'Unknown')}")
            with col2:
                display_confidence_bar(field.confidence, "Field Confidence")
    
    # Quality Assurance
    st.subheader("✅ Quality Assurance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Passed Rules:**")
        for rule in extraction_result.qa.passed_rules:
            st.write(f"✅ {rule}")
    
    with col2:
        st.write("**Failed Rules:**")
        for rule in extraction_result.qa.failed_rules:
            st.write(f"❌ {rule}")
    
    if extraction_result.qa.notes:
        st.info(f"**Notes:** {extraction_result.qa.notes}")

def create_download_button(extraction_result: DocumentExtraction):
    """
    Create a download button for the JSON results
    """
    # Convert to dictionary for JSON serialization
    result_dict = extraction_result.dict()
    
    # Create JSON string
    json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
    
    # Create download button
    st.download_button(
        label="📥 Download JSON Results",
        data=json_str,
        file_name=f"{extraction_result.doc_type}_extraction.json",
        mime="application/json"
    )

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

def configure_tesseract():
    """
    Configure Tesseract OCR path for different operating systems
    """
    import platform
    import os
    import subprocess
    import shutil
    
    # Try to find tesseract in system PATH first
    tesseract_cmd = shutil.which('tesseract')
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return True
    
    # If not in PATH, try common installation locations
    if platform.system() == "Windows":
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
            r'D:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
    
    # Try to run tesseract command directly to test if it's accessible
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    """
    Main Streamlit application function
    """
    # App header
    st.title("🤖 Smart Document Extractor")
    st.markdown("Extract structured data from invoices, medical bills, and prescriptions using AI")
    
    # Configure and test Tesseract
    tesseract_available = configure_tesseract()
    
    if not tesseract_available:
        st.error("⚠️ **Tesseract OCR Configuration Issue**")
        st.markdown("""
        **Tesseract is installed but Python can't access it. Try these solutions:**
        
        **Option 1: Add Tesseract to Windows PATH**
        1. Press `Win + R`, type `sysdm.cpl`, press Enter
        2. Click "Environment Variables"
        3. In "System Variables", find "Path" and click "Edit"
        4. Click "New" and add: `C:\\Program Files\\Tesseract-OCR`
        5. Click OK, restart your command prompt
        
        **Option 2: Manual Path Setting**
        Create a file called `tesseract_config.py` with:
        ```python
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        ```
        
        **Option 3: Use PDF files instead of images for now**
        """)
    else:
        st.success("✅ Tesseract OCR is configured and ready!")
        
        # Add OCR testing section
        with st.expander("🔧 Test OCR Configuration", expanded=False):
            try:
                import subprocess
                result = subprocess.run(['tesseract', '--version'], 
                                     capture_output=True, text=True)
                st.code(result.stdout)
                st.success("Tesseract is working correctly!")
            except Exception as e:
                st.error(f"Tesseract test failed: {e}")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Get your free API key from https://console.groq.com"
        )
        
        if not groq_api_key:
            st.warning("Please enter your Groq API key to continue")
            st.stop()
    
    # File uploader
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="Upload a PDF or image file (PNG, JPG)"
    )
    
    if uploaded_file is not None:
        # Display file information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
        
        # Extract text based on file type
        with st.spinner("🔍 Extracting text from document..."):
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_image(uploaded_file)
        
        if not extracted_text.strip():
            st.error("❌ Could not extract text from the document. Please check the file quality.")
            return
        
        # Show extracted text (optional preview)
        with st.expander("👀 View Extracted Text", expanded=False):
            st.text_area("Raw Text", extracted_text, height=200)
        
        # Detect document type
        doc_type = detect_document_type(extracted_text)
        st.success(f"🎯 Detected document type: **{doc_type.replace('_', ' ').title()}**")
        
        # Initialize extraction chain
        try:
            extraction_chain = DocumentExtractionChain(groq_api_key)
        except Exception as e:
            st.error(f"❌ Error initializing AI model: {str(e)}")
            return
        
        # Extract information
        if st.button("🚀 Extract Information", type="primary"):
            with st.spinner("🧠 AI is analyzing your document..."):
                extraction_result = extraction_chain.extract_information(extracted_text, doc_type)
            
            # Display results
            display_extraction_results(extraction_result)
            
            # Download button
            st.markdown("---")
            create_download_button(extraction_result)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a document to get started")
        
        # Sample instructionsF
        st.markdown("""
        ### How to use this app:
        1. **Get a Groq API key** from [console.groq.com](https://console.groq.com) (free)
        2. **Enter your API key** in the sidebar
        3. **Upload a document** (PDF or image)
        4. **Click "Extract Information"** to analyze the document
        5. **Review the results** and download the JSON output
        
        ### Supported Document Types:
        - 📧 **Invoices** - Company details, amounts, items, taxes
        - 🏥 **Medical Bills** - Patient info, services, charges, providers
        - 💊 **Prescriptions** - Medications, dosages, patient details, pharmacy
        """)

if __name__ == "__main__":
    main()